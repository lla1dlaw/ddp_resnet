
import torch
import torch.nn as nn
import torchvision.models as models
import time
import warnings

def benchmark(model, input_tensor, model_name, num_iterations=100, warmup_iterations=10):
    """
    Runs a benchmark for a given model.

    Args:
        model (nn.Module): The model to benchmark.
        input_tensor (torch.Tensor): A sample input tensor for the model.
        model_name (str): The name of the model for printing results.
        num_iterations (int): The number of iterations to run the benchmark for.
        warmup_iterations (int): The number of initial iterations to discard from timing.

    Returns:
        float: The total time taken for the benchmarked iterations in seconds.
    """
    # Define a simple loss function and optimizer for a realistic training step
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # Create a dummy target tensor
    target = torch.randint(0, 1000, (input_tensor.size(0),)).to(input_tensor.device)

    # Warmup phase
    # This is important to exclude one-time setup costs (like CUDA context creation)
    # from our benchmark measurements.
    print(f"[{model_name}] Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    # Ensure all GPU operations are finished before starting the timer
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    # Benchmark phase
    print(f"[{model_name}] Running {num_iterations} benchmark iterations...")
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    
    # Ensure all GPU operations are finished before stopping the timer
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    print(f"[{model_name}] Completed in {total_time:.4f} seconds.")
    return total_time

def run_benchmark():
    """
    Sets up and runs the benchmark for a non-compiled vs. compiled model.
    """
    # --- Configuration ---
    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_CLASSES = 1000
    
    # Check for PyTorch version and CUDA availability
    print(f"PyTorch Version: {torch.__version__}")
    if not torch.__version__.startswith("2."):
        warnings.warn("torch.compile is a feature of PyTorch 2.0+. This script may not work as expected.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    if device == "cpu":
        warnings.warn("Running on CPU. The performance difference from torch.compile may be less significant than on a GPU.")

    # --- Model and Data Setup ---
    # Create random input data to simulate a batch of images
    # The model expects a 4D tensor: (batch_size, channels, height, width)
    random_data = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Instantiate two identical models
    # We use pretrained=False because we don't need the actual weights for a runtime benchmark
    non_compiled_model = models.resnet50(weights=None).to(device)
    compiled_model = models.resnet50(weights=None).to(device)
    
    # Set models to training mode
    non_compiled_model.train()
    compiled_model.train()

    # --- Compilation Step ---
    print("Compiling the model with torch.compile()... (This may take a moment)")
    try:
        # The 'inductor' backend is the default and generally the fastest
        compiled_model = torch.compile(compiled_model, backend="inductor", mode='max-autotune')
        print("Model compiled successfully.\n")
    except Exception as e:
        print(f"Could not compile model: {e}")
        return

    # --- Run Benchmarks ---
    print("--- Starting Benchmark ---")
    
    # Benchmark the original, non-compiled model
    non_compiled_time = benchmark(non_compiled_model, random_data, "Non-Compiled")
    
    print("-" * 30)

    # Benchmark the compiled model
    compiled_time = benchmark(compiled_model, random_data, "Compiled")

    # --- Results ---
    print("\n--- Benchmark Results ---")
    print(f"Time for Non-Compiled Model: {non_compiled_time:.4f} seconds")
    print(f"Time for Compiled Model:     {compiled_time:.4f} seconds")
    
    if compiled_time > 0:
        speedup = (non_compiled_time - compiled_time) / non_compiled_time * 100
        print(f"\nSpeedup from torch.compile(): {speedup:.2f}%")
    else:
        print("\nCould not calculate speedup due to zero compiled time.")


if __name__ == "__main__":
    run_benchmark()
