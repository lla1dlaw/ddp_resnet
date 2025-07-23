from models import ComplexResNet, RealResNet 

def count_parameters(model):
    """Counts the number of learnable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # --- Configuration ---
    # These values should match the defaults you expect in your models.
    INPUT_CHANNELS = 4
    NUM_CLASSES = 7
    # This is the default activation for the complex network.
    # You can change this to 'zrelu', 'modrelu', etc. if needed.
    ACTIVATION_FUNCTION = 'crelu' 
    ARCH_SIZES = ['WS', 'DN', 'IB']

    # --- Print Header ---
    print("="*65)
    print("           Learnable Parameter Comparison (Original Models)")
    print("="*65)
    print(f"{'Architecture':<15} | {'ComplexResNet Parameters':>25} | {'RealResNet Parameters':>20}")
    print("-"*65)

    # --- Loop through architectures, initialize models, and print results ---
    for arch in ARCH_SIZES:
        # Instantiate the Complex Model using the class from your file
        complex_model = ComplexResNet(
            architecture_type=arch,
            activation_function=ACTIVATION_FUNCTION,
            input_channels=INPUT_CHANNELS,
            num_classes=NUM_CLASSES
        )
        complex_params = count_parameters(complex_model)

        # Instantiate the Real Model using the class from your file
        real_model = RealResNet(
            architecture_type=arch,
            input_channels=INPUT_CHANNELS,
            num_classes=NUM_CLASSES
        )
        real_params = count_parameters(real_model)

        # Print the formatted row with the results
        print(f"{arch:<15} | {complex_params:>25,} | {real_params:>20,}")

    print("="*65)

