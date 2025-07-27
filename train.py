import pretty_errors
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # For dummy data
from torch.distributed import init_process_group, destroy_process_group
from models import RealResNet, ComplexResNet
from Trainer import Trainer
from Datasets import get_dataloaders
import pandas as pd
import glob


def summarize_trials(trial_data_dir: str):
    """
    Reads all trial_*.csv files, calculates statistics, and saves two summary files:
    1. summary_by_epoch.csv: Mean and std for each metric at each epoch.
    2. final_summary.csv: Mean and std of the best epoch from each trial.
    """
    rank = os.environ["LOCAL_RANK"]
    trial_files = glob.glob(os.path.join(trial_data_dir, "trial_*.csv"))
    if not trial_files and rank == 0:
        print(f"No trial data found in {trial_data_dir}. Skipping summary.")
        return

    # FIXED: Re-implemented loop to add the 'trial' column to each DataFrame
    all_trials_df = []
    for i, file in enumerate(trial_files):
        df = pd.read_csv(file)
        df['trial'] = i
        all_trials_df.append(df)

    combined_df = pd.concat(all_trials_df, ignore_index=True)
    print(f"Column Headers: {combined_df.columns.tolist()}")

    metric_cols = [col for col in combined_df.columns if col not in ['epoch', 'trial']]

    summary_by_epoch = combined_df.groupby('epoch')[metric_cols].agg(['mean', 'std'])
    summary_by_epoch.columns = ['_'.join(col).strip() for col in summary_by_epoch.columns.values]
    
    epoch_summary_path = os.path.join(trial_data_dir, "summary_by_epoch.csv")
    summary_by_epoch.to_csv(epoch_summary_path)
    if rank == 0:
        print(f"\nSummary by epoch saved to {epoch_summary_path}")

    best_epoch_indices = combined_df.loc[combined_df.groupby('trial')['val_acc'].idxmax()]
    final_summary_stats = best_epoch_indices[metric_cols].agg(['mean', 'std'])
    
    final_summary_path = os.path.join(trial_data_dir, "final_summary.csv")
    final_summary_stats.to_csv(final_summary_path)
    if rank == 0:
        print(f"Final performance summary saved to {final_summary_path}")
        print("\n--- Final Performance Report ---")
    final_acc = final_summary_stats.loc['mean', 'val_acc']
    final_acc_std = final_summary_stats.loc['std', 'val_acc']
    if rank == 0:
        print(f"Validation Accuracy: {final_acc:.4f} ± {final_acc_std:.4f}")

    test_metrics_file = os.path.join(trial_data_dir, "final_test_metrics.csv")
    if os.path.exists(test_metrics_file):
        test_df = pd.read_csv(test_metrics_file)
        test_summary = test_df.agg(['mean', 'std'])
        test_acc_mean = test_summary.loc['mean', 'test_acc']
        test_acc_std = test_summary.loc['std', 'test_acc']
        if rank == 0: 
            print(f"Test Accuracy:       {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    if rank == 0:
        print("--------------------------------")

def ddp_setup():
    """
    Sets up the distributed data parallel environment.
    Assumes that MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE are in the environment.
    """
    rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        print("- Configuring DDP...")
    init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

# --- NEW: Test function ---
def run_test(rank: int):
    """
    Runs a quick test of the full training and summary pipeline with dummy data.
    """
    print(f"--- RUNNING IN TEST MODE ON RANK {rank} ---")
    # Test parameters
    test_params = {
        'save_every': 1,
        'total_epochs': 2,
        'dataset_name': 'TEST_DATASET',
        'batch_size': 4,
        'model_type': 'real',
        'arch': 'WS',
        'activation': 'crelu',
        'num_trials': 2,
        'split': [0.8, 0.1, 0.1],
        'num_classes': 3,
        'num_channels': 2,
    }

    # Create dummy data on the correct device
    dummy_inputs = torch.randn(100, test_params['num_channels'], 32, 32, device=rank)
    dummy_labels = torch.randint(0, test_params['num_classes'], (100,), device=rank)
    dummy_dataset = TensorDataset(dummy_inputs, dummy_labels)
    
    # The dummy dataset needs these attributes for the Trainer
    dummy_dataset.classes = [f'class_{i}' for i in range(test_params['num_classes'])]
    dummy_dataset.num_classes = test_params['num_classes']
    dummy_dataset.channels = test_params['num_channels']

    # Create dummy dataloaders
    from torch.utils.data import random_split, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    train_set, val_set, test_set = random_split(dummy_dataset, [80, 10, 10])
    
    train_loader = DataLoader(train_set, batch_size=test_params['batch_size'], sampler=DistributedSampler(train_set))
    val_loader = DataLoader(val_set, batch_size=test_params['batch_size'], sampler=DistributedSampler(val_set))
    test_loader = DataLoader(test_set, batch_size=test_params['batch_size'], sampler=DistributedSampler(test_set))
    
    # Call the main training function with test parameters
    main(rank, **test_params, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

# --- MODIFIED: main function now accepts pre-made dataloaders and info ---
def main(rank: int, save_every: int, total_epochs: int, dataset_name: str, batch_size: int, model_type:str, arch: str, activation: str, num_trials: int, split: list[float],
         train_loader=None, val_loader=None, test_loader=None, num_classes=None, num_channels=None):
    polarization = None
    
    # If dataloaders are not provided, load them from disk (normal operation)
    if train_loader is None:
        dataset_prefix = None
        if 'S1SLC_CVDL' in dataset_name and len(dataset_name.split('_')) > 2:
            dataset_prefix = "_".join(dataset_name.split('_')[0:2])
            polarization = dataset_name.split('_')[-1]
        
        effective_dataset_name = f"{dataset_prefix}_{polarization}" if dataset_prefix is not None and polarization is not None else dataset_name
        
        if rank == 0:
            print(f"- Loading Dataset {effective_dataset_name.upper()}...")
        
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=(dataset_prefix if dataset_prefix is not None else dataset_name),
            polarization=polarization, batch_size=batch_size, model_type=model_type, split=split
        )
        
        if rank == 0:
            print(f"Training set length: {len(train_loader)}")
            print(f"Validation set length: {len(val_loader)}")
            print(f"Test set length: {len(test_loader)}")

        root_dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
        num_classes = root_dataset.num_classes
        num_channels = root_dataset.channels
    else: # Use the provided dataloaders (test mode)
        effective_dataset_name = dataset_name

    base_lr = 0.1
    lr = base_lr

    if rank == 0:
        print(f"- Starting Train Loop on Rank {rank} with {torch.cuda.device_count()} GPUs in DDP")

    for trial in range(num_trials):
        if rank == 0:
            print(f"\n---- Starting Trial {trial} ----")
            print(f"- Initializing model...")

        if model_type == 'complex':
            model = ComplexResNet(arch, input_channels=num_channels, num_classes=num_classes, activation_function=activation)
        else: # 'real'
            model = RealResNet(arch, input_channels=num_channels, num_classes=num_classes)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        if rank == 0:
            print(f"- Initializing Trainer...")
        trainer = Trainer(model, arch, effective_dataset_name, num_classes, num_channels, train_loader, val_loader, test_loader, optimizer, save_every, trial, polarization, scheduler=scheduler)
        trainer.train(total_epochs)
    
    if rank == 0:
        summarize_trials(trainer.trial_data_dir)

    print(f"- Rank {rank} training complete.")
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    # --- NEW: Test run flag ---
    parser.add_argument('--test-run', action='store_true', help='Run a quick test with dummy data.')
    parser.add_argument('-arch', '--architecture', type=str, default='WS', choices=['WS', 'DN', 'IB'])
    parser.add_argument('-act', '--activation', metavar='ACT', type=str, default='complex_cardioid', choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'])
    parser.add_argument('--epochs', type=int, default=5, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, default=20, help='How often to save a snapshot')
    parser.add_argument('--dataset', type=str, default='S1SLC_CVDL_HH', help='Dataset to use for trainng.')
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device')
    parser.add_argument('--trials', type=int, default=5, help='The number of trials to run the experiment for.')
    parser.add_argument('--model-type', type=str, default='complex', choices=['complex', 'real'])
    parser.add_argument('--split', type=float, nargs='+', default=[0.8, 0.1, 0.1], help='Train, validation, and test split ratios.')
    parser.add_argument("--local-rank", "--local_rank", type=int, default=os.environ.get("LOCAL_RANK"))
    args = parser.parse_args()

    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    torch.autograd.set_detect_anomaly(True)

    if args.test_run:
        run_test(rank)
    else:
        if not (0.999 < sum(args.split) < 1.001):
            raise ValueError(f"Split ratios must sum to 1.0. Got: {sum(args.split)}")
        main(rank, args.save_every, args.epochs, args.dataset, args.batch_size, args.model_type, args.architecture, args.activation, args.trials, args.split)
