"""
Embedding Extraction

Extract embeddings from a pretrained CNN Autoencoder for downstream classification.

Usage:
    python -m src.training.extract_embeddings --config config/config.yaml --checkpoint path/to/checkpoint.pth
    python -m src.training.extract_embeddings --config config/config.yaml --checkpoint path/to/checkpoint.pth --normalize
"""

import os
import sys
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.configurator import load_config, setup_environment

import utils.logging_config as logconf
logger = logconf.get_logger("EXTRACT")


def create_model(config):
    """Create model based on config."""
    model_type = config.ml.model.type

    if model_type == "CNNAutoencoder":
        from src.models.cnn_ae import CNNAutoencoder

        model = CNNAutoencoder(
            in_channels=3,
            input_height=130,
            input_width=config.preprocess.tokenization.window,
            channels=config.ml.model.channels_per_layer,
            embedding_dim=config.ml.model.embedding_dim,
            use_batchnorm=True
        )
    elif model_type == "UNetAutoencoder":
        from src.models.unet_ae import UNetAutoencoder

        model = UNetAutoencoder(
            in_channels=3,
            input_height=130,
            input_width=config.preprocess.tokenization.window,
            channels=config.ml.model.channels_per_layer,
            embedding_dim=config.ml.model.embedding_dim,
            use_batchnorm=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: str):
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model weights loaded successfully")

    # Log checkpoint info if available
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        logger.info(f"Checkpoint val_loss: {checkpoint['val_loss']:.6f}")

    return model


def load_datasets_from_pickle(config):
    """
    Always load datasets from pickle files created during AE training.

    This ensures embedding extraction uses the exact same data splits
    as autoencoder training, regardless of the load_data_pickle config setting.
    """
    pickle_path = config.get_train_data_root()
    train_path = os.path.join(pickle_path, 'train_ds.pkl')
    val_path = os.path.join(pickle_path, 'val_ds.pkl')
    test_path = os.path.join(pickle_path, 'test_ds.pkl')

    # Verify pickle files exist
    for path, name in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} pickle not found at {path}. "
                f"Run AE training first to create dataset pickles."
            )

    logger.info(f"Loading datasets from pickle (AE training splits)...")
    logger.info(f"Pickle path: {pickle_path}")

    with open(train_path, 'rb') as f:
        train_ds = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_ds = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_ds = pickle.load(f)

    logger.info(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def extract_embeddings_from_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    desc: str = "Extracting"
) -> tuple:
    """
    Extract embeddings and labels from a dataloader.

    Args:
        model: Pretrained autoencoder model
        dataloader: DataLoader to extract from
        device: Device to use
        desc: Progress bar description

    Returns:
        Tuple of (embeddings, labels) as numpy arrays
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # Handle batch format
            if isinstance(batch, dict):
                data = batch['tokens'].to(device)
                labels = batch['labels']
            elif isinstance(batch, (list, tuple)):
                data = batch[0].to(device)
                labels = batch[1] if len(batch) > 1 else None
            else:
                data = batch.to(device)
                labels = None

            # Extract embeddings
            embeddings = model.encode(data)
            all_embeddings.append(embeddings.cpu().numpy())

            # Convert soft labels to hard labels if needed
            if labels is not None:
                if labels.dim() > 1:  # Soft labels: [B, num_classes]
                    hard_labels = labels.argmax(dim=1).numpy()
                else:  # Already hard labels
                    hard_labels = labels.numpy()
                all_labels.append(hard_labels)

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.concatenate(all_labels, axis=0) if all_labels else None

    return embeddings_array, labels_array


def main(config_path: str, checkpoint_path: str, output_dir: str = None, normalize: bool = False):
    """
    Main embedding extraction function.

    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to trained model checkpoint
        output_dir: Output directory for embeddings (default: alongside checkpoint)
        normalize: Whether to apply StandardScaler normalization
    """
    # Load config
    config = load_config(config_path, create_dirs=False)
    setup_environment(config)

    device = config.get_device()

    logger.info("=" * 60)
    logger.info("EMBEDDING EXTRACTION")
    logger.info("=" * 60)

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load datasets from pickle (always use AE training splits)
    train_ds, val_ds, test_ds = load_datasets_from_pickle(config)

    # Create dataloaders (no shuffling for extraction)
    resource_config = config.get_resource_config()
    loader_kwargs = {
        'batch_size': None,  # Pre-batched
        'shuffle': False,
        'num_workers': resource_config['num_workers'],
        'pin_memory': resource_config['pin_memory'],
    }

    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    # Create and load model
    model = create_model(config)
    model = load_checkpoint(checkpoint_path, model, device)
    model = model.to(device)
    model.eval()

    logger.info(f"Model embedding dimension: {model.embedding_dim}")

    # Extract embeddings
    logger.info("Extracting embeddings...")

    X_train, y_train = extract_embeddings_from_loader(
        model, train_loader, device, desc="Train"
    )
    logger.info(f"Train: {X_train.shape}, labels: {y_train.shape if y_train is not None else 'None'}")

    X_val, y_val = extract_embeddings_from_loader(
        model, val_loader, device, desc="Val"
    )
    logger.info(f"Val: {X_val.shape}, labels: {y_val.shape if y_val is not None else 'None'}")

    X_test, y_test = extract_embeddings_from_loader(
        model, test_loader, device, desc="Test"
    )
    logger.info(f"Test: {X_test.shape}, labels: {y_test.shape if y_test is not None else 'None'}")

    # Optional normalization
    scaler = None
    if normalize:
        logger.info("Applying StandardScaler normalization...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Save embeddings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embeddings_path = os.path.join(output_dir, f'embeddings_{timestamp}.npz')

    save_dict = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'embedding_dim': model.embedding_dim,
        'checkpoint_path': checkpoint_path,
        'normalized': normalize,
    }

    np.savez(embeddings_path, **save_dict)
    logger.info(f"Embeddings saved to: {embeddings_path}")

    # Save scaler if used
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to: {scaler_path}")

    # Also save as 'latest' for easy access
    latest_path = os.path.join(output_dir, 'embeddings_latest.npz')
    np.savez(latest_path, **save_dict)
    logger.info(f"Also saved as: {latest_path}")

    # Print summary
    print_summary(X_train, X_val, X_test, y_train, y_val, y_test, output_dir)

    return embeddings_path


def print_summary(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """Print extraction summary."""
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nEmbedding shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    if y_train is not None:
        print(f"\nLabel distribution:")
        for split_name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            dist_str = ', '.join([f"class {u}: {c}" for u, c in zip(unique, counts)])
            print(f"  {split_name}: {dist_str}")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract embeddings from pretrained autoencoder')

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config YAML file'
    )

    parser.add_argument(
        '--checkpoint', '-ckpt',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for embeddings (default: alongside checkpoint)'
    )

    parser.add_argument(
        '--normalize', '-n',
        action='store_true',
        help='Apply StandardScaler normalization to embeddings'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        normalize=args.normalize
    )