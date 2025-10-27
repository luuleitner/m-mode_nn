import os
import argparse

from torch.utils.data import DataLoader

# Intrinsic Library Imports
from data.classifier.loader import ZarrDataloader, create_split_indices
from models.dev.transformer_classifier import TransformerClassifier
from utils import seed, wandb_utils
from utils.utils import load_config


import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")
from utils.utils import get_git_root



def main(config, debug):
    # ---------------------------------------------
    # Setup Wandb
    logger.info("Setup logging in wandb.ai...")
    cfg = load_config(config)
    config = wandb_utils.initialize_wandb(cfg)

    # ---------------------------------------------
    # Set Behaviour
    # Set deterministic/probabilistic behavior
    if cfg.global_setting.run.behaviour == 'deterministic':
        seed.set_seed(cfg.global_setting.run.config.deterministic.seed)
        logger.warning("Model is set to deterministic...")
    else:
        logger.warning("Model is set to probabilistic...")

    operation_mode = cfg.global_setting.run.mode
    debug_level = cfg.global_setting.run.config.debug.level if operation_mode == 'debug' else None

    # ---------------------------------------------
    # Load train-validation-test indices from Metadata file
    logger.info("Initializing train, validation and test dataset from metadata file...")
    train_sq, val_sq, test_sq = create_split_indices(meta=os.path.join(cfg.dprocessing.basepath, 'processed', f'{cfg.dprocessing.id}_metadata.csv'),
                                                     config=cfg)


    train_dataset = ZarrDataloader(data=os.path.join(cfg.dprocessing.basepath, f'processed/{cfg.dprocessing.id}_token.zarr'),
                                   sequence=train_sq)
    val_dataset = ZarrDataloader(data=os.path.join(cfg.dprocessing.basepath, f'processed/{cfg.dprocessing.id}_token.zarr'),
                                   sequence=val_sq)
    test_dataset = ZarrDataloader(data=os.path.join(cfg.dprocessing.basepath, f'processed/{cfg.dprocessing.id}_token.zarr'),
                                   sequence=test_sq)


    train_loader = DataLoader(train_dataset, batch_size=cfg.model.data.batch, shuffle= True, num_workers=4)
    # debug output
    if operation_mode == 'debug' and debug_level >= 1:
        X, y = next(iter(train_loader))
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    else:
        _log_dataset_summary(train_loader, name="Train")

    val_loader = DataLoader(val_dataset, batch_size=cfg.model.data.batch, shuffle=True, num_workers=4)
    # debug output
    if operation_mode == 'debug' and debug_level >= 1:
        X, y = next(iter(train_loader))
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    else:
        _log_dataset_summary(val_loader, name="Validation")

    test_loader = DataLoader(test_dataset, batch_size=cfg.model.data.batch, shuffle=True, num_workers=4)
    # debug output
    if operation_mode == 'debug' and debug_level >= 1:
        X, y = next(iter(train_loader))
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    else:
        _log_dataset_summary(test_loader, name="Test")


    logger.info("Initilization successfully")

    # ---------------------------------------------
    #### Training
    logger.info("Preparing training labels and initializing model...")
    model = TransformerClassifier(seq_length=cfg.model.data.sequence,
                                  embed_dim=cfg.model.data.feature[-1],
                                  num_heads=cfg.model.encoder.num_heads,
                                  num_layers=cfg.model.encoder.num_layers)
    model.to(cfg.training.device)
    logger.info("Done.")


    logger.info("Start training...")
    trainer = Trainer(model=model, Xy_train=train_loader, Xy_val=val_loader, config=cfg)
    logger.info("Done.")


def _log_dataset_summary(dataloader, name=None):
    num_batches = 0
    total_sequences = 0
    total_tokens = 0
    first_batch_shape = None

    for X, y in dataloader:
        B, T, F = X.shape  # B=batch size, T=seq len, F=embedding dim
        first_batch_shape = (B, T, F)
        num_batches += 1
        total_sequences += B
        total_tokens += B * T
        break  # just grab shape from first batch

    logger.info(f"{name} Dataset Summary:")
    logger.info(f"  Number of batches:     {len(dataloader)}")
    logger.info(f"  Sequences per batch:   {first_batch_shape[0]}")
    logger.info(f"  Tokens per sequence:   {first_batch_shape[1]}")
    logger.info(f"  Embedding dimension:   {first_batch_shape[2]}")
    logger.info(f"  Total sequences:       {len(dataloader) * first_batch_shape[0]}")
    logger.info(f"  Total tokens:          {len(dataloader) * first_batch_shape[0] * first_batch_shape[1]}\n")




def parse_args():
    parser = argparse.ArgumentParser(description="Run Transformer training with debug visualization and additional wandb logging.")
    parser.add_argument('--config',
                        default=os.path.join(get_git_root(), "config/config.yaml"),
                        help="Path to config file, relative to Git repo root")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(config=args.config, debug=args.debug)
