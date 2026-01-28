import os
import argparse
import numpy as np
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from config.configurator import load_config, setup_environment
from src.models.archive.cnn_ae import CNNAutoencoder
from src.models.transformer_ae import TransformerAutoencoder, CNNTransformerAutoencoder
from src.training.trainers.trainer_ae import TrainerAE
from src.data.datasets import create_filtered_split_datasets
import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")


def main(config_file='config.yaml'):
    # ---------------------------------------------
    # Setup environment

    # Setup Config Parameters
    config = load_config(config_file)
    np_generator = setup_environment(config)

    # Set Paths
    pickle_path = config.get_train_data_root()
    train_path = os.path.join(pickle_path, 'train_ds.pkl')
    val_path = os.path.join(pickle_path, 'val_ds.pkl')
    test_path = os.path.join(pickle_path, 'test_ds.pkl')

    # ---------------------------------------------
    # Load Dataset
    train_ds, val_ds, test_ds = load_or_create_datasets(
        config.ml.loading.load_data_pickle,
        train_path,
        val_path,
        test_path,
        config.get_dataset_parameters()
    )

    # Create data loaders with enhanced resource configuration
    resource_config = config.get_resource_config()

    train_loader = DataLoader(
        train_ds,
        batch_size=None,  # Each item is already a batch
        shuffle=True,
        num_workers=resource_config['num_workers'],
        pin_memory=resource_config['pin_memory'],
        prefetch_factor=resource_config['prefetch_factor'],
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        shuffle=False,
        num_workers=resource_config['num_workers'],
        pin_memory=resource_config['pin_memory'],
        prefetch_factor=resource_config['prefetch_factor']
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=None,
        shuffle=False,
        num_workers=resource_config['num_workers'],
        pin_memory=resource_config['pin_memory'],
        prefetch_factor=resource_config['prefetch_factor']
    )

    print_inputdata_stats(train_loader)
    logger.info("Dataset loading completed.")

    # ---------------------------------------------
    # Initialize model
    logger.info("Initializing model...")
    model_config = config.ml.model

    model_type = getattr(model_config, 'type')
    if model_type == "CNNAutoencoder":
        model = CNNAutoencoder(
            embedding_dim=config.get_embedding_dim()
        )
    elif model_type == "TransformerAutoencoder":
        model = TransformerAutoencoder(
            seq_length=10,  # From your data shape
            embedding_dim=config.get_embedding_dim(),
            num_heads=getattr(model_config, 'num_heads', 8),
            num_encoder_layers=getattr(model_config, 'num_layers', 4),
            num_decoder_layers=getattr(model_config, 'num_layers', 4),
            dropout=0.1
        )
    elif model_type == "CNNTransformerAutoencoder":
        model = CNNTransformerAutoencoder(
            seq_length=10,  # From your data shape
            embedding_dim=config.get_embedding_dim(),
            num_heads=getattr(model_config, 'num_heads', 8),
            num_encoder_layers=getattr(model_config, 'num_layers', 4),
            num_decoder_layers=getattr(model_config, 'num_layers', 4),
            dropout=0.1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Model initialized: {model_type}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---------------------------------------------
    # Initialize trainer with coordinated path configuration
    logger.info("Initializing trainer...")

    validation_config = config.get_validation_config()
    config_checkpoint_path = config.get_checkpoint_path()

    trainer = TrainerAE(
        model=model,
        device=config.get_device(),
        use_wandb=config.wandb.use_wandb,
        wandb_project=config.wandb.project,
        wandb_config=config.get_wandb_config(),
        plot_every_n_epochs=validation_config['plot_every_n_epochs'],
        results_dir=config_checkpoint_path,  # Use config path directly
        use_config_path=True
    )

    # Verify path coordination
    logger.info(f"Trainer results path: {trainer.get_results_path()}")
    logger.info(f"Config checkpoint path: {config_checkpoint_path}")
    assert trainer.get_results_path() == config_checkpoint_path, "Path mismatch detected!"

    logger.info("✅ Trainer initialized with coordinated paths.")

    # ---------------------------------------------
    # Get all hyperparameters from config
    training_params = get_training_params(config)
    restart_config = config.get_restart_config()

    # Log all hyperparameters
    log_hyperparameters(config, training_params)

    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    # Actual Training starts here. This function is build with try/except
    # to handle training interupts and prevent dataloss using automatic checkpointing

    try:
        # ---------------------------------------------
        # Train the model (with or without restart)
        logger.info("Starting training...")

        # Get separate parameter sets for different methods
        training_params = get_training_params(config)  # For train() method
        setup_params = get_setup_training_params(config)  # For setup_training() method

        if restart_config['enable_restart']:
            logger.info("🔄 Restart mode enabled")
            logger.info(f"Auto-find checkpoint: {restart_config['auto_find_checkpoint']}")
            if restart_config['checkpoint_path']:
                logger.info(f"Specific checkpoint: {restart_config['checkpoint_path']}")
            logger.info(f"Save restart every: {restart_config['save_restart_every']} epochs")

            # Use restart-enabled training (handles setup internally)
            history = trainer.train_with_restart(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader if validation_config['evaluate_on_test'] else None,
                epochs=config.get_epochs(),
                restart_from_checkpoint=restart_config['checkpoint_path'],
                auto_find_checkpoint=restart_config['auto_find_checkpoint'],
                save_restart_every=restart_config['save_restart_every'],
                **training_params
            )
        else:
            logger.info("🆕 Fresh training mode (restart disabled)")

            # Setup training components first with separated parameters
            logger.info("Setting up training components...")
            trainer.setup_training(
                epochs=config.get_epochs(),
                learning_rate=config.get_learning_rate(),
                weight_decay=config.get_weight_decay(),
                steps_per_epoch=len(train_loader),
                **setup_params  # optimizer_type, scheduler_type
            )
            logger.info("✅ Training components setup completed")

            # Get checkpointing configuration
            checkpointing_config = getattr(config.ml.training, 'checkpointing', None)
            save_best = getattr(checkpointing_config, 'save_best', True) if checkpointing_config else True

            # Use training with only compatible parameters
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader if validation_config['evaluate_on_test'] else None,
                epochs=config.get_epochs(),
                save_best=save_best,
                lr_scheduler_config=config.get_lr_scheduler_config(),
                **training_params  # FIXED: Only compatible parameters
            )

        # ---------------------------------------------
        # Generate evaluation report
        logger.info("Generating evaluation report...")
        trainer.generate_full_report(test_loader)

        # ---------------------------------------------
        # Save final training state using coordinated paths
        logger.info("Saving final checkpoint...")

        # Use trainer's coordinated path methods
        final_checkpoint_name = 'final_checkpoint.pth'
        final_checkpoint_path = trainer.create_checkpoint_path(final_checkpoint_name)
        trainer.save_checkpoint(final_checkpoint_path, len(history['train_loss']) - 1)

        logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")

        print("Training completed successfully!")

        # Print final results with enhanced path information
        print_final_results(history, config, restart_config, trainer)

    except KeyboardInterrupt:
        handle_training_interruption(config, trainer, restart_config, "user_interrupt")

    except Exception as e:
        handle_training_interruption(config, trainer, restart_config, "error", error=e)
        raise

    finally:
        # Always cleanup
        trainer.cleanup()



## HELPERS - Data Loading
####################################################################
####################################################################
def load_or_create_datasets(load_data_pickle_flag, train_path, val_path, test_path, dataset_parameters):
    """Load or create datasets with error handling"""
    if load_data_pickle_flag:
        logger.info("Loading datasets from pickle files...")
        try:
            with open(train_path, 'rb') as f:
                train_ds = pickle.load(f)
            with open(val_path, 'rb') as f:
                val_ds = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_ds = pickle.load(f)
            logger.info(f"Successfully loaded datasets from pickle files")
        except Exception as e:
            logger.error(f"Failed to load pickle files: {e}")
            raise
    else:
        logger.info("Creating datasets from metadata file...")
        try:
            train_ds, test_ds, val_ds = create_filtered_split_datasets(
                **dataset_parameters
            )

            logger.info("Saving datasets to pickle files...")
            with open(train_path, 'wb') as f:
                pickle.dump(train_ds, f)
            with open(val_path, 'wb') as f:
                pickle.dump(val_ds, f)
            with open(test_path, 'wb') as f:
                pickle.dump(test_ds, f)
            logger.info("Datasets saved successfully")

        except Exception as e:
            logger.error(f"Failed to create/save datasets: {e}")
            raise

    return train_ds, val_ds, test_ds


## HELPERS - Hyperparameter Handling and Logging
####################################################################
####################################################################

def get_training_params(config) -> dict:
    """Extract training parameters compatible with TrainerAE.train() method"""
    loss_weights = config.get_loss_weights()
    regularization = config.get_regularization_config()

    # FIXED: Only return parameters that train() method accepts
    return {
        'learning_rate': config.get_learning_rate(),
        'weight_decay': config.get_weight_decay(),
        'loss_weights': loss_weights,
        'grad_clip_norm': regularization['grad_clip_norm']
    }

def get_setup_training_params(config) -> dict:
    """Get parameters specifically for setup_training() method"""
    optimizer_config = config.get_optimizer_config()
    lr_scheduler_config = config.get_lr_scheduler_config()

    return {
        'optimizer_type': optimizer_config['type'],
        'scheduler_type': lr_scheduler_config.get('type', 'plateau')
    }


def log_hyperparameters(config, training_params):
    """Log all hyperparameters"""
    logger.info("=" * 60)
    logger.info("TRAINING HYPERPARAMETERS")
    logger.info("=" * 60)

    logger.info(f"Epochs: {config.get_epochs()}")
    logger.info(f"Learning Rate: {config.get_learning_rate()}")
    logger.info(f"Weight Decay: {config.get_weight_decay()}")
    logger.info(f"Batch Size: {config.get_batch_size()}")


    optimizer_config = config.get_optimizer_config()
    lr_scheduler_config = config.get_lr_scheduler_config()

    logger.info(f"Optimizer: {optimizer_config['type']}")
    logger.info(f"LR Scheduler: {lr_scheduler_config.get('type', 'plateau')}")

    logger.info("Loss Weights:")
    for name, weight in config.get_loss_weights().items():
        logger.info(f"  {name}: {weight}")

    logger.info("Regularization:")
    reg_config = config.get_regularization_config()
    for name, value in reg_config.items():
        logger.info(f"  {name}: {value}")

    logger.info("Model:")
    model_type = getattr(config.ml.model, 'type', 'OptimizedWidthReducedAutoencoder')
    logger.info(f"  Type: {model_type}")
    logger.info(f"  Embedding Dim: {config.get_embedding_dim()}")
    logger.info(f"  Channels: {config.ml.model.channels_per_layer}")

    logger.info("=" * 60)





def handle_training_interruption(config, trainer, restart_config, interrupt_type, error=None):
    """Handle training interruptions with proper awareness of trainer's timestamped folder structure"""
    print(f"\nTraining interrupted: {interrupt_type}")

    # CRITICAL: Use trainer's existing timestamp, not a new one!
    # The trainer already created a timestamped folder, we must use that same folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # For emergency filename only
    emergency_filename = f'{interrupt_type}_checkpoint_{timestamp}.pth'
    current_epoch = len(trainer.history['train_loss']) - 1 if trainer.history['train_loss'] else 0

    # FIXED: Always use trainer.create_checkpoint_path() which knows the correct timestamped folder
    emergency_path = None
    try:
        # PRIMARY: Use trainer's coordinated path system
        # This automatically uses the trainer's existing timestamped folder
        emergency_path = trainer.create_checkpoint_path(emergency_filename)
        trainer.save_checkpoint(emergency_path, current_epoch)
        print(f"✅ Emergency checkpoint saved: {emergency_path}")

    except Exception as primary_error:
        logger.error(f"Primary emergency save failed: {primary_error}")

        # FALLBACK: Use trainer's actual results directory (with existing timestamp)
        try:
            # Get trainer's ACTUAL results directory (includes the existing timestamped folder)
            trainer_results_dir = trainer.get_results_path()

            # Ensure the trainer's directory exists (it should, but just in case)
            os.makedirs(trainer_results_dir, exist_ok=True)

            # Create path using trainer's existing folder structure
            simple_emergency_filename = f'emergency_{timestamp}.pth'
            emergency_path = trainer.create_checkpoint_path(simple_emergency_filename)

            # Manual checkpoint creation (bypass trainer.save_checkpoint if it's failing)
            checkpoint = {
                'epoch': current_epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer else None,
                'history': trainer.history,
                'error_info': str(primary_error),
                'timestamp': datetime.now().isoformat(),
                'emergency_save': True,
                'trainer_results_dir': trainer_results_dir,
                'saved_via': 'fallback_manual_save'
            }

            torch.save(checkpoint, emergency_path)
            print(f"✅ Fallback emergency checkpoint saved: {emergency_path}")

        except Exception as fallback_error:
            logger.error(f"Fallback emergency save failed: {fallback_error}")

            # LAST RESORT: Try to save in trainer's parent directory if timestamped folder fails
            try:
                # Extract trainer's base directory (before the timestamped folder)
                trainer_full_path = trainer.get_results_path()

                # Try to identify if this is a timestamped folder structure
                path_parts = trainer_full_path.split(os.sep)

                # Look for timestamped training folder pattern
                timestamped_folder = None
                base_path = None

                for i, part in enumerate(path_parts):
                    if part.startswith('training_') and len(part) == len('training_20250803_142530'):
                        timestamped_folder = part
                        base_path = os.sep.join(path_parts[:i])
                        break

                if base_path and timestamped_folder:
                    # Save to base path with clear indication of the issue
                    last_resort_filename = f'EMERGENCY_TIMESTAMPED_FOLDER_FAILED_{timestamp}.pth'
                    emergency_path = os.path.join(base_path, last_resort_filename)

                    checkpoint_data = {
                        'model_state_dict': trainer.model.state_dict(),
                        'epoch': current_epoch,
                        'emergency_save': True,
                        'timestamped_folder_failed': True,
                        'original_trainer_path': trainer_full_path,
                        'failed_timestamped_folder': timestamped_folder,
                        'saved_to_base': base_path,
                        'warning': f'Could not save to timestamped folder {timestamped_folder}'
                    }
                else:
                    # Fallback to config path if we can't parse the structure
                    config_base_path = config.get_checkpoint_path()
                    os.makedirs(config_base_path, exist_ok=True)

                    last_resort_filename = f'EMERGENCY_STRUCTURE_UNKNOWN_{timestamp}.pth'
                    emergency_path = os.path.join(config_base_path, last_resort_filename)

                    checkpoint_data = {
                        'model_state_dict': trainer.model.state_dict(),
                        'epoch': current_epoch,
                        'emergency_save': True,
                        'structure_unknown': True,
                        'trainer_path': trainer_full_path,
                        'warning': 'Could not parse trainer folder structure'
                    }

                torch.save(checkpoint_data, emergency_path)
                print(f"⚠️  LAST RESORT: Emergency checkpoint saved outside timestamped folder")
                print(f"   Location: {emergency_path}")

                if timestamped_folder:
                    print(f"   Failed timestamped folder: {timestamped_folder}")

                logger.warning(f"Emergency save bypassed timestamped folder structure!")

            except Exception as last_resort_error:
                logger.error(f"All checkpoint save attempts failed: {last_resort_error}")
                emergency_path = None

    # Save restart checkpoint if enabled
    if emergency_path and restart_config['enable_restart']:
        try:
            val_loss = trainer.history['val_loss'][-1] if trainer.history['val_loss'] else float('inf')
            trainer.save_restart_checkpoint(current_epoch, val_loss, interrupt_type)
            print(f"✅ Emergency restart checkpoint also saved")
        except Exception as restart_error:
            logger.warning(f"Failed to save emergency restart checkpoint: {restart_error}")

    # ENHANCED RECOVERY INFORMATION - Focus on timestamped folder structure
    print(f"\n📁 TIMESTAMPED FOLDER STRUCTURE ANALYSIS:")
    try:
        config_path = config.get_checkpoint_path()
        trainer_path = trainer.get_results_path()

        print(f"Config base path:    {config_path}")
        print(f"Trainer actual path: {trainer_path}")

        # Parse the timestamped folder structure
        path_parts = trainer_path.split(os.sep)
        timestamped_folder = None

        for part in path_parts:
            if part.startswith('training_') and len(part) == len('training_20250803_142530'):
                timestamped_folder = part
                break

        if timestamped_folder:
            print(f"Timestamped folder:  {timestamped_folder}")

            # Show the full structure
            relative_path = os.path.relpath(trainer_path, config_path)
            print(f"Full structure:      CONFIG_BASE/{relative_path}/")

            # Verify the timestamped folder exists
            timestamped_dir_exists = os.path.exists(trainer_path)
            print(f"Timestamped dir exists: {'✅ YES' if timestamped_dir_exists else '❌ NO'}")

        else:
            print(f"⚠️  No timestamped folder detected in path structure")

        # Show emergency checkpoint location relative to structure
        if emergency_path:
            if trainer_path in emergency_path:
                print(f"✅ Emergency checkpoint saved in correct timestamped folder")
            else:
                print(f"⚠️  Emergency checkpoint saved outside timestamped folder")
                print(f"   Emergency path: {emergency_path}")

    except Exception as structure_error:
        logger.warning(f"Could not analyze timestamped folder structure: {structure_error}")

    # Show recovery information with timestamp awareness
    print(f"\n📁 RECOVERY INFORMATION:")
    if emergency_path and os.path.exists(emergency_path):
        print(f"Emergency checkpoint: {emergency_path}")
        print(f"Emergency file: {os.path.basename(emergency_path)}")

    # List checkpoints in trainer's timestamped directory
    try:
        trainer_checkpoints = trainer.get_all_checkpoint_paths()
        trainer_total = sum(len(paths) for paths in trainer_checkpoints.values())

        if trainer_total > 0:
            print(f"\nCheckpoints in timestamped folder: {trainer_total} files")
            print(f"Location: {trainer.get_results_path()}")

            for checkpoint_type, paths in trainer_checkpoints.items():
                if paths:
                    latest = os.path.basename(paths[0])
                    print(f"  {checkpoint_type}: {len(paths)} files (latest: {latest})")
        else:
            print(f"\nNo checkpoints found in timestamped folder: {trainer.get_results_path()}")

    except Exception as trainer_list_error:
        logger.warning(f"Could not list trainer checkpoints: {trainer_list_error}")

    # Error details
    if error:
        logger.error(f"Training failed with error: {error}")
        print(f"\n❌ ERROR DETAILS: {str(error)}")

        # Debugging info with timestamp awareness
        try:
            print(f"\n🔍 DEBUG INFO:")
            print(f"  Current epoch: {current_epoch}")
            print(f"  Training history: {len(trainer.history.get('train_loss', []))} epochs")
            print(f"  Optimizer: {'Available' if trainer.optimizer else 'None'}")
            print(f"  Model device: {next(trainer.model.parameters()).device}")

            # Show trainer initialization info
            trainer_dir = trainer.get_results_path()
            print(f"  Trainer results dir: {trainer_dir}")
            print(f"  Trainer dir exists: {os.path.exists(trainer_dir)}")

            if os.path.exists(trainer_dir):
                print(f"  Trainer dir writable: {os.access(trainer_dir, os.W_OK)}")

                # List contents of timestamped folder
                try:
                    contents = os.listdir(trainer_dir)
                    checkpoint_files = [f for f in contents if f.endswith('.pth')]
                    print(f"  Checkpoint files in folder: {len(checkpoint_files)}")
                    if checkpoint_files:
                        print(f"    Recent: {sorted(checkpoint_files)[-1]}")
                except:
                    pass

        except Exception as debug_error:
            logger.warning(f"Could not print debugging info: {debug_error}")

    return emergency_path





## Callback functions - Plot Statistics for IO
####################################################################
####################################################################

def print_inputdata_stats(data):
    """Print input data statistics"""
    try:
        # Get data dimensions from first batch
        sample_batch = next(iter(data))
        if isinstance(sample_batch, (list, tuple)):
            sample_data = sample_batch[0]
        else:
            sample_data = sample_batch

        print("\n" + "=" * 50)
        print("INPUT DATA STATISTICS")
        print("=" * 50)
        print(f"Data shape: {sample_data.shape}")

        if len(sample_data.shape) >= 5:  # [B, T, C, H, W]
            batch_size = sample_data.shape[0]
            sequence_length = sample_data.shape[1]
            channels_nbr = sample_data.shape[2]
            height = sample_data.shape[3]
            width = sample_data.shape[4]

            print(f"Batch size: {batch_size}")
            print(f"Sequence length: {sequence_length}")
            print(f"Channels: {channels_nbr}")
            print(f"Height: {height}")
            print(f"Width: {width}")

            # Calculate total elements
            total_elements = np.prod(sample_data.shape)
            print(f"Total elements per batch: {total_elements:,}")

            # Data type and memory usage
            print(f"Data type: {sample_data.dtype}")
            memory_mb = (total_elements * sample_data.element_size()) / (1024 * 1024)
            print(f"Memory per batch: {memory_mb:.2f} MB")

        print("=" * 50 + "\n")

    except Exception as e:
        logger.warning(f"Could not print input data statistics: {e}")



def print_final_results(history, config, restart_config, trainer):
    """Print final results"""
    print("\n" + "=" * 70)
    print("FINAL TRAINING RESULTS")
    print("=" * 70)

    # Training metrics
    print(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print(f"Final Validation MSE: {history['val_mse'][-1]:.6f}")
    print(f"Total Epochs Completed: {len(history['train_loss'])}")

    # UPDATED: Enhanced path information
    print(f"\n📁 CHECKPOINT LOCATIONS:")
    print(f"Config checkpoint path: {config.get_checkpoint_path()}")
    print(f"Trainer results path: {trainer.get_results_path()}")

    # List all checkpoints using trainer's coordination methods
    all_checkpoints = trainer.get_all_checkpoint_paths()
    total_checkpoints = sum(len(paths) for paths in all_checkpoints.values())
    print(f"Total checkpoints created: {total_checkpoints}")

    for checkpoint_type, paths in all_checkpoints.items():
        if paths:
            print(f"  {checkpoint_type.upper()}: {len(paths)} files")
            for path in paths[:2]:  # Show first 2
                print(f"    - {os.path.basename(path)}")
            if len(paths) > 2:
                print(f"    ... and {len(paths) - 2} more")

    # Best checkpoint information
    best_checkpoint = trainer.get_latest_checkpoint_path()
    if best_checkpoint:
        print(f"\n🏆 Best model: {os.path.basename(best_checkpoint)}")
        print(f"   Full path: {best_checkpoint}")

    # Restart-specific information
    if restart_config['enable_restart']:
        print(f"\n🔄 RESTART INFO:")
        print(f"Restart checkpoints saved every {restart_config['save_restart_every']} epochs")
        latest_restart = trainer.find_latest_checkpoint()
        if latest_restart:
            print(f"Latest restart checkpoint: {os.path.basename(latest_restart)}")

    # Model information
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n🤖 MODEL INFO:")
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Configuration summary
    model_type = getattr(config.ml.model, 'type', 'OptimizedWidthReducedAutoencoder')
    optimizer_config = config.get_optimizer_config()

    print(f"Model type: {model_type}")
    print(f"Embedding dimension: {config.get_embedding_dim()}")
    print(f"Optimizer: {optimizer_config['type']}")
    print(f"Learning rate: {config.get_learning_rate()}")

    print("=" * 70)


## Config validation and override
####################################################################
####################################################################
def apply_config_overrides(config, overrides):
    """Apply command line overrides to configuration"""
    if not overrides:
        return config

    logger.info("Applying configuration overrides:")
    for override in overrides:
        try:
            key, value = override.split('=', 1)
            # Try to convert value to appropriate type
            try:
                # Try int first
                value = int(value)
            except ValueError:
                try:
                    # Try float
                    value = float(value)
                except ValueError:
                    # Try boolean
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    # Otherwise keep as string

            # For now, just log the override - complex dot notation setting needs more work
            logger.info(f"  {key} = {value} (override logged, manual implementation needed)")

        except Exception as e:
            logger.warning(f"Failed to apply override '{override}': {e}")

    return config


def validate_configuration(config):
    """Validate configuration for common issues"""
    return config.validate_configuration()  # Use the method from ConfigurationManager


## CLI Handling
####################################################################
####################################################################
def cli_main():
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Apply overrides
    if args.override:
        config = apply_config_overrides(config, args.override)

    # Validate configuration
    if args.validate_config:
        if validate_configuration(config):
            print("✅ Configuration is valid")
            return 0
        else:
            print("❌ Configuration validation failed")
            return 1

    # Dry run
    if args.dry_run:
        print("🔍 Dry run mode - configuration loaded successfully")
        return 0
    # Validate before training
    if not validate_configuration(config):
        return 1

    # Run training
    try:
        main(args.config)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='wristUS Training pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--override',
        type=str,
        nargs='*',
        default=[],
        help='Override config values using dot notation (e.g., ml.training.epochs=100)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load config and print summary without training'
    )
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration file and exit'
    )

    return parser.parse_args()

if __name__ == "__main__":
    exit_code = cli_main()
    exit(exit_code)