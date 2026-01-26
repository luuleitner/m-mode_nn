import os
import torch
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TrainerRestartManager:
    """
    Handles checkpoint saving, loading, and training resumption for TrainerAE.

    This class provides robust restart capabilities with automatic checkpoint detection,
    training state restoration, and configuration validation.
    """

    def __init__(self, trainer: 'TrainerAE'):
        self.trainer = trainer

    def save_restart_checkpoint(
            self,
            epoch: int,
            val_loss: float,
            checkpoint_type: str = "restart"
    ) -> str:
        """
        Save a comprehensive restart checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            checkpoint_type: Type of checkpoint ("restart", "best", "final")

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"{checkpoint_type}_checkpoint_epoch_{epoch:03d}.pth"
        checkpoint_path = os.path.join(self.trainer.results_dir, checkpoint_name)

        # Comprehensive checkpoint data
        checkpoint_data = {
            # Core training state
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict() if self.trainer.optimizer else None,
            'scheduler_state_dict': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,

            # Training metrics and history
            'val_loss': val_loss,
            'history': self.trainer.history.copy(),

            # Training configuration for validation
            'training_config': {
                'device': str(self.trainer.device),
                'model_class': self.trainer.model.__class__.__name__,
                'results_dir': self.trainer.results_dir,
                'use_wandb': self.trainer.use_wandb,
            },

            # Model architecture info for verification
            'model_info': {
                'total_params': sum(p.numel() for p in self.trainer.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad)
            },

            # Checkpoint metadata
            'checkpoint_type': checkpoint_type,
            'save_timestamp': torch.tensor(epoch).float(),  # Simple timestamp placeholder
        }

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Restart checkpoint saved: {checkpoint_path}")

        # Also save to WandB if enabled
        self.trainer._safe_wandb_save(checkpoint_path)

        return checkpoint_path

    def find_latest_checkpoint(self, checkpoint_dir: str = None) -> Optional[str]:
        """
        Find the latest restart checkpoint in the specified directory.

        Args:
            checkpoint_dir: Directory to search (defaults to trainer results_dir)

        Returns:
            Path to latest checkpoint or None if not found
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.trainer.results_dir

        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return None

        # Look for restart checkpoints
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("restart_checkpoint_") and filename.endswith(".pth"):
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                if os.path.isfile(checkpoint_path):
                    checkpoint_files.append(checkpoint_path)

        if not checkpoint_files:
            # Also check for other checkpoint types
            for filename in os.listdir(checkpoint_dir):
                if "checkpoint" in filename and filename.endswith(".pth"):
                    checkpoint_path = os.path.join(checkpoint_dir, filename)
                    if os.path.isfile(checkpoint_path):
                        checkpoint_files.append(checkpoint_path)

        if not checkpoint_files:
            logger.info(f"No checkpoints found in: {checkpoint_dir}")
            return None

        # Return the most recent checkpoint (by modification time)
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
        return latest_checkpoint

    def load_restart_checkpoint(
            self,
            checkpoint_path: str,
            strict_loading: bool = True,
            validate_config: bool = True
    ) -> Tuple[int, Dict]:
        """
        Load restart checkpoint with comprehensive validation.

        Args:
            checkpoint_path: Path to checkpoint file
            strict_loading: Whether to enforce strict state dict loading
            validate_config: Whether to validate training configuration

        Returns:
            Tuple of (last_epoch, checkpoint_data)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading restart checkpoint: {checkpoint_path}")

        # Load checkpoint data
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.trainer.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Validate checkpoint structure
        required_keys = ['epoch', 'model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint_data]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        # Load model state
        try:
            self.trainer.model.load_state_dict(
                checkpoint_data['model_state_dict'],
                strict=strict_loading
            )
            logger.info("✅ Model state loaded successfully")
        except Exception as e:
            if strict_loading:
                raise RuntimeError(f"Failed to load model state: {e}")
            else:
                logger.warning(f"Model state loaded with warnings: {e}")

        # Load optimizer state if available and optimizer exists
        if self.trainer.optimizer and 'optimizer_state_dict' in checkpoint_data:
            if checkpoint_data['optimizer_state_dict'] is not None:
                try:
                    self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    logger.info("✅ Optimizer state loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
            else:
                logger.info("No optimizer state in checkpoint")

        # Load scheduler state if available and scheduler exists
        if self.trainer.scheduler and 'scheduler_state_dict' in checkpoint_data:
            if checkpoint_data['scheduler_state_dict'] is not None:
                try:
                    self.trainer.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                    logger.info("✅ Scheduler state loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
            else:
                logger.info("No scheduler state in checkpoint")

        # Restore training history
        if 'history' in checkpoint_data:
            self.trainer.history = checkpoint_data['history']
            logger.info(f"✅ Training history restored ({len(self.trainer.history['train_loss'])} epochs)")

        # Validate configuration if requested
        if validate_config and 'training_config' in checkpoint_data:
            self._validate_training_config(checkpoint_data['training_config'])

        # Print checkpoint info
        last_epoch = checkpoint_data['epoch']
        logger.info(f"✅ Checkpoint loaded successfully from epoch {last_epoch}")

        if 'val_loss' in checkpoint_data:
            logger.info(f"   Last validation loss: {checkpoint_data['val_loss']:.6f}")

        if 'model_info' in checkpoint_data:
            current_params = sum(p.numel() for p in self.trainer.model.parameters())
            saved_params = checkpoint_data['model_info']['total_params']
            if current_params != saved_params:
                logger.warning(f"Parameter count mismatch: current={current_params}, saved={saved_params}")

        return last_epoch, checkpoint_data

    def _validate_training_config(self, saved_config: Dict) -> None:
        """Validate that current training setup matches saved configuration."""
        current_config = {
            'device': str(self.trainer.device),
            'model_class': self.trainer.model.__class__.__name__,
            'use_wandb': self.trainer.use_wandb,
        }

        mismatches = []
        for key, saved_value in saved_config.items():
            if key in current_config:
                current_value = current_config[key]
                if current_value != saved_value:
                    mismatches.append(f"{key}: current='{current_value}' vs saved='{saved_value}'")

        if mismatches:
            logger.warning("Configuration mismatches detected:")
            for mismatch in mismatches:
                logger.warning(f"  - {mismatch}")
        else:
            logger.info("✅ Configuration validation passed")

    # Fix 1: Update the get_restart_config function in your training script

    def get_restart_config(config):
        """
        Extract restart configuration from YAML config with sensible defaults.
        Fixed to work with both old and new config structures.
        """
        try:
            # Check if we have the new enhanced config structure
            if hasattr(config.ml.training, 'restart'):
                restart_section = config.ml.training.restart
                restart_config = {
                    'enable_restart': restart_section.enable,
                    'auto_find_checkpoint': restart_section.auto_find_checkpoint,
                    'checkpoint_path': restart_section.checkpoint_path,
                    'save_restart_every': restart_section.save_restart_every
                }
            else:
                # Try the old way with .get() method (for DictConfig)
                training_dict = config.ml.training
                if hasattr(training_dict, 'get'):
                    restart_section = training_dict.get('restart', {})
                else:
                    # Convert to dict if it's a structured config
                    import omegaconf
                    training_dict = omegaconf.OmegaConf.to_container(training_dict, resolve=True)
                    restart_section = training_dict.get('restart', {})

                restart_config = {
                    'enable_restart': restart_section.get('enable', False),
                    'auto_find_checkpoint': restart_section.get('auto_find_checkpoint', True),
                    'checkpoint_path': restart_section.get('checkpoint_path', None),
                    'save_restart_every': restart_section.get('save_restart_every', 10)
                }

            # Validation
            if restart_config['enable_restart']:
                if restart_config['checkpoint_path'] and not os.path.exists(restart_config['checkpoint_path']):
                    logger.warning(f"Specified checkpoint path does not exist: {restart_config['checkpoint_path']}")
                    logger.warning("Will attempt auto-detection instead")
                    restart_config['checkpoint_path'] = None
                    restart_config['auto_find_checkpoint'] = True

            return restart_config

        except AttributeError as e:
            logger.warning(f"Restart configuration not found in YAML: {e}")
            logger.info("Using default restart configuration (disabled)")
            return {
                'enable_restart': False,
                'auto_find_checkpoint': True,
                'checkpoint_path': None,
                'save_restart_every': 10
            }

    # Fix 2: Update the train_with_restart method in TrainerAE class

    def train_with_restart(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            epochs: int = 10,
            restart_from_checkpoint: str = None,
            auto_find_checkpoint: bool = True,
            save_restart_every: int = 5,
            **training_kwargs
    ) -> Dict:
        """
        Train with automatic restart capability - FIXED VERSION.
        """
        start_epoch = 0

        # CRITICAL FIX: Always setup training first, then try to restore from checkpoint
        logger.info("Setting up training components...")
        self.setup_training(
            epochs=epochs,
            learning_rate=training_kwargs.get('learning_rate', 1e-3),
            weight_decay=training_kwargs.get('weight_decay', 1e-4),
            steps_per_epoch=len(train_loader)
        )
        logger.info(f"✅ Optimizer and scheduler initialized")

        # Try to restart from checkpoint AFTER optimizer is initialized
        if restart_from_checkpoint or auto_find_checkpoint:
            try:
                start_epoch, checkpoint_data = self.restart_manager.restart_training(
                    checkpoint_path=restart_from_checkpoint,
                    auto_find_latest=auto_find_checkpoint
                )
                logger.info(f"🔄 Resuming training from epoch {start_epoch}")

                # Adjust total epochs if restarting
                remaining_epochs = epochs - start_epoch
                if remaining_epochs <= 0:
                    logger.info("Training already completed!")
                    return self.history

                logger.info(f"Will train for {remaining_epochs} more epochs (until epoch {epochs})")

            except Exception as e:
                logger.warning(f"Could not restart from checkpoint: {e}")
                logger.info("Starting fresh training...")
                start_epoch = 0

        # Verify optimizer is properly initialized
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized! This should not happen.")

        logger.info(f"✅ Training setup complete. Optimizer: {type(self.optimizer).__name__}")

        # Training loop with restart checkpoints
        best_val_loss = float('inf')
        if self.history['val_loss']:
            best_val_loss = min(self.history['val_loss'])

        for epoch in range(start_epoch, epochs):
            # Training phase
            avg_train_loss = self.train_epoch(
                train_loader,
                epoch,
                training_kwargs.get('loss_weights'),
                training_kwargs.get('grad_clip_norm', 1.0)
            )

            # Validation phase
            avg_val_loss, avg_val_mse = self.validate_epoch(val_loader, epoch)

            # Update scheduler
            if self.scheduler and not hasattr(self.scheduler, 'step_size'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # Record metrics
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_mse'].append(avg_val_mse)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Save restart checkpoint periodically
            if (epoch + 1) % save_restart_every == 0:
                self.restart_manager.save_restart_checkpoint(epoch, avg_val_loss, "restart")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.restart_manager.save_restart_checkpoint(epoch, avg_val_loss, "best")

            # Regular checkpoint saving
            checkpoint_path = os.path.join(self.results_dir, f'latest_checkpoint_{int(epoch):04}.pth')
            self.save_checkpoint(checkpoint_path, epoch, avg_val_loss)

            # Intermediate plotting and evaluation
            if (epoch + 1) % self.plot_every_n_epochs == 0 or epoch == epochs - 1:
                logger.info(f"Generating intermediate plots at epoch {epoch + 1}")

                plot_path = os.path.join(self.results_dir, f'training_curves_epoch_{epoch + 1}.png')
                self.plot_training_curves(save_path=plot_path)

                if test_loader:
                    recon_path = os.path.join(self.results_dir, f'reconstructions_epoch_{epoch + 1}.png')
                    self.visualize_reconstructions(test_loader, save_path=recon_path)

                    test_metrics = self.evaluate(test_loader)
                    logger.info(f"Epoch {epoch + 1} Test Metrics - MSE: {test_metrics['test_mse']:.6f}")

            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train={avg_train_loss:.4f}, "
                f"Val={avg_val_loss:.4f}, "
                f"MSE={avg_val_mse:.4f}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
            )

        # Save final checkpoint
        self.restart_manager.save_restart_checkpoint(epochs - 1, avg_val_loss, "final")

        logger.info("Training completed!")
        return self.history

    # Fix 3: Enhanced save_checkpoint with null checks

    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float = None) -> None:
        """Save training checkpoint with enhanced null safety."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'val_loss': val_loss
        }

        # Verify we have essential components
        if self.optimizer is None:
            logger.warning("Saving checkpoint without optimizer state!")

        torch.save(checkpoint, filepath)
        self._safe_wandb_save(filepath)

    # Fix 4: Enhanced restart manager load method

    def load_restart_checkpoint(
            self,
            checkpoint_path: str,
            strict_loading: bool = True,
            validate_config: bool = True
    ) -> Tuple[int, Dict]:
        """
        Load restart checkpoint with comprehensive validation - FIXED VERSION.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading restart checkpoint: {checkpoint_path}")

        # Load checkpoint data
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.trainer.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Validate checkpoint structure
        required_keys = ['epoch', 'model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint_data]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        # Load model state
        try:
            self.trainer.model.load_state_dict(
                checkpoint_data['model_state_dict'],
                strict=strict_loading
            )
            logger.info("✅ Model state loaded successfully")
        except Exception as e:
            if strict_loading:
                raise RuntimeError(f"Failed to load model state: {e}")
            else:
                logger.warning(f"Model state loaded with warnings: {e}")

        # Load optimizer state if available and optimizer exists
        if self.trainer.optimizer and 'optimizer_state_dict' in checkpoint_data:
            if checkpoint_data['optimizer_state_dict'] is not None:
                try:
                    self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    logger.info("✅ Optimizer state loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
                    logger.warning("Continuing with fresh optimizer state...")
            else:
                logger.info("No optimizer state in checkpoint")
        elif not self.trainer.optimizer:
            logger.warning("No optimizer initialized yet - optimizer state will be skipped")

        # Load scheduler state if available and scheduler exists
        if self.trainer.scheduler and 'scheduler_state_dict' in checkpoint_data:
            if checkpoint_data['scheduler_state_dict'] is not None:
                try:
                    self.trainer.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                    logger.info("✅ Scheduler state loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
                    logger.warning("Continuing with fresh scheduler state...")
            else:
                logger.info("No scheduler state in checkpoint")

        # Restore training history
        if 'history' in checkpoint_data:
            self.trainer.history = checkpoint_data['history']
            logger.info(f"✅ Training history restored ({len(self.trainer.history['train_loss'])} epochs)")

        # Print checkpoint info
        last_epoch = checkpoint_data['epoch']
        logger.info(f"✅ Checkpoint loaded successfully from epoch {last_epoch}")

        if 'val_loss' in checkpoint_data:
            logger.info(f"   Last validation loss: {checkpoint_data['val_loss']:.6f}")

        return last_epoch, checkpoint_data

    # Fix 5: Update the restart_training method in TrainerRestartManager

    def restart_training(
            self,
            checkpoint_path: str = None,
            checkpoint_dir: str = None,
            auto_find_latest: bool = True,
            **training_kwargs
    ) -> Tuple[int, Dict]:
        """
        Convenient method to restart training from a checkpoint - FIXED VERSION.
        """
        # Determine checkpoint to load
        if checkpoint_path is None and auto_find_latest:
            checkpoint_path = self.find_latest_checkpoint(checkpoint_dir)

        if checkpoint_path is None:
            raise ValueError("No checkpoint specified and none found automatically")

        # CRITICAL: Check if optimizer exists before loading checkpoint
        if self.trainer.optimizer is None:
            logger.warning("Optimizer not initialized yet. Checkpoint loading may fail for optimizer state.")

        # Load the checkpoint
        last_epoch, checkpoint_data = self.load_restart_checkpoint(checkpoint_path)

        # Calculate starting epoch for continuation
        start_epoch = last_epoch + 1

        logger.info(f"🔄 Training will restart from epoch {start_epoch}")

        return start_epoch, checkpoint_data
