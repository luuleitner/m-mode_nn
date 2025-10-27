import os
import glob
from git import Repo
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional, Any

from training.trainers.restart_ae import TrainerRestartManager
from include.dasIT.dasIT.features.signal import logcompression

import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")


class TrainerAE:
    """
    Autoencoder trainer
    """

    def __init__(
            self,
            model: torch.nn.Module,
            device: str = 'cuda',
            use_wandb: bool = True,
            wandb_project: str = "autoencoder-training",
            wandb_config: Optional[Dict] = None,
            plot_every_n_epochs: int = 10,
            save_plots: bool = True,
            results_dir: str = "results",
            use_config_path: bool = False
    ):

        ## Class Initialization
        ####################################################################
        ####################################################################
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.use_wandb = use_wandb
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_plots = save_plots

        # Path handling
        if use_config_path:
            self.results_dir = results_dir
            logger.info(f"Using config-specified results directory: {self.results_dir}")
        else:
            self.results_dir = os.path.join(
                results_dir, 'runs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            logger.info(f"Created timestamped results directory: {self.results_dir}")

        # Ensure directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config=wandb_config or {},
                reinit=True,
                dir=get_git_root()
            )
            wandb.watch(self.model)

        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'learning_rates': []
        }
        self.optimizer = None
        self.scheduler = None
        self.best_checkpoint_path = None

        # Restart manager
        self.restart_manager = TrainerRestartManager(self)

        logger.info(f"Trainer initialized on: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    # ========================================
    # PATH COORDINATION
    # ========================================

    def get_results_path(self) -> str:
        """Get the current results directory path."""
        return self.results_dir

    def create_checkpoint_path(self, filename: str) -> str:
        """Create a full checkpoint path."""
        return os.path.join(self.results_dir, filename)

    def get_all_checkpoint_paths(self) -> Dict[str, List[str]]:
        """Get all checkpoint paths organized by type."""
        checkpoint_types = {'best': [], 'latest': [], 'restart': [], 'emergency': []}

        if not os.path.exists(self.results_dir):
            return checkpoint_types

        pattern = os.path.join(self.results_dir, "*.pth")
        all_checkpoints = glob.glob(pattern)

        for filepath in all_checkpoints:
            filename = os.path.basename(filepath).lower()

            if 'best' in filename:
                checkpoint_types['best'].append(filepath)
            elif 'latest' in filename:
                checkpoint_types['latest'].append(filepath)
            elif 'restart' in filename:
                checkpoint_types['restart'].append(filepath)
            elif any(word in filename for word in ['emergency', 'error', 'interrupt']):
                checkpoint_types['emergency'].append(filepath)
            else:
                checkpoint_types['latest'].append(filepath)

        # Sort by modification time (newest first)
        for checkpoint_type in checkpoint_types:
            checkpoint_types[checkpoint_type].sort(key=lambda x: os.path.getmtime(x), reverse=True)

        return checkpoint_types

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the most recent checkpoint."""
        if hasattr(self, 'best_checkpoint_path') and self.best_checkpoint_path and os.path.exists(
                self.best_checkpoint_path):
            return self.best_checkpoint_path

        if not os.path.exists(self.results_dir):
            return None

        checkpoint_pattern = os.path.join(self.results_dir, "*.pth")
        checkpoints = glob.glob(checkpoint_pattern)

        if not checkpoints:
            return None

        return max(checkpoints, key=os.path.getmtime)


    # ========================================
    # TRAINING SETUP
    # ========================================
    def setup_training(
            self,
            epochs: int,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            optimizer_type: str = 'adamw',
            scheduler_type: str = 'plateau',
            steps_per_epoch: int = None
    ) -> None:
        """Setup optimizer and scheduler for training."""
        logger.info(f"Setting up training: optimizer={optimizer_type}, scheduler={scheduler_type}")

        # Setup optimizer
        optimizer_type = optimizer_type.lower()
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                       momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info(f"✅ Optimizer created: {type(self.optimizer).__name__}")

        # Setup scheduler
        if scheduler_type:
            scheduler_type = scheduler_type.lower()

            if scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
                )
            elif scheduler_type == 'onecycle' and steps_per_epoch:
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs // 3, gamma=0.1)
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}")
                self.scheduler = None
        else:
            self.scheduler = None

        if self.scheduler:
            logger.info(f"✅ Scheduler created: {type(self.scheduler).__name__}")
        else:
            logger.info("No scheduler configured")

        logger.info("✅ Training setup completed successfully")


    def setup_training_with_config(
            self,
            epochs: int,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            steps_per_epoch: int = None,
            lr_scheduler_config: Dict = None
    ) -> None:
        """Setup optimizer and scheduler using YAML-style config."""
        # Setup optimizer (AdamW as default)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Setup scheduler based on config
        if lr_scheduler_config:
            if 'factor' in lr_scheduler_config and 'patience' in lr_scheduler_config:
                # ReduceLROnPlateau scheduler - FIXED: removed verbose parameter
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=lr_scheduler_config.get('factor', 0.5),
                    patience=lr_scheduler_config.get('patience', 10),
                    threshold=lr_scheduler_config.get('threshold', 1e-4),
                    min_lr=lr_scheduler_config.get('min_lr', 1e-6)
                )
            elif steps_per_epoch:
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    # ========================================
    # LOSS COMPUTATION
    # ========================================

    def compute_loss(
            self,
            reconstruction: torch.Tensor,
            target: torch.Tensor,
            embedding: torch.Tensor,
            loss_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with multiple components."""
        if loss_weights is None:
            loss_weights = {'mse_weight': 0.8, 'l1_weight': 0.2, 'embedding_reg': 0.001}

        mse_loss = F.mse_loss(reconstruction, target)
        l1_loss = F.l1_loss(reconstruction, target)
        embedding_reg = embedding.pow(2).mean()

        total_loss = (
                loss_weights['mse_weight'] * mse_loss +
                loss_weights['l1_weight'] * l1_loss +
                loss_weights['embedding_reg'] * embedding_reg
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'embedding_reg': embedding_reg.item()
        }

        return total_loss, loss_dict

    # ========================================
    # TRAINING EPOCHS
    # ========================================

    def train_epoch(self, train_loader, epoch, loss_weights=None, grad_clip_norm=1.0) -> float:
        """Train for one epoch."""
        if self.optimizer is None:
            raise RuntimeError(f"Optimizer is None at epoch {epoch + 1}! Call setup_training() first.")

        # Set model to train mode
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

        for batch_idx, batch in enumerate(pbar):
            data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            # 1. FORWARD PASS
            reconstruction, embedding = self.model(data)
            loss, loss_dict = self.compute_loss(reconstruction, data, embedding, loss_weights)

            # 2. Clear accumulated gradients in buffer:
                # WITHOUT zero_grad() - Gradients accumulate:
                # loss1.backward()  # param.grad = 2.5
                # loss2.backward()  # param.grad = 2.5 + 1.3 = 3.8 (ACCUMULATED!)
                #
                # WITH zero_grad() - Fresh gradients each time:
                # optimizer.zero_grad()  # param.grad = None
                # loss1.backward()  # param.grad = 2.5
                # optimizer.zero_grad()  # param.grad = None
                # loss2.backward()  # param.grad = 1.3 (fresh calculation)
            self.optimizer.zero_grad()
            # 3. BACKWARD PASS (GRADIENT COMPUTATION)
            loss.backward()

            # 4. gradient norm clipping (training stability which works against exploding gradients):
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)

            # 5. PARAMETER UPDATE
            self.optimizer.step()

            # Update step-based scheduler
            if self.scheduler and hasattr(self.scheduler, 'step_size'):
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log to WandB
            if self.use_wandb:
                step = epoch * len(train_loader) + batch_idx
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mse_loss': loss_dict['mse_loss'],
                    'train/l1_loss': loss_dict['l1_loss'],
                    'train/embedding_reg': loss_dict['embedding_reg'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step': step
                })

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader, epoch) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_mse = 0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                reconstruction, embedding = self.model(data)

                _, loss_dict = self.compute_loss(reconstruction, data, embedding)
                mse = F.mse_loss(reconstruction, data)

                total_loss += loss_dict['total_loss']
                total_mse += mse.item()

        avg_loss = total_loss / len(val_loader)
        avg_mse = total_mse / len(val_loader)

        if self.use_wandb:
            wandb.log({'val/loss': avg_loss, 'val/mse': avg_mse, 'epoch': epoch})

        return avg_loss, avg_mse

    # ========================================
    # MAIN TRAINING METHODS
    # ========================================

    def train(
            self,
            train_loader,
            val_loader,
            test_loader=None,
            epochs: int = 10,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            loss_weights=None,
            grad_clip_norm: float = 1.0,
            save_best: bool = True,
            lr_scheduler_config=None
    ):
        """Complete training pipeline - requires external setup."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized! Call setup_training() first.")

        best_val_loss = float('inf')
        logger.info("Starting training...")

        for epoch in range(epochs):
            # Training and validation
            avg_train_loss = self.train_epoch(train_loader, epoch, loss_weights, grad_clip_norm)
            avg_val_loss, avg_val_mse = self.validate_epoch(val_loader, epoch)

            # Update epoch-based scheduler
            if self.scheduler and not hasattr(self.scheduler, 'step_size'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # Record metrics
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_mse'].append(avg_val_mse)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Save regular checkpoint
            checkpoint_path = self.create_checkpoint_path(f'latest_checkpoint{epoch:04d}.pth')
            self.save_checkpoint(checkpoint_path, epoch, avg_val_loss)

            # Intermediate plotting and evaluation
            if (epoch + 1) % self.plot_every_n_epochs == 0 or epoch == epochs - 1:
                logger.info(f"Generating intermediate plots at epoch {epoch + 1}")

                plot_path = self.create_checkpoint_path(f'training_curves_epoch_{epoch + 1}.png')
                self.plot_training_curves(save_path=plot_path)

                if test_loader:
                    recon_path = self.create_checkpoint_path(f'reconstructions_epoch_{epoch + 1}.png')
                    self.visualize_reconstructions(test_loader, save_path=recon_path)
                    test_metrics = self.evaluate(test_loader)
                    logger.info(f"Epoch {epoch + 1} Test Metrics - MSE: {test_metrics['test_mse']:.6f}")

            # Save best model
            if save_best and avg_val_loss < best_val_loss:
                best_checkpoint_path = self.create_checkpoint_path(
                    f'best_checkpoint_epoch_{epoch:04d}_loss_{avg_val_loss:.6f}.pth'
                )

                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_loss': avg_val_loss,
                    'history': self.history
                }, best_checkpoint_path)

                # Remove previous best checkpoint
                if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                    try:
                        os.remove(self.best_checkpoint_path)
                        logger.info(f"Removed previous best checkpoint")
                    except Exception as e:
                        logger.warning(f"Failed to remove previous best checkpoint: {e}")

                self.best_checkpoint_path = best_checkpoint_path
                self._safe_wandb_save(best_checkpoint_path)
                logger.info(f"New best model saved: loss {avg_val_loss:.6f} at epoch {epoch + 1}")

            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, "
                f"MSE={avg_val_mse:.4f}, LR={self.optimizer.param_groups[0]['lr']:.2e}"
            )

        logger.info("Training completed!")
        return self.history

    def train_with_restart(
            self,
            train_loader,
            val_loader,
            test_loader=None,
            epochs: int = 10,
            restart_from_checkpoint: str = None,
            auto_find_checkpoint: bool = True,
            save_restart_every: int = 5,
            **training_kwargs
    ):
        """Train with restart capability."""
        start_epoch = 0

        # Setup training first
        logger.info("Setting up training components...")
        self.setup_training(
            epochs=epochs,
            learning_rate=training_kwargs.get('learning_rate', 1e-3),
            weight_decay=training_kwargs.get('weight_decay', 1e-4),
            optimizer_type=training_kwargs.get('optimizer_type', 'adamw'),
            scheduler_type=training_kwargs.get('scheduler_type', 'plateau'),
            steps_per_epoch=len(train_loader)
        )

        # Try to restart from checkpoint
        if restart_from_checkpoint or auto_find_checkpoint:
            try:
                start_epoch, _ = self.restart_manager.restart_training(
                    checkpoint_path=restart_from_checkpoint,
                    auto_find_latest=auto_find_checkpoint
                )
                logger.info(f"🔄 Resuming training from epoch {start_epoch}")

                remaining_epochs = epochs - start_epoch
                if remaining_epochs <= 0:
                    logger.info("Training already completed!")
                    return self.history

            except Exception as e:
                logger.warning(f"Could not restart from checkpoint: {e}")
                start_epoch = 0

        # Training loop with restart checkpoints
        best_val_loss = float('inf')
        if self.history['val_loss']:
            best_val_loss = min(self.history['val_loss'])

        for epoch in range(start_epoch, epochs):
            avg_train_loss = self.train_epoch(
                train_loader, epoch,
                training_kwargs.get('loss_weights'),
                training_kwargs.get('grad_clip_norm', 1.0)
            )
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

            # Regular checkpoint
            checkpoint_path = self.create_checkpoint_path(f'latest_checkpoint_{epoch:04d}.pth')
            self.save_checkpoint(checkpoint_path, epoch, avg_val_loss)

            # Intermediate plotting
            if (epoch + 1) % self.plot_every_n_epochs == 0 or epoch == epochs - 1:
                plot_path = self.create_checkpoint_path(f'training_curves_epoch_{epoch + 1}.png')
                self.plot_training_curves(save_path=plot_path)

                if test_loader:
                    recon_path = self.create_checkpoint_path(f'reconstructions_epoch_{epoch + 1}.png')
                    self.visualize_reconstructions(test_loader, save_path=recon_path)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: Train={avg_train_loss:.4f}, "
                f"Val={avg_val_loss:.4f}, MSE={avg_val_mse:.4f}"
            )

        # Save final checkpoint
        self.restart_manager.save_restart_checkpoint(epochs - 1, avg_val_loss, "final")
        logger.info("Training completed!")
        return self.history

    # ========================================
    # EVALUATION
    # ========================================

    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        total_mse = total_mae = total_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                reconstruction, _ = self.model(data)

                mse = F.mse_loss(reconstruction, data, reduction='sum')
                mae = F.l1_loss(reconstruction, data, reduction='sum')
                batch_size = data.size(0)

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += batch_size

        metrics = {
            'test_mse': total_mse / total_samples,
            'test_mae': total_mae / total_samples,
            'test_samples': total_samples
        }

        logger.info(f"Test Results - MSE: {metrics['test_mse']:.6f}, MAE: {metrics['test_mae']:.6f}")

        if self.use_wandb:
            wandb.log(metrics)

        return metrics

    # ========================================
    # VISUALIZATION
    # ========================================

    def plot_training_curves(self, save_path: str = 'training_curves.png') -> None:
        """Plot and save training curves."""
        if not self.history['train_loss']:
            logger.warning("No training history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs_range = range(1, len(self.history['train_loss']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs_range, self.history['train_loss'], 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(epochs_range, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE curve
        axes[0, 1].plot(epochs_range, self.history['val_mse'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('Validation MSE')
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs_range, self.history['learning_rates'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

        # Loss difference
        if len(self.history['train_loss']) > 1:
            loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
            axes[1, 1].plot(epochs_range, loss_diff, 'orange', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Loss - Train Loss')
            axes[1, 1].set_title('Overfitting Monitor')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if self.use_wandb:
            try:
                wandb.log({"training_curves": wandb.Image(save_path)})
                wandb.save(save_path, base_path=self.results_dir)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")

        plt.close()

    def visualize_reconstructions(self, test_loader, n_samples=3, time_step=5, channel=0,
                                  save_path='reconstruction_comparison.png'):
        """Visualize model reconstructions vs originals."""
        self.model.eval()
        test_samples = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= n_samples:
                    break

                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                reconstruction, _ = self.model(data)

                test_samples.append({
                    'original': data.cpu().numpy(),
                    'reconstruction': reconstruction.cpu().numpy()
                })

        if not test_samples:
            logger.warning("No test samples available for visualization")
            return

        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(test_samples):
            original = sample['original'][0, time_step, channel]  # [132, 5]
            original = logcompression(original, dbrange=65)
            reconstruction = sample['reconstruction'][0, time_step, channel]  # [132, 5]
            reconstruction = logcompression(reconstruction, dbrange=65)

            # Original
            im1 = axes[i, 0].imshow(original, aspect='auto', cmap='gray', extent=[0, 132, 0, 5], origin='lower')
            axes[i, 0].set_title(f'Original Sample {i + 1}')
            axes[i, 0].set_xlabel('Scanline')
            axes[i, 0].set_ylabel('Sample (Depth)')
            plt.colorbar(im1, ax=axes[i, 0])

            # Reconstruction
            im2 = axes[i, 1].imshow(reconstruction, aspect='auto', cmap='gray', extent=[0, 132, 0, 5], origin='lower')
            axes[i, 1].set_title(f'Reconstruction')
            axes[i, 0].set_xlabel('Scanline')
            axes[i, 0].set_ylabel('Sample (Depth)')
            plt.colorbar(im2, ax=axes[i, 1])

            # Difference map
            diff = np.abs(original - reconstruction)
            im3 = axes[i, 2].imshow(diff, aspect='auto', cmap='hot', extent=[0, 132, 0, 5], origin='lower')
            mse = np.mean(diff ** 2)
            axes[i, 2].set_title(f'Absolute Difference (MSE: {mse:.4f})')
            axes[i, 2].set_xlabel('Depth (pixels)')
            axes[i, 2].set_ylabel('Scanline')
            plt.colorbar(im3, ax=axes[i, 2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if self.use_wandb:
            try:
                wandb.log({"reconstructions": wandb.Image(save_path)})
                wandb.save(save_path, base_path=self.results_dir)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")

        plt.close()

    def visualize_scanlines(self, test_loader, n_samples=2, n_scanlines=3, time_step=5, channel=0,
                            save_path='scanline_comparison.png'):
        """Visualize individual scanline comparisons."""
        self.model.eval()
        test_samples = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= n_samples:
                    break

                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                reconstruction, _ = self.model(data)

                test_samples.append({
                    'original': data.cpu().numpy(),
                    'reconstruction': reconstruction.cpu().numpy()
                })

        if not test_samples:
            logger.warning("No test samples available for scanline visualization")
            return

        fig, axes = plt.subplots(n_samples, n_scanlines, figsize=(15, 6 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        if n_scanlines == 1:
            axes = axes.reshape(-1, 1)

        for i, sample in enumerate(test_samples):
            for w in range(n_scanlines):
                ax = axes[i, w]

                original = sample['original'][0, time_step, channel, :, w]
                reconstructed = sample['reconstruction'][0, time_step, channel, :, w]

                ax.plot(original, 'b-', label='Original', linewidth=2)
                ax.plot(reconstructed, 'r--', label='Reconstruction', linewidth=2)
                ax.set_title(f'Sample {i + 1}, Scanline {w + 1}')
                ax.set_xlabel('Depth (pixels)')
                ax.set_ylabel('Intensity')
                ax.legend()
                ax.grid(True, alpha=0.3)

                mse = np.mean((original - reconstructed) ** 2)
                ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if self.use_wandb:
            try:
                wandb.log({"scanlines": wandb.Image(save_path)})
                wandb.save(save_path, base_path=self.results_dir)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")

        plt.close()

    def visualize_embeddings(self, test_loader, max_samples=50, save_path='embedding_analysis.png'):
        """Visualize embedding statistics and distributions."""
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= max_samples:
                    break

                data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                _, embedding = self.model(data)
                all_embeddings.append(embedding.cpu().numpy())

        if not all_embeddings:
            logger.warning("No embeddings available for visualization")
            return

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Embedding magnitude over time (if temporal dimension exists)
        if len(all_embeddings.shape) > 2:  # [N, T, D]
            embedding_norms = np.linalg.norm(all_embeddings, axis=2)
            mean_norms = np.mean(embedding_norms, axis=0)
            std_norms = np.std(embedding_norms, axis=0)

            time_steps = range(1, len(mean_norms) + 1)
            axes[0].plot(time_steps, mean_norms, 'b-', linewidth=2)
            axes[0].fill_between(time_steps, mean_norms - std_norms, mean_norms + std_norms, alpha=0.3, color='blue')
            axes[0].set_xlabel('Time Step')
            axes[0].set_ylabel('Embedding L2 Norm')
            axes[0].set_title('Embedding Magnitude Over Time')
        else:
            # Just plot embedding norms
            embedding_norms = np.linalg.norm(all_embeddings, axis=1)
            axes[0].hist(embedding_norms, bins=30, density=True, alpha=0.7, color='blue')
            axes[0].set_xlabel('Embedding L2 Norm')
            axes[0].set_ylabel('Density')
            axes[0].set_title('Embedding Norm Distribution')

        axes[0].grid(True, alpha=0.3)

        # Embedding value distribution
        axes[1].hist(all_embeddings.flatten(), bins=50, density=True, alpha=0.7, color='green')
        axes[1].set_xlabel('Embedding Value')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Embedding Value Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if self.use_wandb:
            try:
                wandb.log({"embeddings": wandb.Image(save_path)})
                wandb.save(save_path, base_path=self.results_dir)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")

        plt.close()

    def generate_full_report(self, test_loader):
        """Generate complete training report with all visualizations."""
        logger.info("Generating full training report...")

        # Plot training curves
        plot_path = self.create_checkpoint_path('final_training_curves.png')
        self.plot_training_curves(save_path=plot_path)

        # Evaluate on test set
        test_metrics = self.evaluate(test_loader)

        # Generate visualizations
        recon_path = self.create_checkpoint_path('final_reconstructions.png')
        self.visualize_reconstructions(test_loader, save_path=recon_path)

        scanline_path = self.create_checkpoint_path('final_scanlines.png')
        self.visualize_scanlines(test_loader, save_path=scanline_path)

        embedding_path = self.create_checkpoint_path('final_embeddings.png')
        self.visualize_embeddings(test_loader, save_path=embedding_path)

        # Summary table for WandB
        if self.use_wandb:
            summary_table = wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["Final Train Loss", f"{self.history['train_loss'][-1]:.6f}"],
                    ["Final Val Loss", f"{self.history['val_loss'][-1]:.6f}"],
                    ["Final Val MSE", f"{self.history['val_mse'][-1]:.6f}"],
                    ["Test MSE", f"{test_metrics['test_mse']:.6f}"],
                    ["Test MAE", f"{test_metrics['test_mae']:.6f}"],
                    ["Total Parameters", f"{sum(p.numel() for p in self.model.parameters()):,}"]
                ]
            )
            wandb.log({"training_summary": summary_table})

        logger.info("Report generation completed!")

    # ========================================
    # CHECKPOINT MANAGEMENT
    # ========================================

    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'val_loss': val_loss
        }

        if self.optimizer is None:
            logger.warning("Saving checkpoint without optimizer state!")

        torch.save(checkpoint, filepath)
        self._safe_wandb_save(filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load training checkpoint and return last epoch."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def _safe_wandb_save(self, filepath: str, base_path: str = None):
        """Safely save file to WandB."""
        if not self.use_wandb:
            return

        try:
            if not os.path.exists(filepath):
                logger.warning(f"File does not exist for WandB save: {filepath}")
                return

            if base_path is None:
                base_path = self.results_dir

            wandb.save(filepath, base_path=base_path)
            logger.debug(f"Successfully saved to WandB: {filepath}")

        except Exception as e:
            logger.warning(f"WandB save failed for {filepath}: {e}")

    # ========================================
    # RESTART DELEGATION
    # ========================================

    def save_restart_checkpoint(self, epoch: int, val_loss: float, checkpoint_type: str = "restart") -> str:
        """Save restart checkpoint (delegated to restart manager)."""
        return self.restart_manager.save_restart_checkpoint(epoch, val_loss, checkpoint_type)

    def find_latest_checkpoint(self, checkpoint_dir: str = None) -> Optional[str]:
        """Find latest checkpoint (delegated to restart manager)."""
        return self.restart_manager.find_latest_checkpoint(checkpoint_dir)

    # ========================================
    # CLEANUP
    # ========================================

    def cleanup(self):
        """Clean up resources and finish WandB run."""
        if self.use_wandb:
            wandb.finish()
        logger.info("Trainer cleanup completed")


def get_git_root(path=None):
    """Returns the absolute path to the root of the Git repository."""
    repo_path = path or os.getcwd()
    repo = Repo(repo_path, search_parent_directories=True)
    return repo.git.rev_parse('--show-toplevel')


if __name__ == "__main__":
    import sys
    sys.path.insert(0, get_git_root())
    
    logger.info("=" * 60)
    logger.info("DEBUG UNIT TEST FOR AUTOENCODER")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 1. Create synthetic data for debugging
    def create_debug_data(n_samples=32, seq_len=10, channels=3, height=132, width=5):
        """Create simple patterns for debugging"""
        data = []
        for i in range(n_samples):
            # Create a simple pattern that should be easy to reconstruct
            pattern = torch.zeros((seq_len, channels, height, width))
            # Add a moving sine wave pattern
            for t in range(seq_len):
                for c in range(channels):
                    x = torch.linspace(0, 4*np.pi, height)
                    pattern[t, c, :, :] = torch.sin(x + t*0.5 + c*np.pi/3).unsqueeze(-1) * 0.5
            # Add small noise
            pattern += torch.randn_like(pattern) * 0.01
            data.append(pattern)
        return torch.stack(data)
    
    # 2. Test both models
    from models.cnnAE import CNNAutoencoder
    from models.transformerAE import CNNTransformerAutoencoder
    
    models_to_test = {
        'CNN': CNNAutoencoder(embedding_dim=128),
        'CNNTransformer': CNNTransformerAutoencoder(embedding_dim=128)
    }
    
    for model_name, model in models_to_test.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {model_name}")
        logger.info(f"{'='*40}")
        
        model = model.to(device)
        model.train()
        
        # Create data
        train_data = create_debug_data(64).to(device)
        val_data = create_debug_data(16).to(device)
        test_data = create_debug_data(8).to(device)
        
        # Simple DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        train_loader = DataLoader(TensorDataset(train_data), batch_size=8, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=8, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_data), batch_size=8, shuffle=False)
        
        # 3. Sanity checks
        logger.info("\n--- SANITY CHECKS ---")
        
        # Check forward pass
        with torch.no_grad():
            sample = train_data[:2]
            recon, embed = model(sample)
            
            # Check shapes
            assert recon.shape == sample.shape, f"Shape mismatch: {recon.shape} vs {sample.shape}"
            logger.info(f"✓ Output shape correct: {recon.shape}")
            
            # Check value ranges
            logger.info(f"Input range: [{sample.min():.3f}, {sample.max():.3f}]")
            logger.info(f"Output range: [{recon.min():.3f}, {recon.max():.3f}]")
            logger.info(f"Embedding shape: {embed.shape}, norm: {embed.norm():.3f}")
            
            # Check if model outputs are not NaN or Inf
            assert not torch.isnan(recon).any(), "NaN in reconstruction!"
            assert not torch.isinf(recon).any(), "Inf in reconstruction!"
            assert not torch.isnan(embed).any(), "NaN in embedding!"
            logger.info("✓ No NaN/Inf in outputs")
        
        # 4. Gradient flow check
        logger.info("\n--- GRADIENT FLOW CHECK ---")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Single training step
        sample = next(iter(train_loader))[0]
        recon, embed = model(sample)
        
        # Compute simple MSE loss
        loss = F.mse_loss(recon, sample)
        initial_loss = loss.item()
        logger.info(f"Initial loss: {initial_loss:.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'weight' in name and grad_norm > 0:
                    grad_norms[name[:30]] = grad_norm
        
        logger.info("Sample gradient norms:")
        for name, norm in list(grad_norms.items())[:5]:
            logger.info(f"  {name}: {norm:.6f}")
        
        # Check for vanishing/exploding gradients
        all_grad_norms = [g for g in grad_norms.values()]
        if all_grad_norms:
            max_grad = max(all_grad_norms)
            min_grad = min([g for g in all_grad_norms if g > 0])
            logger.info(f"Gradient range: [{min_grad:.2e}, {max_grad:.2e}]")
            
            if max_grad > 100:
                logger.warning("⚠ Possible exploding gradients!")
            elif max_grad < 1e-6:
                logger.warning("⚠ Possible vanishing gradients!")
            else:
                logger.info("✓ Gradient magnitudes look reasonable")
        
        # 5. Overfitting test on single batch
        logger.info("\n--- OVERFITTING TEST (10 steps) ---")
        model.train()
        single_batch = train_data[:8].to(device)
        
        losses = []
        for step in range(10):
            recon, embed = model(single_batch)
            loss = F.mse_loss(recon, single_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        logger.info(f"Loss progression: {losses[0]:.6f} → {losses[-1]:.6f}")
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        logger.info(f"Improvement: {improvement:.1f}%")
        
        if improvement < 10:
            logger.warning("⚠ Model not learning on single batch - check architecture!")
        else:
            logger.info("✓ Model can overfit single batch")
        
        # 6. Quick train/test comparison
        logger.info("\n--- TRAIN vs TEST ERROR ---")
        model.eval()
        with torch.no_grad():
            # Train error
            train_sample = train_data[:16].to(device)
            train_recon, _ = model(train_sample)
            train_mse = F.mse_loss(train_recon, train_sample).item()
            
            # Test error  
            test_sample = test_data.to(device)
            test_recon, _ = model(test_sample)
            test_mse = F.mse_loss(test_recon, test_sample).item()
            
        logger.info(f"Train MSE: {train_mse:.6f}")
        logger.info(f"Test MSE: {test_mse:.6f}")
        ratio = test_mse / train_mse if train_mse > 0 else float('inf')
        logger.info(f"Test/Train ratio: {ratio:.2f}")
        
        if ratio > 10:
            logger.warning("⚠ HUGE generalization gap detected!")
            logger.info("\nPossible issues:")
            logger.info("  1. Model architecture too complex")
            logger.info("  2. Training data too different from test")
            logger.info("  3. Normalization mismatch")
            logger.info("  4. Learning rate too high")
            logger.info("  5. Not enough regularization")
        
        # 7. Check for common issues
        logger.info("\n--- DIAGNOSTIC SUMMARY ---")
        if initial_loss > 10:
            logger.warning("⚠ Very high initial loss - check data scaling")
        if ratio > 5:
            logger.warning("⚠ Poor generalization - add dropout/regularization")
        if improvement < 20:
            logger.warning("⚠ Slow learning - increase learning rate or check architecture")
    
    logger.info("\n" + "="*60)
    logger.info("DEBUGGING COMPLETE")
    logger.info("="*60)