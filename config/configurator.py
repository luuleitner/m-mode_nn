"""
ConfigurationManager - Simple configuration management with OmegaConf
Handles YAML configuration loading and provides comprehensive access methods
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf, DictConfig
import torch


class ConfigurationManager:
    """
    Configuration manager that uses OmegaConf directly without complex dataclasses.
    Works with existing YAML structures and provides comprehensive access methods.
    """

    def __init__(self, config_dict: DictConfig):
        self._config = config_dict
        self.global_setting = config_dict.global_setting
        self.ml = config_dict.ml
        self.wandb = config_dict.wandb
        self.preprocess = config_dict.get('preprocess', {})
        self.description = config_dict.get('description', '')

        # Add experiment section with defaults if missing
        self.experiment = config_dict.get('experiment', self._create_default_experiment())

    def _create_default_experiment(self):
        """Create default experiment configuration"""
        default_experiment = {
            'name': 'autoencoder_experiment',
            'description': 'Autoencoder training experiment',
            'version': '1.0.0',
            'resources': {
                'num_workers': 2,
                'pin_memory': True,
                'prefetch_factor': 2
            },
            'reproducibility': {
                'deterministic_algorithms': True,
                'benchmark_mode': False
            }
        }
        return OmegaConf.create(default_experiment)

    # ========================================
    # CONVENIENCE GETTERS
    # ========================================

    def get_epochs(self) -> int:
        return getattr(self.ml.training, 'epochs', 100)  # Default to 100 if missing

    def get_learning_rate(self) -> float:
        # Handle both 'lr' and 'learning_rate' keys for flexibility
        if hasattr(self.ml.training, 'lr'):
            return self.ml.training.lr
        elif hasattr(self.ml.training, 'learning_rate'):
            return self.ml.training.learning_rate
        else:
            # Return default if neither exists
            return 1e-3

    def get_weight_decay(self) -> float:
        return getattr(self.ml.training, 'weight_decay', 1e-4)

    def get_batch_size(self) -> int:
        return getattr(self.ml.dataset, 'target_batch_size', 50)  # Default to 50 if missing

    def get_device(self) -> str:
        return self.global_setting.run.device

    def get_embedding_dim(self) -> int:
        """Handle both int and list embedding dimensions"""
        emb_dim = self.ml.model.embedding_dim
        if isinstance(emb_dim, list):
            return emb_dim[-1]  # Return last element for progressive dims
        return emb_dim

    def get_train_data_root(self) -> str:
        return getattr(self.ml.dataset, 'data_root', '')  # Default to empty string if missing

    def get_process_data_root(self) -> str:
        return getattr(self.preprocess.data, 'basepath', '')  # Default to empty string if missing

    def get_checkpoint_path(self) -> str:
        path = getattr(self.ml.training.checkpointing, 'checkpoint_path', None)
        if path:
            return os.path.join(path, 'checkpoints')
        return None

    def is_deterministic(self) -> bool:
        return self.global_setting.run.behaviour == "deterministic"

    def get_seed(self) -> Optional[int]:
        if self.is_deterministic():
            return self.global_setting.run.config.deterministic.seed
        return None

    # ========================================
    # HYPERPARAMETER GETTERS
    # ========================================

    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights with defaults"""
        if hasattr(self.ml.training, 'loss_weights'):
            lw = self.ml.training.loss_weights
            return {
                'mse_weight': getattr(lw, 'mse_weight', 0.8),
                'l1_weight': getattr(lw, 'l1_weight', 0.2),
                'embedding_reg': getattr(lw, 'embedding_reg', 0.001)
            }
        else:
            # Return defaults if not in config
            return {
                'mse_weight': 0.8,
                'l1_weight': 0.2,
                'embedding_reg': 0.001
            }

    def get_restart_config(self) -> Dict[str, Any]:
        """Get restart configuration with defaults"""
        if hasattr(self.ml.training, 'restart'):
            restart = self.ml.training.restart
            return {
                'enable_restart': getattr(restart, 'enable', False),
                'auto_find_checkpoint': getattr(restart, 'auto_find_checkpoint', True),
                'checkpoint_path': getattr(restart, 'checkpoint_path', None),
                'save_restart_every': getattr(restart, 'save_restart_every', 10)
            }
        else:
            # Return defaults if not in config
            return {
                'enable_restart': False,
                'auto_find_checkpoint': True,
                'checkpoint_path': None,
                'save_restart_every': 10
            }

    def get_lr_scheduler_config(self) -> Dict[str, Any]:
        """Get LR scheduler configuration"""
        if hasattr(self.ml.training, 'lr_scheduler'):
            sched = self.ml.training.lr_scheduler
            return {
                'type': 'plateau',  # Default type
                'factor': getattr(sched, 'factor', 0.5),
                'patience': getattr(sched, 'patience', 10),
                'threshold': getattr(sched, 'threshold', 1e-4),
                'min_lr': getattr(sched, 'min_lr', 1e-6)
            }
        else:
            return {
                'type': 'plateau',
                'factor': 0.5,
                'patience': 10,
                'threshold': 1e-4,
                'min_lr': 1e-6
            }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration with defaults"""
        if hasattr(self.ml.training, 'optimizer'):
            opt = self.ml.training.optimizer
            return {
                'type': getattr(opt, 'type', 'adamw'),
                'betas': getattr(opt, 'betas', [0.9, 0.999]),
                'eps': getattr(opt, 'eps', 1e-8),
                'momentum': getattr(opt, 'momentum', 0.9)
            }
        else:
            return {
                'type': 'adamw',
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'momentum': 0.9
            }

    def get_regularization_config(self) -> Dict[str, Any]:
        """Get regularization configuration with defaults"""
        if hasattr(self.ml.training, 'regularization'):
            reg = self.ml.training.regularization
            return {
                'grad_clip_norm': getattr(reg, 'grad_clip_norm', 1.0),
                'dropout_rate': getattr(reg, 'dropout_rate', 0.0),
                'batch_norm': getattr(reg, 'batch_norm', True)
            }
        else:
            return {
                'grad_clip_norm': 1.0,
                'dropout_rate': 0.0,
                'batch_norm': True
            }

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration with defaults"""
        if hasattr(self.ml.training, 'validation'):
            val = self.ml.training.validation
            early_stopping = getattr(val, 'early_stopping', {})
            return {
                'plot_every_n_epochs': getattr(val, 'plot_every_n_epochs', 10),
                'evaluate_on_test': getattr(val, 'evaluate_on_test', True),
                'early_stopping_enable': getattr(early_stopping, 'enable', False),
                'early_stopping_patience': getattr(early_stopping, 'patience', 20),
                'early_stopping_min_delta': getattr(early_stopping, 'min_delta', 1e-6)
            }
        else:
            return {
                'plot_every_n_epochs': 10,
                'evaluate_on_test': True,
                'early_stopping_enable': False,
                'early_stopping_patience': 20,
                'early_stopping_min_delta': 1e-6
            }

    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration with defaults"""
        if hasattr(self.experiment, 'resources'):
            res = self.experiment.resources
            return {
                'num_workers': getattr(res, 'num_workers', 2),
                'pin_memory': getattr(res, 'pin_memory', True),
                'prefetch_factor': getattr(res, 'prefetch_factor', 2)
            }
        else:
            return {
                'num_workers': 2,
                'pin_memory': True,
                'prefetch_factor': 2
            }

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration with defaults"""
        return {
            'name': getattr(self.experiment, 'name', 'autoencoder_experiment'),
            'description': getattr(self.experiment, 'description', 'Autoencoder training'),
            'version': getattr(self.experiment, 'version', '1.0.0')
        }

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get WandB configuration dictionary"""
        return {
            'epochs': self.get_epochs(),
            'learning_rate': self.get_learning_rate(),
            'weight_decay': self.get_weight_decay(),
            'batch_size': self.get_batch_size(),
            'embedding_dim': self.get_embedding_dim(),
            'device': self.get_device(),
            'model_type': getattr(self.ml.model, 'type', 'OptimizedWidthReducedAutoencoder'),
            'optimizer_type': self.get_optimizer_config()['type'],
            'loss_weights': self.get_loss_weights(),
            'deterministic': self.is_deterministic(),
            'seed': self.get_seed() if self.is_deterministic() else None,
            'grad_clip_norm': self.get_regularization_config()['grad_clip_norm']
        }

    def get_dataset_parameters(self) -> Dict[str, Any]:
        """Get dataset parameters for function calls"""
        ds = self.ml.dataset
        return {
            'data_root': ds.data_root,
            'metadata_file': os.path.join(ds.data_root, 'metadata.csv') if ds.data_root else '',
            'target_batch_size': ds.target_batch_size,
            'test_val_participant_filter': getattr(ds, 'test_val_participant_filter', None),
            'test_val_session_filter': getattr(ds, 'test_val_session_filter', None),
            'test_val_experiment_filter': getattr(ds, 'test_val_experiment_filter', None),
            'test_val_label_filter': getattr(ds, 'test_val_label_filter', None),
            'test_val_split_ratio': getattr(ds, 'test_val_split_ratio', 0.5),
            'split_level': getattr(ds, 'split_level', 'sequence'),
            'random_seed': getattr(ds, 'random_seed', 353),
            'global_participant_filter': getattr(ds, 'global_participant_filter', None),
            'global_session_filter': getattr(ds, 'global_session_filter', None),
            'global_experiment_filter': getattr(ds, 'global_experiment_filter', None),
            'global_label_filter': getattr(ds, 'global_label_filter', None),
            'shuffle_experiments': getattr(ds, 'shuffle_experiments', True),
            'shuffle_sequences': getattr(ds, 'shuffle_sequences', True),
        }

    # ========================================
    # SETUP METHODS
    # ========================================

    def setup_deterministic_behavior(self):
        """Setup deterministic behavior"""
        if self.is_deterministic() and self.get_seed() is not None:
            import torch
            import random

            seed = self.get_seed()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            print(f"Pipeline set to deterministic with seed: {seed}")
            return np.random.default_rng(seed=seed)
        else:
            print("Pipeline set to probabilistic...")
            return np.random.default_rng()

    def post_init_setup(self):
        """Post-initialization setup"""
        # Set paths if not provided
        if not self.get_train_data_root() and hasattr(self.global_setting.paths, 'train_base_data_path'):
            train_path = self.global_setting.paths.train_base_data_path
            if train_path:
                self.ml.dataset.train_data_root = train_path

        if not self.get_process_data_root() and hasattr(self.global_setting.paths, 'process_base_data_path'):
            self.preprocess.data.process_data_root = self.global_setting.paths.process_base_data_path

        if not self.get_checkpoint_path() and hasattr(self.global_setting.paths, 'process_base_data_path'):
            self.ml.training.checkpoint_path = self.global_setting.paths.process_base_data_path

        # Create directories
        if self.get_checkpoint_path():
            os.makedirs(self.get_checkpoint_path(), exist_ok=True)

        if self.get_train_data_root():
            os.makedirs(self.get_train_data_root(), exist_ok=True)

        # Setup WandB API key
        if hasattr(self.wandb, 'api_key') and self.wandb.api_key:
            os.environ['WANDB_API_KEY'] = self.wandb.api_key

    # ========================================
    # UTILITY METHODS
    # ========================================

    def validate_configuration(self) -> bool:
        """Validate configuration for common issues"""
        issues = []

        # Check required paths
        if not self.get_train_data_root():
            issues.append("Data root path is not set")

        if not self.get_checkpoint_path():
            issues.append("Checkpoint path is not set")

        # Check training parameters
        if self.get_epochs() <= 0:
            issues.append("Epochs must be positive")

        if self.get_learning_rate() <= 0:
            issues.append("Learning rate must be positive")

        if self.get_batch_size() <= 0:
            issues.append("Batch size must be positive")

        # Check loss weights sum to reasonable value
        loss_weights = self.get_loss_weights()
        weight_sum = loss_weights['mse_weight'] + loss_weights['l1_weight']
        if weight_sum <= 0:
            issues.append("Main loss weights (mse + l1) must sum to positive value")

        # Check device availability
        if self.get_device() == 'cuda' and not torch.cuda.is_available():
            issues.append("CUDA device requested but not available")

        if issues:
            print("Configuration validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("✅ Configuration validation passed")
        return True

    def save_config(self, save_path: str):
        """Save current configuration to YAML file"""
        with open(save_path, 'w') as f:
            OmegaConf.save(self._config, f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return OmegaConf.to_container(self._config, resolve=True)


def load_config(config_path: str) -> ConfigurationManager:
    """
    Load configuration using ConfigurationManager
    """
    try:
        # Load YAML with OmegaConf
        yaml_config = OmegaConf.load(config_path)

        # Create configuration manager
        config = ConfigurationManager(yaml_config)

        # Post-initialization setup
        config.post_init_setup()

        return config

    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        raise


def setup_environment(config: ConfigurationManager) -> np.random.Generator:
    """Setup environment with ConfigurationManager"""
    np_generator = config.setup_deterministic_behavior()
    print_config_summary(config)
    return np_generator


def print_config_summary(config: ConfigurationManager):
    """Print comprehensive configuration summary"""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)

    print(f"\n🌐 GLOBAL SETTINGS:")
    print(f"  Run Type: {config.global_setting.run.type}")
    print(f"  Mode: {config.global_setting.run.mode}")
    print(f"  Behavior: {config.global_setting.run.behaviour}")
    print(f"  Device: {config.get_device()}")
    if config.is_deterministic():
        print(f"  Seed: {config.get_seed()}")

    print(f"\n🤖 MODEL:")
    print(f"  Embedding Dim: {config.get_embedding_dim()}")
    print(f"  Channels: {config.ml.model.channels_per_layer}")

    print(f"\n🏋️ TRAINING:")
    print(f"  Epochs: {config.get_epochs()}")
    print(f"  Learning Rate: {config.get_learning_rate()}")
    print(f"  Weight Decay: {config.get_weight_decay()}")
    print(f"  Batch Size: {config.get_batch_size()}")
    print(f"  Optimizer: {config.get_optimizer_config()['type']}")

    print(f"\n📊 LOSS WEIGHTS:")
    loss_weights = config.get_loss_weights()
    for name, weight in loss_weights.items():
        print(f"  {name}: {weight}")

    print(f"\n🔧 REGULARIZATION:")
    reg_config = config.get_regularization_config()
    print(f"  Grad Clip Norm: {reg_config['grad_clip_norm']}")
    print(f"  Dropout Rate: {reg_config['dropout_rate']}")

    print(f"\n🔄 RESTART:")
    restart = config.get_restart_config()
    print(f"  Enabled: {restart['enable_restart']}")
    if restart['enable_restart']:
        print(f"  Auto Find: {restart['auto_find_checkpoint']}")
        print(f"  Save Every: {restart['save_restart_every']} epochs")

    print(f"\n📈 WANDB:")
    print(f"  Project: {config.wandb.project}")
    print(f"  Use WandB: {config.wandb.use_wandb}")

    print("=" * 80 + "\n")


# Backward compatibility function
def get_restart_config(config: ConfigurationManager) -> Dict[str, Any]:
    """Backward compatibility for restart config"""
    return config.get_restart_config()