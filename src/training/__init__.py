"""
Training Module

New modular training architecture with:
- BaseTrainer: Core training loop
- Adapters: Model-specific data handling (CNN, Transformer)
- Callbacks: Extensible training hooks (checkpointing, visualization, logging)
"""

from .base_trainer import BaseTrainer
from .adapters import BaseAdapter, CNNAdapter
from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    VisualizationCallback,
    WandBCallback
)

__all__ = [
    'BaseTrainer',
    'BaseAdapter',
    'CNNAdapter',
    'Callback',
    'CallbackList',
    'CheckpointCallback',
    'VisualizationCallback',
    'WandBCallback'
]
