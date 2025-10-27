import numpy as np

import os

import pickle

import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

from utils.utils import load_config
from utils.seed import set_seed
from models.dev.AEcnn3d import Autoencoder3DCNN
from data.loader import create_filtered_split_datasets
from training.trainers.trainer_ae import AETrainer


import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")


def load_or_create_datasets(load_data_pickle_flag, train_path, val_path, test_path, dataset_parameters):
    if load_data_pickle_flag:
        logger.info("Loading datasets from pickle files...")
        with open(train_path, 'rb') as f:
            train_ds = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_ds = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_ds = pickle.load(f)
    else:
        logger.info("Initializing train, validation and test dataset from metadata file...")
        train_ds, test_ds, val_ds = create_filtered_split_datasets(**dataset_parameters)
        logger.info("Saving datasets in pickle files...")
        with open(train_path, 'wb') as f:
            pickle.dump(train_ds, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_ds, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_ds, f)

    return train_ds, val_ds, test_ds


def main(config_file='config.yaml'):
    # ---------------------------------------------
    # Setup Wandb
    logger.info("Setup logging in wandb.ai...")
    config = load_config(config_file)

    # ---------------------------------------------
    # Set Seed Behaviour
    if config.global_setting.run.behaviour == 'deterministic':
        seed = config.global_setting.run.config.deterministic.seed
        set_seed(seed)
        np_seed_generator = np.random.default_rng(seed=seed)
        logger.warning("Pipline is set to deterministic...")
    else:
        logger.warning("Pipline is set to probabilistic...")

    # ---------------------------------------------
    # Set Operation Modes
    operation_mode = config.global_setting.run.mode
    debug_level = config.global_setting.run.config.debug.level if operation_mode == 'debug' else None

    # Set Datahandling
    load_data_pickle_flag = config.ml.loading.load_data_pickle
    pickle_path = config.ml.dataset.data_root
    train_path = os.path.join(pickle_path, 'train_ds.pkl')
    val_path = os.path.join(pickle_path, 'val_ds.pkl')
    test_path = os.path.join(pickle_path, 'test_ds.pkl')

    # Set Dataset Definitions
    dataset_parameters = config.ml.dataset

    # Set Network Dimensions
    model_parameters = config.ml.model



    # # ---------------------------------------------
    # # Load Dataset
    train_ds, val_ds, test_ds = load_or_create_datasets(load_data_pickle_flag, train_path, val_path, test_path, dataset_parameters)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # Each item is already a batch of 200
        shuffle=True,  # Shuffle batches across epochs
        num_workers=2,  # Parallel loading
        pin_memory=True,  # Faster GPU transfer
        drop_last=False  # Keep last incomplete batch
    )

    sequence_length = test_ds[0].shape[1]
    channels_nbr = test_ds[0].shape[2]
    height = test_ds[0].shape[3]
    width = test_ds[0].shape[4]


    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,  # No shuffling for validation
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,  # No shuffling for test
        num_workers=2,
        pin_memory=True
    )
    logger.info("DONE.")

    # ---------------------------------------------
    # Load model
    logger.info("Initializing model...")
    model = Autoencoder3DCNN(
        input_sequences=sequence_length,
        input_channels=channels_nbr,
        input_height=height,
        input_width=width,
        **model_parameters
    )
    logger.info("DONE.")


    # ---------------------------------------------
    #### Start Training
    logger.info("Start training...")
    AETrainer(model=model, Xy_train=train_loader, Xy_val=val_loader, config=config)
    logger.info("DONE.")



if __name__ == '__main__':
    main(config_file='/config/config.yaml')

