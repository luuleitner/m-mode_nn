"""
Precompute K-Fold Cross-Validation Dataset Splits

Creates multiple fold directories, each containing train/val/test pickle files.
Supports four CV strategies:
  - experiment_kfold: K-fold over experiments within participant(s)
  - session_loso: Leave-One-Session-Out
  - participant_lopo: Leave-One-Participant-Out (cross-subject generalization)
  - participant_within: Within-participant modeling (subject-specific, no cross-subject)

Usage:
    python -m src.data.precompute_kfolds --config config/config.yaml
    python -m src.data.precompute_kfolds --config config/config.yaml --strategy session_loso
    python -m src.data.precompute_kfolds --config config/config.yaml --strategy experiment_kfold --n-folds 5
    python -m src.data.precompute_kfolds --config config/config.yaml --force

Output structure:
    data_root/
    └── cv_folds/
        ├── config.json         # CV configuration used
        ├── fold_0/
        │   ├── train_ds.pkl
        │   ├── val_ds.pkl
        │   ├── test_ds.pkl
        │   └── fold_info.json  # What was held out
        ├── fold_1/
        └── ...
"""

import os
import sys
import argparse
import pickle
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from omegaconf import OmegaConf

from config.configurator import load_config
from src.data.datasets import create_filtered_split_datasets, FilteredSplitH5Dataset

import utils.logging_config as logconf
logger = logconf.get_logger("PRECOMPUTE_KFOLDS")


def get_metadata_info(metadata_path):
    """Load metadata and extract available participants, sessions, experiments."""
    df = pd.read_csv(metadata_path)

    # Standardize column names
    column_mapping = {
        'file path': 'file_path',
        'token_id local': 'token_id',
        'sequence id': 'sequence_id'
    }
    df = df.rename(columns=column_mapping)

    info = {
        'total_sequences': len(df),
        'participants': sorted(df['participant'].unique().tolist()),
        'sessions': sorted(df['session'].unique().tolist()),
        'experiments': df['file_path'].nunique(),
        'experiments_per_participant': df.groupby('participant')['file_path'].nunique().to_dict(),
        'sessions_per_participant': df.groupby('participant')['session'].nunique().to_dict(),
    }

    return df, info


def create_experiment_kfold_splits(config, metadata_df, cv_config, base_params):
    """
    Create K-fold splits over experiments within selected participant(s).

    Each fold holds out a subset of experiments for test/val.
    """
    strategy_config = cv_config.get('experiment_kfold', {})
    participant_filter = strategy_config.get('participant_filter')
    session_filter = strategy_config.get('session_filter')
    n_folds = strategy_config.get('n_folds', 5)
    test_val_split_ratio = strategy_config.get('test_val_split_ratio', 0.5)

    # Filter metadata to selected participants/sessions
    filtered_df = metadata_df.copy()

    if participant_filter:
        filtered_df = filtered_df[filtered_df['participant'].isin(participant_filter)]
        logger.info(f"Filtered to participants {participant_filter}: {len(filtered_df)} sequences")

    if session_filter:
        filtered_df = filtered_df[filtered_df['session'].isin(session_filter)]
        logger.info(f"Filtered to sessions {session_filter}: {len(filtered_df)} sequences")

    # Get unique experiments
    experiments = sorted(filtered_df['file_path'].unique().tolist())
    n_experiments = len(experiments)

    if n_experiments < n_folds:
        logger.warning(f"Only {n_experiments} experiments available, reducing n_folds from {n_folds} to {n_experiments}")
        n_folds = n_experiments

    logger.info(f"Creating {n_folds}-fold CV over {n_experiments} experiments")

    # Create K-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_params.get('random_seed', 42))

    folds = []
    for fold_idx, (train_idx, test_val_idx) in enumerate(kf.split(experiments)):
        train_experiments = [experiments[i] for i in train_idx]
        test_val_experiments = [experiments[i] for i in test_val_idx]

        fold_info = {
            'fold_idx': fold_idx,
            'strategy': 'experiment_kfold',
            'n_folds': n_folds,
            'participant_filter': participant_filter,
            'session_filter': session_filter,
            'train_experiments': train_experiments,
            'test_val_experiments': test_val_experiments,
            'test_val_split_ratio': test_val_split_ratio,
        }

        # Create dataset parameters for this fold
        fold_params = base_params.copy()
        fold_params['test_val_strategy'] = 'filter'
        fold_params['test_val_experiment_filter'] = test_val_experiments
        fold_params['test_val_split_ratio'] = test_val_split_ratio
        fold_params['global_participant_filter'] = participant_filter
        fold_params['global_session_filter'] = session_filter

        folds.append((fold_info, fold_params))

    return folds


def create_session_loso_splits(config, metadata_df, cv_config, base_params):
    """
    Create Leave-One-Session-Out splits.

    Each fold holds out one session for test/val.
    """
    strategy_config = cv_config.get('session_loso', {})
    participant_filter = strategy_config.get('participant_filter')
    session_filter = strategy_config.get('session_filter')
    test_val_split_ratio = strategy_config.get('test_val_split_ratio', 0.5)

    # Filter metadata
    filtered_df = metadata_df.copy()

    if participant_filter:
        filtered_df = filtered_df[filtered_df['participant'].isin(participant_filter)]
        logger.info(f"Filtered to participants {participant_filter}: {len(filtered_df)} sequences")

    # Get sessions to fold over
    available_sessions = sorted(filtered_df['session'].unique().tolist())

    if session_filter:
        sessions = [s for s in session_filter if s in available_sessions]
        if not sessions:
            raise ValueError(f"No sessions from {session_filter} found in data. Available: {available_sessions}")
    else:
        sessions = available_sessions

    n_sessions = len(sessions)
    logger.info(f"Creating {n_sessions}-fold LOSO CV over sessions: {sessions}")

    folds = []
    for fold_idx, holdout_session in enumerate(sessions):
        fold_info = {
            'fold_idx': fold_idx,
            'strategy': 'session_loso',
            'n_folds': n_sessions,
            'participant_filter': participant_filter,
            'holdout_session': holdout_session,
            'train_sessions': [s for s in sessions if s != holdout_session],
            'test_val_split_ratio': test_val_split_ratio,
        }

        # Create dataset parameters for this fold
        fold_params = base_params.copy()
        fold_params['test_val_strategy'] = 'filter'
        fold_params['test_val_session_filter'] = [holdout_session]
        fold_params['test_val_split_ratio'] = test_val_split_ratio
        fold_params['global_participant_filter'] = participant_filter
        # Don't set global_session_filter - we want all sessions, just hold out one

        folds.append((fold_info, fold_params))

    return folds


def create_participant_lopo_splits(config, metadata_df, cv_config, base_params):
    """
    Create Leave-One-Participant-Out splits.

    Each fold holds out one participant for test/val.
    """
    strategy_config = cv_config.get('participant_lopo', {})
    participant_filter = strategy_config.get('participant_filter')
    test_val_split_ratio = strategy_config.get('test_val_split_ratio', 0.5)

    # Get participants to fold over
    available_participants = sorted(metadata_df['participant'].unique().tolist())

    if participant_filter:
        participants = [p for p in participant_filter if p in available_participants]
        if not participants:
            raise ValueError(f"No participants from {participant_filter} found. Available: {available_participants}")
    else:
        participants = available_participants

    n_participants = len(participants)
    logger.info(f"Creating {n_participants}-fold LOPO CV over participants: {participants}")

    folds = []
    for fold_idx, holdout_participant in enumerate(participants):
        fold_info = {
            'fold_idx': fold_idx,
            'strategy': 'participant_lopo',
            'n_folds': n_participants,
            'holdout_participant': holdout_participant,
            'train_participants': [p for p in participants if p != holdout_participant],
            'test_val_split_ratio': test_val_split_ratio,
        }

        # Create dataset parameters for this fold
        fold_params = base_params.copy()
        fold_params['test_val_strategy'] = 'filter'
        fold_params['test_val_participant_filter'] = [holdout_participant]
        fold_params['test_val_split_ratio'] = test_val_split_ratio
        # Don't set global_participant_filter - we want all participants, just hold out one

        folds.append((fold_info, fold_params))

    return folds


def create_participant_within_splits(config, metadata_df, cv_config, base_params):
    """
    Create within-participant splits with optional nested K-fold CV.

    When inner_folds > 1: Creates K folds per participant (nested CV)
    When inner_folds = 1 or None: Creates single split per participant (legacy behavior)

    This enables:
    - Robust per-subject performance estimates with uncertainty
    - Proper variance estimation (intra-subject and inter-subject)
    - Statistical comparison between subjects

    Output: N_participants × inner_folds total folds
    """
    strategy_config = cv_config.get('participant_within', {})
    participant_filter = strategy_config.get('participant_filter')
    inner_folds = strategy_config.get('inner_folds', 1) or 1  # Default to 1 (single split)
    test_val_split_ratio = strategy_config.get('test_val_split_ratio', 0.5)
    # Legacy support: if train_ratio specified and inner_folds=1, use it
    train_ratio = strategy_config.get('train_ratio', None)

    # Get participants to create folds for
    available_participants = sorted(metadata_df['participant'].unique().tolist())

    # Handle participant_filter: null, "all", or list of IDs
    if participant_filter is None or participant_filter == 'all':
        participants = available_participants
    elif isinstance(participant_filter, list):
        participants = [p for p in participant_filter if p in available_participants]
        if not participants:
            raise ValueError(f"No participants from {participant_filter} found. Available: {available_participants}")
    else:
        # Single participant ID
        if participant_filter in available_participants:
            participants = [participant_filter]
        else:
            raise ValueError(f"Participant {participant_filter} not found. Available: {available_participants}")

    n_participants = len(participants)
    total_folds = n_participants * inner_folds

    logger.info(f"Creating nested within-participant CV")
    logger.info(f"  Participants: {n_participants} ({participants})")
    logger.info(f"  Inner folds per participant: {inner_folds}")
    logger.info(f"  Total folds: {total_folds}")
    logger.info(f"  Val/Test split ratio: {test_val_split_ratio:.0%}/{1-test_val_split_ratio:.0%}")

    folds = []
    global_fold_idx = 0

    for participant in participants:
        # Get this participant's experiments
        participant_df = metadata_df[metadata_df['participant'] == participant]
        participant_experiments = sorted(participant_df['file_path'].unique().tolist())
        n_experiments = len(participant_experiments)

        if n_experiments < inner_folds:
            logger.warning(f"Participant {participant} has only {n_experiments} experiments, "
                          f"but inner_folds={inner_folds}. Reducing to {n_experiments} folds for this participant.")
            effective_inner_folds = n_experiments
        else:
            effective_inner_folds = inner_folds

        if effective_inner_folds < 2:
            logger.warning(f"Participant {participant}: Not enough experiments for K-fold, using single split")
            effective_inner_folds = 1

        logger.info(f"  Participant {participant}: {n_experiments} experiments -> {effective_inner_folds} inner folds")

        if effective_inner_folds == 1:
            # Single split mode (legacy behavior)
            if train_ratio is None:
                train_ratio = 0.8  # Default

            rng = np.random.RandomState(base_params.get('random_seed', 42) + participant)
            shuffled_experiments = rng.permutation(participant_experiments).tolist()

            n_train = max(1, int(n_experiments * train_ratio))
            train_experiments = shuffled_experiments[:n_train]
            test_val_experiments = shuffled_experiments[n_train:]

            fold_info = {
                'fold_idx': global_fold_idx,
                'strategy': 'participant_within',
                'participant': int(participant),
                'inner_fold_idx': 0,
                'inner_folds': 1,
                'n_participants': n_participants,
                'n_experiments': n_experiments,
                'train_experiments': train_experiments,
                'test_val_experiments': test_val_experiments,
                'test_val_split_ratio': test_val_split_ratio,
            }

            fold_params = base_params.copy()
            fold_params['test_val_strategy'] = 'filter'
            fold_params['test_val_experiment_filter'] = test_val_experiments
            fold_params['test_val_split_ratio'] = test_val_split_ratio
            fold_params['global_participant_filter'] = [participant]

            folds.append((fold_info, fold_params))
            global_fold_idx += 1

        else:
            # K-fold CV within this participant
            kf = KFold(n_splits=effective_inner_folds, shuffle=True,
                      random_state=base_params.get('random_seed', 42) + participant)

            experiments_array = np.array(participant_experiments)

            for inner_fold_idx, (train_idx, test_val_idx) in enumerate(kf.split(experiments_array)):
                train_experiments = experiments_array[train_idx].tolist()
                test_val_experiments = experiments_array[test_val_idx].tolist()

                fold_info = {
                    'fold_idx': global_fold_idx,
                    'strategy': 'participant_within',
                    'participant': int(participant),
                    'inner_fold_idx': inner_fold_idx,
                    'inner_folds': effective_inner_folds,
                    'n_participants': n_participants,
                    'n_experiments': n_experiments,
                    'train_experiments': train_experiments,
                    'test_val_experiments': test_val_experiments,
                    'test_val_split_ratio': test_val_split_ratio,
                }

                fold_params = base_params.copy()
                fold_params['test_val_strategy'] = 'filter'
                fold_params['test_val_experiment_filter'] = test_val_experiments
                fold_params['test_val_split_ratio'] = test_val_split_ratio
                fold_params['global_participant_filter'] = [participant]

                folds.append((fold_info, fold_params))
                global_fold_idx += 1

                logger.debug(f"    Inner fold {inner_fold_idx}: {len(train_experiments)} train, "
                           f"{len(test_val_experiments)} test/val experiments")

    logger.info(f"Created {len(folds)} total folds")
    return folds


def save_fold(fold_dir, train_ds, val_ds, test_ds, fold_info):
    """Save a single fold's datasets and info."""
    os.makedirs(fold_dir, exist_ok=True)

    # Save datasets
    with open(os.path.join(fold_dir, 'train_ds.pkl'), 'wb') as f:
        pickle.dump(train_ds, f)

    with open(os.path.join(fold_dir, 'val_ds.pkl'), 'wb') as f:
        pickle.dump(val_ds, f)

    with open(os.path.join(fold_dir, 'test_ds.pkl'), 'wb') as f:
        pickle.dump(test_ds, f)

    # Save fold info
    with open(os.path.join(fold_dir, 'fold_info.json'), 'w') as f:
        # Convert any non-serializable types
        serializable_info = {}
        for k, v in fold_info.items():
            if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                serializable_info[k] = v
            else:
                serializable_info[k] = str(v)
        json.dump(serializable_info, f, indent=2)

    return True


def precompute_kfolds(config_path, strategy=None, n_folds=None, force=False):
    """
    Main function to precompute K-fold cross-validation splits.
    """
    # Load config
    config = load_config(config_path, create_dirs=False)

    # Get paths
    data_root = config.get_train_data_root()
    if not data_root:
        logger.error("train_base_data_path not set in config")
        return False

    metadata_path = os.path.join(data_root, 'metadata.csv')
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False

    # Load CV config
    cv_config = getattr(config, 'cross_validation', None)
    if cv_config is None:
        logger.error("No 'cross_validation' section in config")
        return False

    # Convert OmegaConf to dict for easier handling
    if hasattr(cv_config, '_content') or hasattr(cv_config, 'items'):
        cv_config = OmegaConf.to_container(cv_config, resolve=True)
    else:
        cv_config = {}

    # Override strategy if provided via CLI
    if strategy:
        cv_config['strategy'] = strategy

    cv_strategy = cv_config.get('strategy', 'experiment_kfold')
    output_subdir = cv_config.get('output_subdir', 'cv_folds')

    # Override n_folds if provided via CLI (only for experiment_kfold)
    if n_folds and cv_strategy == 'experiment_kfold':
        if 'experiment_kfold' not in cv_config:
            cv_config['experiment_kfold'] = {}
        cv_config['experiment_kfold']['n_folds'] = n_folds

    # Output directory
    output_dir = os.path.join(data_root, output_subdir)

    # Check if folds already exist
    if os.path.exists(output_dir) and not force:
        existing_folds = [d for d in os.listdir(output_dir) if d.startswith('fold_')]
        if existing_folds:
            logger.info(f"CV folds already exist at {output_dir}")
            logger.info(f"Found {len(existing_folds)} folds. Use --force to overwrite.")
            return True

    # Load metadata
    logger.info("=" * 60)
    logger.info("PRECOMPUTING K-FOLD CROSS-VALIDATION SPLITS")
    logger.info("=" * 60)
    logger.info(f"Strategy: {cv_strategy}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output: {output_dir}")

    metadata_df, meta_info = get_metadata_info(metadata_path)

    logger.info(f"\nDataset info:")
    logger.info(f"  Total sequences: {meta_info['total_sequences']}")
    logger.info(f"  Participants: {meta_info['participants']}")
    logger.info(f"  Sessions: {meta_info['sessions']}")
    logger.info(f"  Experiments: {meta_info['experiments']}")

    # Get base dataset parameters from config
    dataset_config = config.get_dataset_parameters()
    base_params = {
        'metadata_file': metadata_path,
        'target_batch_size': dataset_config.get('target_batch_size', 50),
        'dataset_key': dataset_config.get('dataset_key', 'token'),
        'random_seed': dataset_config.get('random_seed', 42),
        'split_level': dataset_config.get('split_level', 'experiment'),
        'shuffle_experiments': dataset_config.get('shuffle_experiments', True),
        'shuffle_sequences': dataset_config.get('shuffle_sequences', True),
        'balance_classes': dataset_config.get('balance_classes', False),
        'balance_strategy': dataset_config.get('balance_strategy', 'oversample'),
        'oversample_config': dataset_config.get('oversample_config'),
        'data_root': dataset_config.get('data_root'),
    }

    # Create fold configurations based on strategy
    logger.info(f"\nCreating fold configurations...")

    if cv_strategy == 'experiment_kfold':
        folds = create_experiment_kfold_splits(config, metadata_df, cv_config, base_params)
    elif cv_strategy == 'session_loso':
        folds = create_session_loso_splits(config, metadata_df, cv_config, base_params)
    elif cv_strategy == 'participant_lopo':
        folds = create_participant_lopo_splits(config, metadata_df, cv_config, base_params)
    elif cv_strategy == 'participant_within':
        folds = create_participant_within_splits(config, metadata_df, cv_config, base_params)
    else:
        logger.error(f"Unknown CV strategy: {cv_strategy}")
        return False

    n_folds_actual = len(folds)
    logger.info(f"Created {n_folds_actual} fold configurations")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save CV config
    cv_run_config = {
        'strategy': cv_strategy,
        'n_folds': n_folds_actual,
        'created_at': datetime.now().isoformat(),
        'data_root': data_root,
        'cv_config': cv_config,
    }
    with open(os.path.join(output_dir, 'cv_config.json'), 'w') as f:
        json.dump(cv_run_config, f, indent=2, default=str)

    # Create each fold
    logger.info("\n" + "=" * 60)
    logger.info("CREATING FOLDS")
    logger.info("=" * 60)

    for fold_info, fold_params in folds:
        fold_idx = fold_info['fold_idx']

        # Create fold directory name based on strategy
        if cv_strategy == 'participant_within' and fold_info.get('inner_folds', 1) > 1:
            # Nested CV: P{participant}_fold{inner_fold_idx}
            fold_dir_name = f"P{fold_info['participant']}_fold{fold_info['inner_fold_idx']}"
        else:
            # Standard: fold_{idx}
            fold_dir_name = f'fold_{fold_idx}'

        fold_dir = os.path.join(output_dir, fold_dir_name)

        logger.info(f"\n--- Fold {fold_idx}/{n_folds_actual - 1} ({fold_dir_name}) ---")

        if cv_strategy == 'experiment_kfold':
            logger.info(f"Holdout experiments: {len(fold_info['test_val_experiments'])}")
        elif cv_strategy == 'session_loso':
            logger.info(f"Holdout session: {fold_info['holdout_session']}")
        elif cv_strategy == 'participant_lopo':
            logger.info(f"Holdout participant: {fold_info['holdout_participant']}")
        elif cv_strategy == 'participant_within':
            inner_info = f", inner fold {fold_info.get('inner_fold_idx', 0)}/{fold_info.get('inner_folds', 1)-1}" if fold_info.get('inner_folds', 1) > 1 else ""
            logger.info(f"Participant: {fold_info['participant']} ({fold_info['n_experiments']} experiments{inner_info})")

        try:
            # Create datasets for this fold
            train_ds, test_ds, val_ds = create_filtered_split_datasets(
                **fold_params,
                _suppress_split_info=True,
                _suppress_metadata_info=True
            )

            # Add fold statistics to info
            fold_info['train_sequences'] = len(train_ds.metadata) if hasattr(train_ds, 'metadata') else len(train_ds)
            fold_info['val_sequences'] = len(val_ds.metadata) if hasattr(val_ds, 'metadata') else len(val_ds)
            fold_info['test_sequences'] = len(test_ds.metadata) if hasattr(test_ds, 'metadata') else len(test_ds)
            fold_info['train_batches'] = len(train_ds)
            fold_info['val_batches'] = len(val_ds)
            fold_info['test_batches'] = len(test_ds)

            # Save fold
            save_fold(fold_dir, train_ds, val_ds, test_ds, fold_info)

            logger.info(f"  Train: {fold_info['train_batches']} batches ({fold_info['train_sequences']} seq)")
            logger.info(f"  Val:   {fold_info['val_batches']} batches ({fold_info['val_sequences']} seq)")
            logger.info(f"  Test:  {fold_info['test_batches']} batches ({fold_info['test_sequences']} seq)")
            logger.info(f"  Saved to: {fold_dir}")

        except Exception as e:
            logger.error(f"Failed to create fold {fold_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION PRECOMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Strategy: {cv_strategy}")
    logger.info(f"Folds created: {n_folds_actual}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nTo train with CV, use:")
    logger.info(f"  python -m src.training.train_cnn_cls --config config/config.yaml --cv-dir {output_dir}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Precompute K-fold cross-validation dataset splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config defaults
  python -m src.data.precompute_kfolds --config config/config.yaml

  # Override strategy
  python -m src.data.precompute_kfolds --config config/config.yaml --strategy session_loso

  # K-fold with custom number of folds
  python -m src.data.precompute_kfolds --config config/config.yaml --strategy experiment_kfold --n-folds 10

  # Force overwrite existing folds
  python -m src.data.precompute_kfolds --config config/config.yaml --force
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['experiment_kfold', 'session_loso', 'participant_lopo', 'participant_within'],
        default=None,
        help='CV strategy (overrides config)'
    )
    parser.add_argument(
        '--n-folds', '-n',
        type=int,
        default=None,
        help='Number of folds for experiment_kfold strategy (overrides config)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing folds'
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1

    success = precompute_kfolds(
        config_path,
        strategy=args.strategy,
        n_folds=args.n_folds,
        force=args.force
    )

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
