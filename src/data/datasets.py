import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import random
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

# Load label config from centralized source
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_label_config_path = os.path.join(_project_root, 'preprocessing/label_logic/label_config.yaml')
with open(_label_config_path) as _f:
    _label_config = yaml.safe_load(_f)

# Derive num_classes from include_noise setting
_classes_config = _label_config.get('classes', {})
_INCLUDE_NOISE = _classes_config.get('include_noise', True)
_NOISE_CLASS = _classes_config.get('noise_class', 0)
_NUM_CLASSES = 5 if _INCLUDE_NOISE else 4


def get_label_config():
    """Get label configuration for external use."""
    return {
        'include_noise': _INCLUDE_NOISE,
        'noise_class': _NOISE_CLASS,
        'num_classes': _NUM_CLASSES,
        'names': _classes_config.get('names', {}),
    }


def remap_labels_exclude_noise(labels):
    """
    Remap labels when noise is excluded.

    For hard labels: 1,2,3,4 → 0,1,2,3 (subtract 1)
    For soft labels: remove noise column (index 0), keep columns 1-4, renormalize

    Args:
        labels: numpy array with shape:
            - (batch,) for hard labels (integers in range [1, 4])
            - (batch, 5) for soft labels (probability distributions)

    Returns:
        Remapped labels:
            - (batch,) with values in range [0, 3] for hard labels
            - (batch, 4) renormalized probabilities for soft labels
    """
    # Check if soft labels (2D with 5 columns) or hard labels (1D)
    if labels.ndim == 2 and labels.shape[-1] == 5:
        # Soft labels: remove noise column (0), keep columns 1-4
        remapped = labels[:, 1:5].copy()
        # Renormalize so probabilities sum to 1
        row_sums = remapped.sum(axis=-1, keepdims=True)
        # Handle edge case: if all probability was in noise class, use uniform distribution
        zero_mask = row_sums < 1e-8
        row_sums = np.maximum(row_sums, 1e-8)
        remapped = remapped / row_sums
        # Set uniform distribution for degenerate cases (shouldn't happen with proper filtering)
        if np.any(zero_mask):
            remapped[zero_mask.squeeze(-1)] = 0.25
        return remapped
    else:
        # Hard labels: subtract 1
        return labels - 1

from src.data.augmentations import SignalAugmenter


class FilteredSplitH5Dataset(Dataset):
    """
    Dataset that filters specific data for test/validation and uses remainder for training

    Logic:
    1. Apply filters to identify test/validation candidates
    2. Split filtered data between test/validation
    3. Use remaining (non-filtered) data for training
    """

    def __init__(self,
                 metadata_file,
                 target_batch_size=200,
                 dataset_key='token',
                 data_root=None,
                 split_type='train',
                 test_val_strategy='filter',
                 test_val_participant_filter=None,
                 test_val_session_filter=None,
                 test_val_experiment_filter=None,
                 test_val_label_filter=None,
                 test_val_random_experiments=None,
                 test_val_multi_session=True,
                 test_val_split_ratio=0.5,
                 split_level='sequence',
                 random_seed=42,
                 global_participant_filter=None,
                 global_session_filter=None,
                 global_experiment_filter=None,
                 global_label_filter=None,
                 shuffle_experiments=True,
                 shuffle_sequences=True,
                 # Class balancing parameters
                 balance_classes=False,
                 balance_strategy='oversample',
                 oversample_config=None,
                 include_noise=None,
                 # Sequence grouping parameters
                 sequence_grouping=None,
                 _suppress_split_info=False,
                 _suppress_metadata_info=False):
        """
        Args:
            split_type: Which dataset to return ('train', 'test', 'val')
            test_val_strategy: 'filter' or 'random' - determines which method to use
            test_val_*_filter: Filters for test/val (used when strategy='filter')
            test_val_random_experiments: Number of experiments (used when strategy='random')
            test_val_multi_session: Prefer multi-session coverage (used when strategy='random')
            test_val_split_ratio: How to split filtered data (0.5 = 50% test, 50% val)
            split_level: Whether to split filtered data by 'sequence' or 'experiment'
            global_*_filter: Filters applied to ALL data before any splitting
            balance_classes: Enable class balancing for training set
            balance_strategy: 'oversample' or 'undersample'
            oversample_config: Configuration for oversampling:
                {
                    'method': 'augment' | 'duplicate' | 'mixed',
                    'target_ratio': 1.0,  # Match majority class
                    'augmentations': {...}  # Augmentation config
                }
            include_noise: Override for noise inclusion (None = use label_config.yaml)
            _suppress_split_info: If True, suppress printing of split information
        """

        self.target_batch_size = target_batch_size
        self.dataset_key = dataset_key
        self.data_root = Path(data_root) if data_root else None
        self.split_type = split_type
        self.random_seed = random_seed
        self.balance_classes = balance_classes
        self.balance_strategy = balance_strategy
        self.oversample_config = oversample_config or {}

        # Sequence grouping: only for train split to preserve temporal order in shuffling
        seq_cfg = sequence_grouping or {}
        self.sequence_grouping_enabled = (
            seq_cfg.get('enabled', False) and split_type == 'train'
        )
        self.seq_len = seq_cfg.get('seq_len', 15)
        self.seq_stride = seq_cfg.get('seq_stride', self.seq_len)

        # Noise class handling: use parameter or fall back to global config
        self.include_noise = include_noise if include_noise is not None else _INCLUDE_NOISE
        self.num_classes = 5 if self.include_noise else 4

        # Auto-apply label filter when noise excluded
        if not self.include_noise:
            # Filter to only include movement labels (1, 2, 3, 4)
            if global_label_filter is None:
                global_label_filter = [1, 2, 3, 4]
                if not _suppress_metadata_info:
                    print(f"Noise excluded: auto-applying global_label_filter={global_label_filter}")

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load metadata and apply global filters
        self._load_metadata(metadata_file, suppress_print=_suppress_metadata_info)
        self._apply_global_filters(
            global_participant_filter, global_session_filter,
            global_experiment_filter, global_label_filter
        )

        # Create train/test/val splits
        self._create_splits(
            test_val_strategy,
            test_val_participant_filter, test_val_session_filter,
            test_val_experiment_filter, test_val_label_filter,
            test_val_random_experiments, test_val_multi_session,
            test_val_split_ratio, split_level, _suppress_split_info, _suppress_metadata_info
        )

        # Process the requested split
        self.metadata = self.splits[split_type].copy()
        self._process_metadata(shuffle_experiments, shuffle_sequences)

        # Build batch mapping (balanced for train if enabled)
        if self.balance_classes and split_type == 'train':
            self._build_batch_mapping_balanced()
        else:
            self._build_batch_mapping()

        self._print_dataset_info()

    def _load_metadata(self, metadata_file, suppress_print=False):
        """Load and standardize metadata"""

        self.full_metadata = pd.read_csv(
            metadata_file,
            sep=',' if metadata_file.endswith('.csv') else ','
        )

        # Standardize column names
        column_mapping = {
            'file path': 'file_path',
            'token_id local': 'token_id',
            'sequence id': 'sequence_id'
        }
        self.full_metadata = self.full_metadata.rename(columns=column_mapping)

        if not suppress_print:  # ONLY PRINT IF NOT SUPPRESSED
            print(f"Loaded metadata: {len(self.full_metadata)} total sequences")

    def _apply_global_filters(self, participant_filter, session_filter,
                              experiment_filter, label_filter):  # REMOVE suppress_print parameter
        """Apply global filters to all data before splitting"""

        self.metadata = self.full_metadata.copy()
        original_len = len(self.metadata)

        if participant_filter is not None:
            self.metadata = self.metadata[
                self.metadata['participant'].isin(participant_filter)
            ]
            print(f"Global participant filter: {original_len} -> {len(self.metadata)} sequences")

        if session_filter is not None:
            self.metadata = self.metadata[
                self.metadata['session'].isin(session_filter)
            ]

            print(f"Global session filter: -> {len(self.metadata)} sequences")

        if experiment_filter is not None:
            # Support both file_path (full paths) and experiment (short names)
            if any('/' in str(e) or '.h5' in str(e) for e in experiment_filter):
                self.metadata = self.metadata[
                    self.metadata['file_path'].isin(experiment_filter)
                ]
            else:
                self.metadata = self.metadata[
                    self.metadata['experiment'].isin(experiment_filter)
                ]
            print(f"Global experiment filter: -> {len(self.metadata)} sequences")

        if label_filter is not None:
            # Support both column names: 'label_logic' or 'token label_logic'
            label_col = 'label_logic' if 'label_logic' in self.metadata.columns else 'token label_logic'
            self.metadata = self.metadata[
                self.metadata[label_col].isin(label_filter)
            ]
            print(f"Global label filter ({label_col}): -> {len(self.metadata)} sequences")

    def _create_splits(self, test_val_strategy,
                       test_val_participant_filter, test_val_session_filter,
                       test_val_experiment_filter, test_val_label_filter,
                       test_val_random_experiments, test_val_multi_session,
                       test_val_split_ratio, split_level, suppress_split_print=False,
                       suppress_metadata_print=False):
        """
        Create train/test/val splits based on explicit strategy.

        Strategy options:
            - "filter": Use test_val_*_filter parameters (combined with AND logic)
            - "random": Randomly select N experiments using test_val_random_experiments
        """

        # Start with all globally filtered data
        all_data = self.metadata.copy().reset_index(drop=True)

        # =========================================================
        # STRATEGY: RANDOM EXPERIMENT SELECTION
        # =========================================================
        if test_val_strategy == 'random':
            if test_val_random_experiments is None or test_val_random_experiments <= 0:
                raise ValueError("strategy='random' requires test_val_random_experiments > 0")

            # Support percentage (0 < value < 1) or absolute count (>= 1)
            total_experiments = all_data['file_path'].nunique()
            if 0 < test_val_random_experiments < 1:
                # Interpret as percentage
                num_experiments = max(1, int(total_experiments * test_val_random_experiments))
                if not suppress_metadata_print:
                    print(f"Using RANDOM strategy: selecting {test_val_random_experiments:.0%} = {num_experiments} experiments (of {total_experiments})")
            else:
                # Interpret as absolute count
                num_experiments = int(test_val_random_experiments)
                if not suppress_metadata_print:
                    print(f"Using RANDOM strategy: selecting {num_experiments} experiments")

            test_val_candidates = self._select_random_experiments(
                all_data, num_experiments, test_val_multi_session,
                suppress_print=suppress_metadata_print
            )

        # =========================================================
        # STRATEGY: FILTER-BASED SELECTION
        # =========================================================
        elif test_val_strategy == 'filter':
            if not suppress_metadata_print:
                print("Using FILTER strategy")

            test_val_candidates = all_data.copy()
            filters_applied = []

            if test_val_participant_filter is not None:
                test_val_candidates = test_val_candidates[
                    test_val_candidates['participant'].isin(test_val_participant_filter)
                ]
                filters_applied.append(f"participant={test_val_participant_filter}")

            if test_val_session_filter is not None:
                test_val_candidates = test_val_candidates[
                    test_val_candidates['session'].isin(test_val_session_filter)
                ]
                filters_applied.append(f"session={test_val_session_filter}")

            if test_val_experiment_filter is not None:
                # Support both file_path (full paths) and experiment (short names)
                if any('/' in str(e) or '.h5' in str(e) for e in test_val_experiment_filter):
                    # Filter values look like file paths
                    test_val_candidates = test_val_candidates[
                        test_val_candidates['file_path'].isin(test_val_experiment_filter)
                    ]
                else:
                    # Filter values look like experiment names
                    test_val_candidates = test_val_candidates[
                        test_val_candidates['experiment'].isin(test_val_experiment_filter)
                    ]
                filters_applied.append(f"experiment_filter={len(test_val_experiment_filter)} items")

            if test_val_label_filter is not None:
                # Support both column names: 'label_logic' or 'token label_logic'
                label_col = 'label_logic' if 'label_logic' in test_val_candidates.columns else 'token label_logic'
                test_val_candidates = test_val_candidates[
                    test_val_candidates[label_col].isin(test_val_label_filter)
                ]
                filters_applied.append(f"label={test_val_label_filter}")

            if not filters_applied:
                raise ValueError("strategy='filter' requires at least one test_val_*_filter to be set")

            if not suppress_metadata_print:
                print(f"  Filters (AND): {' AND '.join(filters_applied)}")

        else:
            raise ValueError(f"Unknown test_val_strategy: '{test_val_strategy}'. Use 'filter' or 'random'")

        test_val_candidates = test_val_candidates.reset_index(drop=True)

        if not suppress_metadata_print:
            print(f"Test/Val candidates after filtering: {len(test_val_candidates)} sequences")

        # Split test/val candidates
        if len(test_val_candidates) == 0:
            test_data = pd.DataFrame()
            val_data = pd.DataFrame()
        elif test_val_split_ratio == 0.0:
            # All filtered data goes to validation
            test_data = pd.DataFrame()
            val_data = test_val_candidates
        elif test_val_split_ratio == 1.0:
            # All filtered data goes to test
            test_data = test_val_candidates
            val_data = pd.DataFrame()
        else:
            # Split filtered data between test and validation
            test_data, val_data = self._split_test_val_data(
                test_val_candidates, test_val_split_ratio, split_level
            )

        # Training data = everything NOT in test/val candidates
        if len(test_val_candidates) > 0:
            # Create identifier for test/val candidates to exclude from training
            if split_level == 'experiment':
                # Exclude entire experiments
                test_val_experiments = set(test_val_candidates['file_path'].unique())
                train_data = all_data[
                    ~all_data['file_path'].isin(test_val_experiments)
                ].reset_index(drop=True)
            else:
                # Exclude specific sequences (more complex - need to match exactly)
                test_val_candidates['_temp_id'] = (
                        test_val_candidates['file_path'].astype(str) + '_' +
                        test_val_candidates['sequence_id'].astype(str) + '_' +
                        test_val_candidates['token_id'].astype(str)
                )
                all_data['_temp_id'] = (
                        all_data['file_path'].astype(str) + '_' +
                        all_data['sequence_id'].astype(str) + '_' +
                        all_data['token_id'].astype(str)
                )

                test_val_ids = set(test_val_candidates['_temp_id'])
                train_data = all_data[
                    ~all_data['_temp_id'].isin(test_val_ids)
                ].drop('_temp_id', axis=1).reset_index(drop=True)

                # Clean up temp columns
                test_val_candidates = test_val_candidates.drop('_temp_id', axis=1)
        else:
            # No test/val data, everything goes to training
            train_data = all_data

        self.splits = {
            'train': train_data,
            'test': test_data,
            'val': val_data
        }

        # ONLY PRINT IF NOT SUPPRESSED
        if not suppress_split_print:
            print(f"Final splits:")
            print(f"  - Train: {len(train_data)} sequences")
            print(f"  - Test: {len(test_data)} sequences")
            print(f"  - Val: {len(val_data)} sequences")

    def _select_random_experiments(self, all_data, num_experiments, multi_session=True,
                                      suppress_print=False):
        """
        Randomly select experiments for test/val set.

        Args:
            all_data: DataFrame with all data
            num_experiments: Number of experiments to select
            multi_session: If True and multiple sessions exist, ensure experiments
                          come from multiple sessions for better generalization

        Returns:
            DataFrame containing only the selected experiments
        """
        # Get unique experiments with their session info
        experiment_info = all_data.groupby('file_path').agg({
            'session': 'first',
            'experiment': 'first'
        }).reset_index()

        available_experiments = experiment_info['file_path'].tolist()
        num_available = len(available_experiments)

        if num_experiments > num_available:
            if not suppress_print:
                print(f"Warning: Requested {num_experiments} experiments but only "
                      f"{num_available} available. Using all.")
            num_experiments = num_available

        # Check if multi-session selection is possible and desired
        unique_sessions = experiment_info['session'].unique()
        num_sessions = len(unique_sessions)

        if multi_session and num_sessions > 1 and num_experiments >= 2:
            # Try to select experiments from multiple sessions
            selected_experiments = self._select_multi_session_experiments(
                experiment_info, num_experiments, suppress_print
            )
        else:
            # Simple random selection
            random.seed(self.random_seed)
            selected_experiments = random.sample(available_experiments, num_experiments)

        if not suppress_print:
            selected_sessions = experiment_info[
                experiment_info['file_path'].isin(selected_experiments)
            ]['session'].unique()
            print(f"Random experiment selection: {num_experiments} experiments "
                  f"from {len(selected_sessions)} session(s)")

        # Filter data to only include selected experiments
        return all_data[all_data['file_path'].isin(selected_experiments)]

    def _select_multi_session_experiments(self, experiment_info, num_experiments,
                                          suppress_print=False):
        """
        Select experiments ensuring coverage from multiple sessions.

        Strategy:
        1. Group experiments by session
        2. Distribute selection across sessions as evenly as possible
        3. Random selection within each session
        """
        random.seed(self.random_seed)

        # Group experiments by session
        session_experiments = {}
        for session in experiment_info['session'].unique():
            session_exps = experiment_info[
                experiment_info['session'] == session
            ]['file_path'].tolist()
            session_experiments[session] = session_exps

        sessions = list(session_experiments.keys())
        num_sessions = len(sessions)

        # Calculate how many experiments to take from each session
        base_per_session = num_experiments // num_sessions
        remainder = num_experiments % num_sessions

        selected = []

        # Shuffle sessions to randomize which get the extra experiments
        random.shuffle(sessions)

        for i, session in enumerate(sessions):
            available = session_experiments[session]
            # First 'remainder' sessions get one extra experiment
            take = base_per_session + (1 if i < remainder else 0)
            take = min(take, len(available))

            if take > 0:
                session_selected = random.sample(available, take)
                selected.extend(session_selected)

        # If we still need more experiments (some sessions had fewer than expected)
        while len(selected) < num_experiments:
            # Find experiments not yet selected
            all_exps = experiment_info['file_path'].tolist()
            remaining = [e for e in all_exps if e not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))

        return selected

    def _split_test_val_data(self, test_val_data, test_ratio, split_level):
        """Split the filtered test/val data between test and validation"""

        if split_level == 'sequence':
            # Simple random split of sequences
            if len(test_val_data) == 1:
                # Only one sequence - put it in test
                return test_val_data, pd.DataFrame()

            # Support both column names for stratification
            stratify_col = None
            if 'label_logic' in test_val_data.columns:
                stratify_col = test_val_data['label_logic']
            elif 'token label_logic' in test_val_data.columns:
                stratify_col = test_val_data['token label_logic']

            test_data, val_data = train_test_split(
                test_val_data,
                test_size=test_ratio,
                random_state=self.random_seed,
                stratify=stratify_col
            )

        elif split_level == 'experiment':
            # Split by experiments to prevent data leakage
            unique_experiments = test_val_data['file_path'].unique()

            if len(unique_experiments) == 1:
                # Only one experiment - put it in test
                return test_val_data, pd.DataFrame()

            test_experiments, val_experiments = train_test_split(
                unique_experiments,
                test_size=test_ratio,
                random_state=self.random_seed
            )

            test_data = test_val_data[
                test_val_data['file_path'].isin(test_experiments)
            ].reset_index(drop=True)

            val_data = test_val_data[
                test_val_data['file_path'].isin(val_experiments)
            ].reset_index(drop=True)

        return test_data, val_data

    def _process_metadata(self, shuffle_experiments, shuffle_sequences):
        """Process metadata and group by experiment.

        When sequence_grouping is enabled, tokens within each experiment are
        grouped into overlapping sequences (sliding window over token indices).
        Shuffling then operates on sequence groups instead of individual tokens,
        preserving temporal order within each sequence.
        """

        if len(self.metadata) == 0:
            self.experiment_groups = {}
            self.experiment_list = []
            return

        # Group sequences by experiment (file path)
        self.experiment_groups = {}

        for file_path, group in self.metadata.groupby('file_path'):
            # Sort by token_id to maintain sequence order
            group_sorted = group.sort_values('token_id').reset_index(drop=True)

            if self.sequence_grouping_enabled:
                group_sorted = self._apply_sequence_grouping(
                    group_sorted, shuffle_sequences
                )
            elif shuffle_sequences:
                group_sorted = group_sorted.sample(frac=1).reset_index(drop=True)

            self.experiment_groups[file_path] = group_sorted

        # Get list of experiments
        self.experiment_list = list(self.experiment_groups.keys())

        if shuffle_experiments:
            random.shuffle(self.experiment_list)

    def _apply_sequence_grouping(self, group, shuffle):
        """Group consecutive tokens into overlapping sequences via sliding window.

        Tokens may appear in multiple sequences due to overlap (seq_stride < seq_len).
        Shuffling operates on whole sequence groups, preserving temporal order within.

        Args:
            group: DataFrame of tokens from one experiment, sorted by token_id.
            shuffle: Whether to shuffle the sequence groups.

        Returns:
            DataFrame with tokens ordered by (optionally shuffled) sequence groups.
        """
        num_tokens = len(group)
        seq_len = self.seq_len
        seq_stride = self.seq_stride
        num_sequences = (num_tokens - seq_len) // seq_stride + 1

        if num_sequences <= 0:
            return group

        sequence_groups = []
        for seq_idx in range(num_sequences):
            start = seq_idx * seq_stride
            end = start + seq_len
            sequence_groups.append(group.iloc[start:end])

        if shuffle:
            random.shuffle(sequence_groups)

        return pd.concat(sequence_groups, ignore_index=True)

    def _build_batch_mapping(self):
        """Build mapping of batch_idx -> list of (file_path, sequence_metadata)"""

        self.batch_mapping = []

        if len(self.metadata) == 0:
            return

        current_batch = []
        current_batch_size = 0

        # Iterate through experiments and their sequences
        for exp_file in self.experiment_list:
            exp_sequences = self.experiment_groups[exp_file].copy()

            # Process all sequences from this experiment
            for idx, row in exp_sequences.iterrows():
                # Add sequence to current batch
                current_batch.append({
                    'file_path': exp_file,
                    'sequence_metadata': row.to_dict()
                })
                current_batch_size += 1

                # If batch is full, save it and start new batch
                if current_batch_size >= self.target_batch_size:
                    self.batch_mapping.append(current_batch)
                    current_batch = []
                    current_batch_size = 0

        # Add remaining sequences as final batch (if any)
        if current_batch:
            self.batch_mapping.append(current_batch)

    def _build_batch_mapping_balanced(self):
        """
        Build balanced batch mapping by oversampling minority classes.

        Strategy:
        1. Keep ALL majority class samples (appear exactly once)
        2. Oversample minority classes to match majority class count
        3. Apply augmentation to oversampled minority samples
        4. Shuffle and build batches with ~equal class distribution
        """
        self.batch_mapping = []

        if len(self.metadata) == 0:
            return

        # Get label column
        label_col = 'label_logic' if 'label_logic' in self.metadata.columns else 'token label_logic'
        if label_col not in self.metadata.columns:
            print("Warning: No label column found, falling back to unbalanced batching")
            self._build_batch_mapping()
            return

        # Collect all sequences with their metadata
        all_sequences = []
        for exp_file in self.experiment_list:
            exp_sequences = self.experiment_groups[exp_file]
            for idx, row in exp_sequences.iterrows():
                all_sequences.append({
                    'file_path': exp_file,
                    'sequence_metadata': row.to_dict(),
                    'label': int(row[label_col]),
                    'is_augmented': False
                })

        # Group sequences by label
        sequences_by_label = defaultdict(list)
        for seq in all_sequences:
            sequences_by_label[seq['label']].append(seq)

        # Count samples per class
        class_counts = {label: len(seqs) for label, seqs in sequences_by_label.items()}
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]

        print(f"Class distribution before balancing: {class_counts}")
        print(f"Majority class: {majority_class} with {majority_count} samples")

        # Determine oversampling method
        method = self.oversample_config.get('method', 'augment')
        target_ratio = self.oversample_config.get('target_ratio', 1.0)
        target_count = int(majority_count * target_ratio)

        # Create augmenter if needed (store as class attribute for __getitem__)
        self.augmenter = None
        if method in ['augment', 'mixed']:
            aug_config = self.oversample_config.get('augmentations', {})
            self.augmenter = SignalAugmenter(config=aug_config, seed=self.random_seed)
            print(f"Augmentation config: {self.augmenter.get_config()}")

        # Build balanced sequence pool
        balanced_sequences = []

        for label, sequences in sequences_by_label.items():
            current_count = len(sequences)

            # Add all original sequences
            balanced_sequences.extend(sequences)

            # If minority class, oversample to match target
            if current_count < target_count:
                samples_needed = target_count - current_count
                print(f"Class {label}: oversampling {current_count} -> {target_count} "
                      f"(+{samples_needed} samples)")

                # Determine how to create oversampled sequences
                if method == 'duplicate':
                    # Pure duplication
                    oversampled = self._oversample_duplicate(sequences, samples_needed)
                elif method == 'augment':
                    # All augmented
                    oversampled = self._oversample_augment(sequences, samples_needed, self.augmenter)
                elif method == 'mixed':
                    # Mix of duplicates and augmented
                    augment_ratio = self.oversample_config.get('augment_ratio', 0.5)
                    n_augment = int(samples_needed * augment_ratio)
                    n_duplicate = samples_needed - n_augment
                    oversampled = (
                        self._oversample_duplicate(sequences, n_duplicate) +
                        self._oversample_augment(sequences, n_augment, self.augmenter)
                    )
                else:
                    raise ValueError(f"Unknown oversample method: {method}")

                balanced_sequences.extend(oversampled)

        # Shuffle all sequences
        random.shuffle(balanced_sequences)

        # Report new distribution
        new_counts = Counter(seq['label'] for seq in balanced_sequences)
        augmented_counts = Counter(seq['label'] for seq in balanced_sequences if seq['is_augmented'])
        print(f"Class distribution after balancing: {dict(new_counts)}")
        print(f"Augmented samples per class: {dict(augmented_counts)}")

        # Build batches from balanced pool
        current_batch = []
        for seq in balanced_sequences:
            current_batch.append({
                'file_path': seq['file_path'],
                'sequence_metadata': seq['sequence_metadata'],
                'is_augmented': seq['is_augmented']
            })

            if len(current_batch) >= self.target_batch_size:
                self.batch_mapping.append(current_batch)
                current_batch = []

        # Add remaining as final batch
        if current_batch:
            self.batch_mapping.append(current_batch)

        # Update metadata to include is_augmented flag
        self._update_metadata_with_augmented_flag(balanced_sequences)

        print(f"Built {len(self.batch_mapping)} balanced batches "
              f"(was {len(all_sequences) // self.target_batch_size + 1} unbalanced)")

    def _oversample_duplicate(self, sequences, n_samples):
        """Create n_samples by duplicating existing sequences."""
        oversampled = []
        for i in range(n_samples):
            # Sample with replacement from original sequences
            original = random.choice(sequences)
            oversampled.append({
                'file_path': original['file_path'],
                'sequence_metadata': original['sequence_metadata'].copy(),
                'label': original['label'],
                'is_augmented': False  # Exact duplicate, not augmented
            })
        return oversampled

    def _oversample_augment(self, sequences, n_samples, augmenter):
        """Create n_samples by augmenting existing sequences."""
        oversampled = []
        for i in range(n_samples):
            # Sample with replacement from original sequences
            original = random.choice(sequences)
            # Mark as augmented - actual augmentation happens in __getitem__
            aug_metadata = original['sequence_metadata'].copy()
            aug_metadata['is_augmented'] = True
            aug_metadata['augment_seed'] = self.random_seed + i  # Unique seed per sample

            oversampled.append({
                'file_path': original['file_path'],
                'sequence_metadata': aug_metadata,
                'label': original['label'],
                'is_augmented': True
            })
        return oversampled

    def _update_metadata_with_augmented_flag(self, balanced_sequences):
        """Update metadata DataFrame to reflect balanced sequences."""
        # Create new metadata from balanced sequences
        new_metadata = []
        for seq in balanced_sequences:
            meta = seq['sequence_metadata'].copy()
            meta['is_augmented'] = seq['is_augmented']
            new_metadata.append(meta)

        self.metadata = pd.DataFrame(new_metadata)

    def __len__(self):
        return len(self.batch_mapping)

    def __getitem__(self, batch_idx):
        """
        Load and return a batch of sequences with labels.

        IMPORTANT: Order of samples in output matches order in batch_mapping.
        This is critical for correct label-sample association.
        """
        if batch_idx >= len(self.batch_mapping):
            raise IndexError(f"Batch index {batch_idx} out of range")

        batch_info = self.batch_mapping[batch_idx]
        batch_sequences = []
        batch_labels = []

        # Cache open file handles to avoid reopening same file multiple times
        file_cache = {}

        try:
            # Process items IN ORDER - preserving batch_mapping order is critical!
            for item in batch_info:
                file_path = item['file_path']
                seq_meta = item['sequence_metadata']
                is_augmented = item.get('is_augmented', False)
                augment_seed = seq_meta.get('augment_seed', None)

                # Get full file path
                if self.data_root:
                    full_path = self.data_root / file_path
                else:
                    full_path = Path(file_path)

                # Open file if not already cached
                if file_path not in file_cache:
                    file_cache[file_path] = h5py.File(full_path, 'r')

                f = file_cache[file_path]
                token_idx = int(seq_meta['token_id'])

                # Load token: [channels, height, width]
                sequence_data = f[self.dataset_key][token_idx].copy()

                # Apply augmentation if this is an augmented sample (balance augmenter)
                if is_augmented and hasattr(self, 'augmenter') and self.augmenter is not None:
                    # Set seed for reproducible augmentation
                    if augment_seed is not None:
                        np.random.seed(augment_seed)
                    sequence_data = self.augmenter(sequence_data)

                # Apply general on-the-fly augmentation (all training samples, stochastic)
                if hasattr(self, 'general_augmenter') and self.general_augmenter is not None:
                    sequence_data = self.general_augmenter(sequence_data)

                batch_sequences.append(sequence_data)

                # Load labels if available
                if 'label' in f:
                    label_data = f['label'][token_idx]
                    batch_labels.append(label_data)

        finally:
            # Always close cached file handles
            for f in file_cache.values():
                f.close()

        # Stack all sequences in the batch
        if len(batch_sequences) == 0:
            return {'tokens': torch.empty(0), 'labels': torch.empty(0)}

        final_batch = np.stack(batch_sequences, axis=0).astype(np.float32)

        # Stack labels
        if len(batch_labels) > 0:
            final_labels = np.stack(batch_labels, axis=0).astype(np.float32)

            # Remap labels when noise is excluded: 1,2,3,4 → 0,1,2,3
            if not self.include_noise:
                final_labels = remap_labels_exclude_noise(final_labels)
        else:
            final_labels = None

        return {
            'tokens': torch.from_numpy(final_batch),
            'labels': torch.from_numpy(final_labels) if final_labels is not None else None
        }

    def set_general_augmenter(self, augmenter):
        """Attach augmenter for on-the-fly training augmentation (all samples, stochastic)."""
        if self.split_type != 'train':
            raise ValueError(f"General augmenter only for train split, got '{self.split_type}'")
        self.general_augmenter = augmenter

    def get_batch_metadata(self, batch_idx):
        """Get metadata for all sequences in a specific batch"""

        if batch_idx >= len(self.batch_mapping):
            raise IndexError(f"Batch index {batch_idx} out of range")

        batch_info = self.batch_mapping[batch_idx]
        metadata_list = [item['sequence_metadata'] for item in batch_info]

        return pd.DataFrame(metadata_list)

    def get_split_info(self):
        """Get information about the current splits"""

        info = {
            'current_split': self.split_type,
            'current_split_size': len(self.metadata),
        }

        if hasattr(self, 'splits'):
            for split_name, split_data in self.splits.items():
                info[f'{split_name}_size'] = len(split_data)
                if len(split_data) > 0:
                    info[f'{split_name}_experiments'] = split_data['file_path'].nunique()
                    if 'participant' in split_data.columns:
                        info[f'{split_name}_participants'] = split_data['participant'].nunique()
                    if 'session' in split_data.columns:
                        info[f'{split_name}_sessions'] = split_data['session'].nunique()

        return info

    def _print_dataset_info(self):
        """Print dataset information"""
        noise_status = "included" if self.include_noise else "excluded (labels remapped)"
        print(f"\n{self.split_type.upper()} Dataset initialized:")
        print(f"  - Total sequences: {len(self.metadata)}")
        print(f"  - Unique experiments: {len(self.experiment_groups)}")
        print(f"  - Total batches: {len(self.batch_mapping)}")
        print(f"  - Num classes: {self.num_classes} (noise {noise_status})")
        if self.sequence_grouping_enabled:
            overlap = self.seq_len - self.seq_stride
            print(f"  - Sequence grouping: seq_len={self.seq_len}, "
                  f"stride={self.seq_stride}, overlap={overlap}")

    def get_sample_weights(self):
        """
        DEPRECATED: Use balance_classes=True in dataset config instead.

        This method computed batch-level weights for WeightedRandomSampler,
        but this approach is ineffective when all batches have similar class
        distribution. The new approach uses balanced batch construction at
        precompute time.

        Returns:
            torch.Tensor: Uniform weights (legacy compatibility)
        """
        import warnings
        warnings.warn(
            "get_sample_weights() is deprecated. Use balance_classes=True "
            "in dataset config for proper class balancing at batch construction time.",
            DeprecationWarning
        )
        return torch.ones(len(self.batch_mapping))

    def get_class_weights(self):
        """
        Compute class weights for weighted loss function.

        Returns:
            dict: {class_id: weight} mapping with remapped labels if noise excluded
        """
        # Support both column names: 'label_logic' or 'token label_logic'
        label_col = None
        if 'label_logic' in self.metadata.columns:
            label_col = 'label_logic'
        elif 'token label_logic' in self.metadata.columns:
            label_col = 'token label_logic'

        if label_col is None:
            return {i: 1.0 for i in range(self.num_classes)}

        # Get class counts from metadata
        labels = self.metadata[label_col].values.copy()

        # Remap labels if noise excluded (1,2,3,4 → 0,1,2,3)
        if not self.include_noise:
            labels = labels - 1

        class_counts = Counter(labels)
        total_samples = len(labels)

        # Inverse frequency weighting
        class_weights = {}
        for cls, count in class_counts.items():
            class_weights[int(cls)] = total_samples / (self.num_classes * count)

        # Ensure all classes are present
        for cls in range(self.num_classes):
            if cls not in class_weights:
                class_weights[cls] = 1.0

        return class_weights


def create_filtered_split_datasets(
        metadata_file,
        target_batch_size=200,
        dataset_key='token',
        test_val_strategy='filter',
        test_val_session_filter=None,
        test_val_participant_filter=None,
        test_val_experiment_filter=None,
        test_val_label_filter=None,
        test_val_random_experiments=None,
        test_val_multi_session=True,
        test_val_split_ratio=0.5,
        split_level='sequence',
        random_seed=353,
        global_participant_filter=None,
        global_session_filter=None,
        global_experiment_filter=None,
        global_label_filter=None,
        shuffle_experiments=True,
        shuffle_sequences=True,
        # Class balancing parameters (only applied to train set)
        balance_classes=False,
        balance_strategy='oversample',
        oversample_config=None,
        # Noise class handling
        include_noise=None,
        # Sequence grouping
        sequence_grouping=None,
        **kwargs):
    """
    Convenience function to create datasets with filtered splitting

    Args:
        test_val_strategy: 'filter' or 'random' - determines split method
        test_val_*_filter: Data matching these filters (AND logic) for test/val
        test_val_random_experiments: Number of experiments for random strategy
        test_val_multi_session: Prefer multi-session coverage in random strategy
        test_val_split_ratio: How to split filtered data (0.5 = 50% test, 50% val)
        balance_classes: Enable class balancing for training set
        balance_strategy: 'oversample' or 'undersample'
        oversample_config: Configuration for oversampling method and augmentations
        include_noise: Include noise class (None = use label_config.yaml setting)
        sequence_grouping: Sequence grouping config dict (only applied to train set)

    Returns:
        Tuple of (train_dataset, test_dataset, val_dataset)
    """

    # Filter out private kwargs that are handled explicitly for test/val datasets
    filtered_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ('_suppress_split_info', '_suppress_metadata_info')}

    common_kwargs = {
        'metadata_file': metadata_file,
        'target_batch_size': target_batch_size,
        'dataset_key': dataset_key,
        'test_val_strategy': test_val_strategy,
        'test_val_session_filter': test_val_session_filter,
        'test_val_participant_filter': test_val_participant_filter,
        'test_val_experiment_filter': test_val_experiment_filter,
        'test_val_label_filter': test_val_label_filter,
        'test_val_random_experiments': test_val_random_experiments,
        'test_val_multi_session': test_val_multi_session,
        'test_val_split_ratio': test_val_split_ratio,
        'split_level': split_level,
        'random_seed': random_seed,
        'global_participant_filter': global_participant_filter,
        'global_session_filter': global_session_filter,
        'global_experiment_filter': global_experiment_filter,
        'global_label_filter': global_label_filter,
        'shuffle_experiments': shuffle_experiments,
        'shuffle_sequences': shuffle_sequences,
        'balance_classes': balance_classes,
        'balance_strategy': balance_strategy,
        'oversample_config': oversample_config,
        'include_noise': include_noise,
        'sequence_grouping': sequence_grouping,
        **filtered_kwargs
    }

    print("Creating datasets with filtered splitting...")

    # Create train dataset first (prints all info)
    train_dataset = FilteredSplitH5Dataset(
        split_type='train',
        **common_kwargs
    )

    # Create test and val datasets with suppressed prints
    test_dataset = FilteredSplitH5Dataset(
        split_type='test',
        _suppress_split_info=True,
        _suppress_metadata_info=True,
        **common_kwargs
    )
    val_dataset = FilteredSplitH5Dataset(
        split_type='val',
        _suppress_split_info=True,
        _suppress_metadata_info=True,
        **common_kwargs
    )

    return train_dataset, test_dataset, val_dataset


if __name__ == "__main__":
    # Your specific use case: Filter session 11, split 50/50 between test/val
    # Everything else goes to training
    train_ds, test_ds, val_ds = create_filtered_split_datasets(
        metadata_file='/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_Env_TWi0005_TSt0002_SWi0010/metadata.csv',
        target_batch_size=200,
        test_val_session_filter=[14],  # Session 11 becomes test/val data
        test_val_split_ratio=0.5,  # 50% of session 11 → test, 50% → val
        split_level='sequence',  # Split at sequence level
        random_seed=42
    )

    # Verify the splits
    print("\n" + "=" * 50)
    print("VERIFICATION:")
    print("=" * 50)

    train_info = train_ds.get_split_info()
    test_info = test_ds.get_split_info()
    val_info = val_ds.get_split_info()

    print(f"Train sessions: {train_ds.metadata['session'].unique() if len(train_ds.metadata) > 0 else 'None'}")
    print(f"Test sessions: {test_ds.metadata['session'].unique() if len(test_ds.metadata) > 0 else 'None'}")
    print(f"Val sessions: {val_ds.metadata['session'].unique() if len(val_ds.metadata) > 0 else 'None'}")

    # Alternative examples:

    # Example 2: Filter specific participants for test/val
    # train_ds, test_ds, val_ds = create_filtered_split_datasets(
    #     metadata_file='path/to/metadata.csv',
    #     test_val_participant_filter=[10, 23],  # These participants → test/val
    #     test_val_split_ratio=0.3,              # 30% test, 70% val
    #     split_level='experiment'               # Split by experiments
    # )

    # Example 3: Filter specific experiments AND sessions
    # train_ds, test_ds, val_ds = create_filtered_split_datasets(
    #     metadata_file='path/to/metadata.csv',
    #     test_val_session_filter=[11, 23],
    #     test_val_experiment_filter=['exp_A', 'exp_B'],
    #     test_val_split_ratio=0.5
    # )