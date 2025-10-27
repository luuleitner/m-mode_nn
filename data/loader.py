import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Literal
from sklearn.model_selection import train_test_split


class FilteredSplitH5Dataset(Dataset):
    """
    Dataset that filters specific data for test/validation and uses remainder for training

    Logic:
    1. Apply filters to identify test/validation candidates
    2. Split filtered data between test/validation
    3. Use remaining (non-filtered) data for training
    """

    def __init__(self,
                 metadata_file: str,
                 target_batch_size: int = 200,
                 dataset_key: str = 'token',
                 data_root: Optional[str] = None,
                 # What to return
                 split_type: Literal['train', 'test', 'val'] = 'train',
                 # Filters for test/val data (train gets everything else)
                 test_val_participant_filter: Optional[List] = None,
                 test_val_session_filter: Optional[List] = None,
                 test_val_experiment_filter: Optional[List] = None,
                 test_val_label_filter: Optional[List] = None,
                 # How to split the filtered test/val data
                 test_val_split_ratio: float = 0.5,  # 0.5 = 50% test, 50% val
                 split_level: Literal['sequence', 'experiment'] = 'sequence',
                 random_seed: int = 42,
                 # Global filters (applied to ALL data before train/test/val split)
                 global_participant_filter: Optional[List] = None,
                 global_session_filter: Optional[List] = None,
                 global_experiment_filter: Optional[List] = None,
                 global_label_filter: Optional[List] = None,
                 # Shuffling
                 shuffle_experiments: bool = True,
                 shuffle_sequences: bool = True,
                 # Print control
                 _suppress_split_info: bool = False,
                 _suppress_metadata_info: bool = False):
        """
        Args:
            split_type: Which dataset to return ('train', 'test', 'val')
            test_val_*_filter: Filters to identify data for test/validation
            test_val_split_ratio: How to split filtered data (0.5 = 50% test, 50% val)
            split_level: Whether to split filtered data by 'sequence' or 'experiment'
            global_*_filter: Filters applied to ALL data before any splitting
            _suppress_split_info: If True, suppress printing of split information
        """

        self.target_batch_size = target_batch_size
        self.dataset_key = dataset_key
        self.data_root = Path(data_root) if data_root else None
        self.split_type = split_type
        self.random_seed = random_seed

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
            test_val_participant_filter, test_val_session_filter,
            test_val_experiment_filter, test_val_label_filter,
            test_val_split_ratio, split_level, _suppress_split_info, _suppress_metadata_info   # PASS THE FLAG
        )

        # Process the requested split
        self.metadata = self.splits[split_type].copy()
        self._process_metadata(shuffle_experiments, shuffle_sequences)
        self._build_batch_mapping()

        self._print_dataset_info()

    def _load_metadata(self, metadata_file: str, suppress_print: bool = False):
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
            self.metadata = self.metadata[
                self.metadata['experiment'].isin(experiment_filter)
            ]
            print(f"Global experiment filter: -> {len(self.metadata)} sequences")

        if label_filter is not None:
            self.metadata = self.metadata[
                self.metadata['label'].isin(label_filter)
            ]
            print(f"Global label filter: -> {len(self.metadata)} sequences")

    def _create_splits(self, test_val_participant_filter, test_val_session_filter,
                       test_val_experiment_filter, test_val_label_filter,
                       test_val_split_ratio, split_level, suppress_split_print=False,
                       suppress_metadata_print=False):  # ADD BOTH PARAMETERS
        """Create train/test/val splits based on filters"""

        # Start with all globally filtered data
        all_data = self.metadata.copy().reset_index(drop=True)

        # Identify test/validation candidates using filters
        test_val_candidates = all_data.copy()
        filter_applied = False

        if test_val_participant_filter is not None:
            test_val_candidates = test_val_candidates[
                test_val_candidates['participant'].isin(test_val_participant_filter)
            ]
            filter_applied = True

        if test_val_session_filter is not None:
            test_val_candidates = test_val_candidates[
                test_val_candidates['session'].isin(test_val_session_filter)
            ]
            filter_applied = True

        if test_val_experiment_filter is not None:
            test_val_candidates = test_val_candidates[
                test_val_candidates['experiment'].isin(test_val_experiment_filter)
            ]
            filter_applied = True

        if test_val_label_filter is not None:
            test_val_candidates = test_val_candidates[
                test_val_candidates['label'].isin(test_val_label_filter)
            ]
            filter_applied = True

        if not filter_applied:
            raise ValueError("No test/validation filters specified. Please specify at least one test_val_*_filter")

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

    def _split_test_val_data(self, test_val_data, test_ratio, split_level):
        """Split the filtered test/val data between test and validation"""

        if split_level == 'sequence':
            # Simple random split of sequences
            if len(test_val_data) == 1:
                # Only one sequence - put it in test
                return test_val_data, pd.DataFrame()

            test_data, val_data = train_test_split(
                test_val_data,
                test_size=test_ratio,
                random_state=self.random_seed,
                stratify=test_val_data['label'] if 'label' in test_val_data.columns else None
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

    def _process_metadata(self, shuffle_experiments: bool, shuffle_sequences: bool):
        """Process metadata and group by experiment"""

        if len(self.metadata) == 0:
            self.experiment_groups = {}
            self.experiment_list = []
            return

        # Group sequences by experiment (file path)
        self.experiment_groups = {}

        for file_path, group in self.metadata.groupby('file_path'):
            # Sort by token_id to maintain sequence order
            group_sorted = group.sort_values('token_id').reset_index(drop=True)

            if shuffle_sequences:
                group_sorted = group_sorted.sample(frac=1).reset_index(drop=True)

            self.experiment_groups[file_path] = group_sorted

        # Get list of experiments
        self.experiment_list = list(self.experiment_groups.keys())

        if shuffle_experiments:
            random.shuffle(self.experiment_list)

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

    def __len__(self):
        return len(self.batch_mapping)

    def __getitem__(self, batch_idx: int) -> torch.Tensor:
        """Load and return a batch of sequences"""

        if batch_idx >= len(self.batch_mapping):
            raise IndexError(f"Batch index {batch_idx} out of range")

        batch_info = self.batch_mapping[batch_idx]
        batch_sequences = []

        # Group batch info by file to minimize file operations
        file_groups = defaultdict(list)
        for item in batch_info:
            file_groups[item['file_path']].append(item['sequence_metadata'])

        # Load sequences from each file
        for file_path, sequences_metadata in file_groups.items():
            # Get full file path
            if self.data_root:
                full_path = self.data_root / file_path
            else:
                full_path = Path(file_path)

            # Load data from H5 file
            with h5py.File(full_path, 'r') as f:
                file_sequences = []

                for seq_meta in sequences_metadata:
                    sequence_idx = int(seq_meta['sequence_id'])

                    # Load sequence: [seq_length, channels, height, width]
                    sequence_data = f[self.dataset_key][sequence_idx]
                    file_sequences.append(sequence_data)

                # Stack sequences from this file
                if file_sequences:
                    file_batch = np.stack(file_sequences, axis=0)
                    batch_sequences.append(file_batch)

        # Concatenate all sequences in the batch
        if len(batch_sequences) == 1:
            final_batch = batch_sequences[0]
        elif len(batch_sequences) > 1:
            final_batch = np.concatenate(batch_sequences, axis=0)
        else:
            # Empty batch - shouldn't happen but handle gracefully
            return torch.empty(0)

        return torch.from_numpy(final_batch)

    def get_batch_metadata(self, batch_idx: int) -> pd.DataFrame:
        """Get metadata for all sequences in a specific batch"""

        if batch_idx >= len(self.batch_mapping):
            raise IndexError(f"Batch index {batch_idx} out of range")

        batch_info = self.batch_mapping[batch_idx]
        metadata_list = [item['sequence_metadata'] for item in batch_info]

        return pd.DataFrame(metadata_list)

    def get_split_info(self) -> Dict:
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
        print(f"\n{self.split_type.upper()} Dataset initialized:")
        print(f"  - Total sequences: {len(self.metadata)}")
        print(f"  - Unique experiments: {len(self.experiment_groups)}")
        print(f"  - Total batches: {len(self.batch_mapping)}")


def create_filtered_split_datasets(
        metadata_file: str,
        target_batch_size: int = 200,
        dataset_key: str = 'token',  # ADD THIS LINE
        # Test/Val filters - data matching these becomes test/val
        test_val_session_filter: Optional[List] = None,
        test_val_participant_filter: Optional[List] = None,
        test_val_experiment_filter: Optional[List] = None,
        test_val_label_filter: Optional[List] = None,
        # How to split the filtered data
        test_val_split_ratio: float = 0.5,  # 0.5 = 50% test, 50% val
        split_level: str = 'sequence',
        random_seed: int = 353,  # ADD THIS LINE
        # Global filters  # ADD THESE LINES
        global_participant_filter: Optional[List] = None,
        global_session_filter: Optional[List] = None,
        global_experiment_filter: Optional[List] = None,
        global_label_filter: Optional[List] = None,
        # Shuffling  # ADD THESE LINES
        shuffle_experiments: bool = True,
        shuffle_sequences: bool = True,
        **kwargs
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Convenience function to create datasets with filtered splitting

    Args:
        test_val_*_filter: Data matching these filters becomes test/validation data
        test_val_split_ratio: How to split filtered data (0.5 = 50% test, 50% val)

    Returns:
        Tuple of (train_dataset, test_dataset, val_dataset)
    """

    common_kwargs = {
        'metadata_file': metadata_file,
        'target_batch_size': target_batch_size,
        'dataset_key': dataset_key,  # ADD THIS LINE
        'test_val_session_filter': test_val_session_filter,
        'test_val_participant_filter': test_val_participant_filter,
        'test_val_experiment_filter': test_val_experiment_filter,
        'test_val_label_filter': test_val_label_filter,
        'test_val_split_ratio': test_val_split_ratio,
        'split_level': split_level,
        'random_seed': random_seed,  # ADD THIS LINE
        'global_participant_filter': global_participant_filter,  # ADD THESE LINES
        'global_session_filter': global_session_filter,
        'global_experiment_filter': global_experiment_filter,
        'global_label_filter': global_label_filter,
        'shuffle_experiments': shuffle_experiments,
        'shuffle_sequences': shuffle_sequences,
        **kwargs
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

# Example usage
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