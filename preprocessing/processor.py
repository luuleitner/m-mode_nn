import re
import numpy as np
import pandas as pd

import os
import glob
import datetime
import yaml

from tqdm import tqdm


from include.dasIT.dasIT.features.signal import envelope, logcompression, analytic_signal

from preprocessing.dimension_checker import DimensionChecker
from config.configurator import load_config, setup_environment
from preprocessing.visualization.plot_callback import plot_mmode
from utils.saving import init_dataset, append_and_save
from preprocessing.signal_utils import peak_normalization, Z_normalization, butter_bandpass_filter, butter_lowpass_filter, Time_Gain_Compensation, extract_sliding_windows, apply_joystick_filters
from preprocessing.label_logic.label_logic import (
    create_position_peak_labels,
    create_5class_position_peak_labels
)
from preprocessing.soft_labels import SoftLabelGenerator, window_hard_labels

import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")


class DataProcessor():
    def __init__(self, config_file='config.yaml', auto_run=True):
        # Setup Config Parameters
        self._config_path = config_file  # Store for dimension checker
        self._config = load_config(config_file)
        self._np_seed_generator = setup_environment(self._config)
        self._auto_run = auto_run

        # Set Operation Mode
        self._operation_mode = self._config.global_setting.run.mode
        self._debug_level = self._config.global_setting.run.config.debug.level if self._operation_mode == 'debug' else None

        # Set Processing Parameters
        # Signal Processing and Flags

        # ---Clipping
        self._clip_flag = self._config.preprocess.signal.clip.apply
        self._clip_initial_size = self._config.preprocess.signal.clip.initial_size
        self._clip_samples2remove_start = self._config.preprocess.signal.clip.samples2remove_start
        self._clip_samples2remove_end = self._config.preprocess.signal.clip.samples2remove_end

        #---Bandpass Filtering
        self._bandpass_flag = self._config.preprocess.signal.bandpass.apply
        self._bandpass_lowcut = self._config.preprocess.signal.bandpass.lowcut
        self._bandpass_highcut = self._config.preprocess.signal.bandpass.highcut
        self._bandpass_fs = self._config.preprocess.signal.bandpass.fs
        self._bandpass_order = self._config.preprocess.signal.bandpass.order
        
        #---Time Gain Compensation (TGC)
        self._tgc_flag = self._config.preprocess.signal.tgc.apply
        self._tgc_fs = self._config.preprocess.signal.tgc.freq
        self._tgc_coef_att = self._config.preprocess.signal.tgc.coef_att

        #---Envelope
        self._envelope_flag = self._config.preprocess.signal.envelope.apply
        self._envelope_interp = self._config.preprocess.signal.envelope.interp
        self._envelope_padding_flag = self._config.preprocess.signal.envelope.padding.apply
        self._envelope_padding_mode = self._config.preprocess.signal.envelope.padding.mode
        self._envelope_padding_amount = self._config.preprocess.signal.envelope.padding.amount

        #---Envelope Lowpass (anti-aliasing before decimation)
        envelope_lp = getattr(self._config.preprocess.signal.envelope, 'lowpass', None)
        if envelope_lp is not None:
            self._envelope_lp_flag = getattr(envelope_lp, 'apply', False)
            self._envelope_lp_mode = getattr(envelope_lp, 'mode', 'auto')
            self._envelope_lp_manual_cutoff = getattr(envelope_lp, 'manual_cutoff', 2e6)
            self._envelope_lp_order = getattr(envelope_lp, 'order', 4)
        else:
            self._envelope_lp_flag = False
            self._envelope_lp_mode = 'auto'
            self._envelope_lp_manual_cutoff = 2e6
            self._envelope_lp_order = 4

        #---Decimation
        self._decimation_flag = self._config.preprocess.signal.decimation.apply
        self._decimation_factor = self._config.preprocess.signal.decimation.factor

        #---Log Compression
        self._logcompression_flag = self._config.preprocess.signal.logcompression.apply
        self._logcompression_dbrange = self._config.preprocess.signal.logcompression.db

        #---Normalization
        # In the standard mode the normalization is done channel wise for each exerpiment
        self._normalization_flag = self._config.preprocess.signal.normalization.apply
        self._normalization_technique = self._config.preprocess.signal.normalization.method

        #---Differentiation (temporal gradient along pulse axis)
        diff_cfg = getattr(self._config.preprocess.signal, 'differentiation', None)
        if diff_cfg is not None:
            self._differentiation_flag = getattr(diff_cfg, 'apply', False)
            self._differentiation_method = getattr(diff_cfg, 'method', 'gradient')
            self._differentiation_order = getattr(diff_cfg, 'order', 1)
        else:
            self._differentiation_flag = False
            self._differentiation_method = 'gradient'
            self._differentiation_order = 1

        #---Percentile Clipping (clip outliers to Nth percentile)
        pclip_cfg = getattr(self._config.preprocess.signal, 'percentile_clip', None)
        if pclip_cfg is not None:
            self._percentile_clip_flag = getattr(pclip_cfg, 'apply', False)
            self._percentile_clip_value = getattr(pclip_cfg, 'percentile', 99)
            self._percentile_clip_symmetric = getattr(pclip_cfg, 'symmetric', True)
        else:
            self._percentile_clip_flag = False
            self._percentile_clip_value = 99
            self._percentile_clip_symmetric = True

        # Tokens
        self._token_window = self._config.preprocess.tokenization.window
        self._token_stride = self._config.preprocess.tokenization.stride

        # Sequences
        self._sequence_window = self._config.preprocess.sequencing.window

        # Output mode: transformer (sequenced) or flat (CNN-ready)
        self._output_mode = getattr(self._config.preprocess.output, 'mode', 'transformer')

        # Label configuration - load from separate label_config.yaml
        self._label_config = self._load_label_config()
        self._label_method = self._label_config.get('method', 'position_peak')
        self._label_axis = self._label_config.get('axis', 'dual')  # x | y | dual
        self._joystick_filters = self._label_config.get('filters', {})

        # Position peak parameters
        position_peak_config = self._label_config.get('position_peak', {})
        self._pp_deriv_thresh = position_peak_config.get('deriv_threshold_percent', 10.0)
        self._pp_pos_thresh = position_peak_config.get('pos_threshold_percent', 5.0)
        self._pp_peak_window = position_peak_config.get('peak_window', 3)
        self._pp_timeout = position_peak_config.get('timeout_samples', 500)

        # Class configuration (centralized)
        classes_config = self._label_config.get('classes', {})
        self._include_noise = classes_config.get('include_noise', True)
        self._noise_class = classes_config.get('noise_class', 0)
        self._class_names = classes_config.get('names', {})

        # Derive num_label_classes based on axis mode:
        # - dual axis: 5 classes (Noise, Up, Down, Left, Right)
        # - single axis (x or y): 3 classes (Noise, Positive, Negative)
        if self._label_axis == 'dual':
            self._num_label_classes = 5
        else:
            self._num_label_classes = 3

        # Soft labels configuration
        soft_labels_config = self._label_config.get('soft_labels', {})
        self._soft_labels_enabled = soft_labels_config.get('enabled', False)

        if self._soft_labels_enabled:
            self._soft_label_gen = SoftLabelGenerator(
                num_classes=self._num_label_classes,
                weighting=soft_labels_config.get('weighting', 'gaussian'),
                gaussian_sigma_ratio=soft_labels_config.get('gaussian_sigma_ratio', 0.25)
            )
        else:
            self._soft_label_gen = None

        # Saving
        self._save_strategy = self._config.preprocess.data.save_ftype
        self._save_path_id = self._config.preprocess.data.id
        self._h5file = None

        # Setup metadata table and token ID tracker
        self._metastructure = ['token_id local', 'start', 'end', 'participant', 'session', 'experiment', 'token label_logic']
        self._metadata = pd.DataFrame(columns=self._metastructure)
        self._tokenid = np.zeros(1)

        # Define paths to raw and processed data
        self._data_path_processed = os.path.join(self._config.preprocess.data.basepath, 'processed')
        self._data_path_raw = os.path.join(self._config.preprocess.data.basepath, 'raw')
        if self._data_path_raw is None:
            raise ValueError("Data path not found in the config file under 'experiment: path'")

        if self._output_mode == 'transformer':
            self._file_save_id = f'TokenWin{int(self._config.preprocess.tokenization.window):02}_TokenStr{int(self._config.preprocess.tokenization.stride):02}_SeqWin{int(self._config.preprocess.sequencing.window):02}'
        else:  # flat mode
            soft_suffix = '_soft' if self._soft_labels_enabled else ''
            self._file_save_id = f'Window{int(self._config.preprocess.tokenization.window):02}_Stride{int(self._config.preprocess.tokenization.stride):02}_Labels{soft_suffix}'

        # Other Flags
        self._save_flag = self._config.preprocess.tokenization.tokens2file
        self._set_token_startendID_flag = self._config.preprocess.tokenization.startendID

        # Only run full pipeline if auto_run=True
        if not self._auto_run:
            return

        # Create unique output folder structure: dataset/params/run_timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_folder = os.path.join(self._data_path_processed, self._save_path_id)
        params_folder = os.path.join(dataset_folder, self._file_save_id)
        self._output_folder = os.path.join(params_folder, f"run_{timestamp}")

        # Create all directories
        os.makedirs(self._output_folder, exist_ok=True)
        logger.info(f"Created unique output folder: {self._output_folder}")

        # Create/update 'latest' symlink pointing to this run
        latest_link = os.path.join(params_folder, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(f"run_{timestamp}", latest_link)
        logger.info(f"Updated 'latest' symlink -> run_{timestamp}")

        # ---------------------------------------------
        # --------------PREPROCESS---------------------
        # ---------------------------------------------
        # (1) PREPROCESS: Load folder structure depending on selected strategy
        if self._config.preprocess.data.strategy == "all":
            self._load_fstructure()
        elif self._config.preprocess.data.strategy == "selection_file":
            selection_path = self._config.preprocess.data.selection_file
            self._load_fstructure_from_selection(selection_path)
        else:
            raise ValueError(f"Unknown strategy: {self._config.preprocess.data.strategy}. Use 'all' or 'selection_file'")
        # ---------------------------------------------
        # (2) PREPROCESS: Process each experiment found in the file structure

        # Log start of processing and save replication info
        self._log_processing_stage('start_processing', {
            'total_experiments': len(self._fstructure),
            'strategy': self._config.preprocess.data.strategy,
            'operation_mode': self._operation_mode,
            'debug_level': self._debug_level,
            'data_dimensions': {
                'token_window': self._token_window,
                'token_stride': self._token_stride,
                'sequence_window': self._sequence_window,
                'expected_token_shape': f'[num_tokens, channels, height, {self._token_window}]',
                'expected_sequence_shape': f'[num_sequences, {self._sequence_window}, channels, height, {self._token_window}]'
            }
        })
        self._save_replication_info()

        self._process_experiments()

        # ----------- >>>>> STOP


    def _log_processing_stage(self, stage, details):
        """Log processing stages to global and local YAML logs"""

        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'run_id': os.path.basename(self._output_folder),
            'stage': stage,
            'details': details
        }
        
        # Global processing log (append mode) 
        global_log_path = os.path.join(self._data_path_processed, 'processing_log.yaml')
        if os.path.exists(global_log_path):
            with open(global_log_path, 'r') as f:
                global_log = yaml.safe_load(f) or []
        else:
            global_log = []
        
        global_log.append(log_entry)
        with open(global_log_path, 'w') as f:
            yaml.dump(global_log, f, default_flow_style=False)
        
        # Local run log
        local_log_path = os.path.join(self._output_folder, 'processing_log.yaml')
        if os.path.exists(local_log_path):
            with open(local_log_path, 'r') as f:
                local_log = yaml.safe_load(f) or []
        else:
            local_log = []
        
        local_log.append(log_entry)
        with open(local_log_path, 'w') as f:
            yaml.dump(local_log, f, default_flow_style=False)

    def _load_label_config(self):
        """Load label configuration from label_logic/label_config.yaml"""
        label_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'label_logic', 'label_config.yaml'
        )
        with open(label_config_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_replication_info(self):
        """Save all info needed to replicate this processing run"""
        import shutil
        
        # Calculate expected dimensions
        height_after_clip = (self._clip_initial_size - self._clip_samples2remove_start - self._clip_samples2remove_end) if self._clip_flag else 'full'
        height_after_decimation = (self._clip_initial_size - self._clip_samples2remove_start - self._clip_samples2remove_end) // self._decimation_factor if self._clip_flag and self._decimation_flag else 'varies'
        
        replication_info = {
            'run_info': {
                'run_id': os.path.basename(self._output_folder),
                'timestamp': datetime.datetime.now().isoformat(),
                'output_folder': self._output_folder
            },
            'data_dimensions': {
                'input': {
                    'raw_signal': '[channels, time_samples, a_mode_lines]',
                    'expected_channels': '3 (ultrasound channels)'
                },
                'after_processing': {
                    'after_clipping': f'[channels, {height_after_clip}, a_mode_lines]' if self._clip_flag else 'unchanged',
                    'after_decimation': f'[channels, {height_after_decimation}, a_mode_lines]' if self._decimation_flag else 'unchanged',
                    'with_startend_markers': '+2 samples in height dimension' if self._set_token_startendID_flag else 'no markers'
                },
                'tokenization': {
                    'token_shape': f'[num_tokens, channels, height, {self._token_window}]',
                    'token_window': self._token_window,
                    'token_stride': self._token_stride,
                    'overlap': f'{self._token_window - self._token_stride} samples' if self._token_stride < self._token_window else 'no overlap'
                },
                'sequencing': {
                    'sequence_shape': f'[num_sequences, {self._sequence_window}, channels, height, {self._token_window}]',
                    'tokens_per_sequence': self._sequence_window
                },
                'output_format': {
                    'data_format': self._save_strategy,
                    'data_key': 'X' if self._save_strategy == 'h5' else 'data',
                    'label_key': 'y' if self._save_strategy == 'h5' else 'label_logic'
                }
            },
            'processing_parameters': {
                'bandpass_flag': self._bandpass_flag,
                'bandpass_lowcut': self._bandpass_lowcut,
                'bandpass_highcut': self._bandpass_highcut,
                'bandpass_fs': self._bandpass_fs,
                'bandpass_order': self._bandpass_order,
                'tgc_flag': self._tgc_flag,
                'tgc_fs': self._tgc_fs,
                'tgc_coef_att': self._tgc_coef_att,
                'clip_flag': self._clip_flag,
                'clip_initial_size': self._clip_initial_size,
                'clip_samples2remove_start': self._clip_samples2remove_start,
                'clip_samples2remove_end': self._clip_samples2remove_end,
                'decimation_flag': self._decimation_flag,
                'decimation_factor': self._decimation_factor,
                'envelope_flag': self._envelope_flag,
                'envelope_padding_flag': self._envelope_padding_flag,
                'envelope_interp': self._envelope_interp,
                'envelope_padding_mode': self._envelope_padding_mode,
                'envelope_padding_amount': self._envelope_padding_amount,
                'logcompression_flag': self._logcompression_flag,
                'logcompression_dbrange': self._logcompression_dbrange,
                'normalization_flag': self._normalization_flag,
                'normalization_technique': self._normalization_technique,
                'differentiation_flag': self._differentiation_flag,
                'differentiation_method': self._differentiation_method,
                'differentiation_order': self._differentiation_order,
                'percentile_clip_flag': self._percentile_clip_flag,
                'percentile_clip_value': self._percentile_clip_value,
                'percentile_clip_symmetric': self._percentile_clip_symmetric,
                'token_window': self._token_window,
                'token_stride': self._token_stride,
                'sequence_window': self._sequence_window,
                'save_strategy': self._save_strategy,
                'output_mode': self._output_mode,
                'label_method': self._label_method,
                'label_axis': self._label_axis,
                'position_peak': {
                    'deriv_threshold_percent': self._pp_deriv_thresh,
                    'pos_threshold_percent': self._pp_pos_thresh,
                    'peak_window': self._pp_peak_window,
                    'timeout_samples': self._pp_timeout
                },
                'soft_labels_enabled': self._soft_labels_enabled,
                'num_label_classes': self._num_label_classes
            },
            'input_data': {
                'data_path_raw': self._data_path_raw,
                'strategy': self._config.preprocess.data.strategy,
                'experiments_found': len(self._fstructure),
                'experiment_paths': self._fstructure
            }
        }
        

        replication_path = os.path.join(self._output_folder, 'replication_info.yaml')
        with open(replication_path, 'w') as f:
            yaml.dump(replication_info, f, default_flow_style=False, indent=2)
        
        logger.info(f"Replication info saved: {replication_path}")


    def _load_fstructure(self):
        """Load all experiment folders from raw data directory.

        Auto-detects directory hierarchy:
          - sgambato format: P*/session*/exp*
          - leitner format: session*_W_*/<numbered_dirs>
        """
        # Try sgambato format first: P*/session*/exp*
        all_paths = glob.glob(os.path.join(self._data_path_raw, "P*", "session*", "exp*"))

        if not all_paths:
            # Fallback: leitner format — session*/<numbered_dirs>
            all_paths = glob.glob(os.path.join(self._data_path_raw, "session*", "*"))
            all_paths = [p for p in all_paths
                         if os.path.isdir(p) and os.path.basename(p).isdigit()]
            if all_paths:
                logger.info("Auto-detected leitner directory hierarchy (session*/N)")

        self._fstructure = sorted([
            f for f in all_paths
            if os.path.isdir(f)
        ])
        logger.info(f"Found {len(self._fstructure)} experiments in {self._data_path_raw}")

    def _load_fstructure_from_selection(self, selection_path):
        """
        Load experiment selection from a CSV or YAML selection file.

        CSV format (recommended - easy to edit in spreadsheet):
            participant,session,experiment,include[,path]
            0,0,0,1,/path/to/P000/session000/exp000
            0,0,1,0    # excluded (include=0)

        YAML format: Not supported for new hierarchy (use CSV instead)
        """
        import csv as csv_module

        # Resolve path
        if not os.path.isabs(selection_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            selection_path = os.path.join(project_root, selection_path)

        if not os.path.exists(selection_path):
            raise FileNotFoundError(f"Selection file not found: {selection_path}")

        # Detect file type and load
        if selection_path.endswith('.csv'):
            self._fstructure = self._load_selection_csv(selection_path, csv_module)
        else:
            self._fstructure = self._load_selection_yaml(selection_path)

    def _load_selection_csv(self, csv_path, csv_module):
        """Load selection from CSV file.

        CSV format: participant,session,experiment,include[,path]
        """
        selected_experiments = []

        with open(csv_path, 'r', newline='') as f:
            reader = csv_module.DictReader(f)

            for row in reader:
                include = row.get('include', '1')

                # Handle various include formats: 1, 0, true, false, yes, no
                if str(include).lower() not in ('1', 'true', 'yes', ''):
                    continue

                # Use path column if available, otherwise construct from components
                if 'path' in row and row['path']:
                    exp_path = row['path']
                else:
                    participant = int(row['participant'])
                    session = int(row['session'])
                    experiment = int(row['experiment'])
                    exp_path = os.path.join(
                        self._data_path_raw,
                        f"P{participant:03d}",
                        f"session{session:03d}",
                        f"exp{experiment:03d}"
                    )

                if os.path.isdir(exp_path):
                    selected_experiments.append(exp_path)
                else:
                    logger.warning(f"Experiment path not found: {exp_path}")

        selected_experiments = sorted(selected_experiments)
        logger.info(f"CSV selection loaded: {len(selected_experiments)} experiments")
        return selected_experiments

    def _load_selection_yaml(self, yaml_path):
        """Load selection from YAML file - not supported for new hierarchy."""
        raise NotImplementedError(
            "YAML selection is not supported for the new P/session/exp hierarchy. "
            "Please use CSV format instead. Generate with: python utils/generate_selection.py"
        )


    def _process_experiments(self):
        if self._operation_mode == 'debug' and self._debug_level >= 1:
            n_to_select = 3
            n_to_select = min(n_to_select, len(self._fstructure))
            indices = self._np_seed_generator.choice(len(self._fstructure), size=n_to_select, replace=False)
            self._plot_flag = [1 if i in indices else 0 for i in range(len(self._fstructure))]
        else:
            self._plot_flag = [0 for _ in range(len(self._fstructure))]

        filepath_meta = []
        sequence_id = []

        for i, e in enumerate(tqdm(self._fstructure, desc="Processing experiments")):

            # Apply the processing
            plot_flag = self._plot_flag[i]
            experiment_data, experiment_label, experiment_token2sequence_id = self._process_experiment(e, plot_flag)
            sequence_id.append(experiment_token2sequence_id)

            # Save processed data
            if self._save_flag:
                self.__save_data(data=experiment_data, label=experiment_label)
                #file_path_id = np.repeat(self.__clean_datapath(self._experiment_file_id), self._token_len)
                file_path_id = np.repeat(self._experiment_file_id, self._token_len)
                filepath_meta.append(file_path_id)
                
                # Log experiment data dimensions
                if i == 0:  # Log dimensions for first experiment as example
                    self._log_processing_stage('first_experiment_dimensions', {
                        'experiment_path': e,
                        'sequence_shape': list(experiment_data.shape),
                        'sequence_label_shape': list(experiment_label.shape),
                        'num_sequences': experiment_data.shape[0],
                        'sequences_per_window': experiment_data.shape[1] if len(experiment_data.shape) > 1 else None,
                        'channels': experiment_data.shape[2] if len(experiment_data.shape) > 2 else None,
                        'height_after_processing': experiment_data.shape[3] if len(experiment_data.shape) > 3 else None,
                        'width_per_token': experiment_data.shape[4] if len(experiment_data.shape) > 4 else None
                    })

                if self._save_strategy == 'h5':
                    if hasattr(self, '_h5file') and self._h5file is not None:
                        self._h5file.close()
                        self._h5file = None
                elif self._save_strategy == 'zarr':
                    pass
                logger.info(f'Experiment {i + 1} from {len(self._fstructure)} experiments processed and saved successfully')
        logger.info(f'{len(self._fstructure)} experiments processes successfully.')


        # Save metadata
        meta = pd.DataFrame(np.stack((np.hstack(sequence_id), np.hstack(filepath_meta)), axis=1), columns=['sequence id', 'file path'])
        self._metadata = pd.concat([self._metadata, meta], axis=1)
        self._metadata.to_csv(os.path.join(self._output_folder, 'metadata.csv'), index=False)
        logger.info('Metadata saved successfully.')
        
        # Log completion of processing
        self._log_processing_stage('complete_processing', {
            'experiments_processed': len(self._fstructure),
            'sequences_created': len(np.hstack(sequence_id)),
            'output_folder': self._output_folder,
            'metadata_file': 'metadata.csv',
            'total_tokens': len(self._metadata),
            'data_storage_format': self._save_strategy,
            'file_naming_pattern': self._file_save_id
        })
        
        # Update dimensions guide with actual processed data
        try:
            dimension_checker = DimensionChecker(config_path=self._config_path if hasattr(self, '_config_path') else 'config/config.yaml')
            dimension_checker.update_guide(processed_data_path=self._output_folder)
            logger.info('Dimensions guide updated with processed data information')
        except Exception as e:
            logger.warning(f'Could not update dimensions guide: {e}')



    ### PROCESSING CORE
    ##############################################################################################
    ##############################################################################################
    def _process_experiment(self, exp, plot_flag):
        try:
            # Find ultrasound and joystick files
            files = glob.glob(os.path.join(exp, "_US_ch*"))
            nbr_us_channels = len(files)
            files += glob.glob(os.path.join(exp, "_joystick*"))

            # Load and decimate data if required
            data = []
            for f in files:
                d = np.load(f, allow_pickle=True)
                data.append(d.T)
                if self._operation_mode == 'debug' and self._debug_level >= 2:
                    plot_mmode(d.T, 60)
        except:
            logger.error(f"Cannot process experiment {exp}")

        # Stack ultrasound and joystick data separately
        data_X = np.stack(data[:nbr_us_channels], axis=0)
        data_y = np.stack(data[nbr_us_channels:], axis=0)

        # Parse participant, session, and experiment info from folder path
        # Expected format: .../P000/session000/exp000
        session_path = os.path.normpath(exp)

        # Try sgambato format: .../P000/session000/exp000
        match = re.search(r"P(\d+)/session(\d+)/exp(\d+)", session_path)
        if match:
            self._participant_id = str(int(match.group(1)))
            self._session_id = str(int(match.group(2)))
            self._experiment_id = str(int(match.group(3)))
        else:
            # Try leitner format: .../session14_W_001/5
            match = re.search(r"session(\d+)_\w+_(\d+)[/\\](\d+)$", session_path)
            if match:
                self._session_id = str(int(match.group(1)))
                self._participant_id = str(int(match.group(2)))
                self._experiment_id = str(int(match.group(3)))
            else:
                logger.warning(f"Could not parse experiment info from path: {session_path}")
                self._participant_id = None
                self._session_id = None
                self._experiment_id = None

        # ---------------------------------------------
        # Run Processing on Experimental Data
        # Log raw data dimensions
        raw_shape = data_X.shape
        
        # Signal Processing
        data_X = self._signal_processing(data_X)
        processed_shape = data_X.shape

        # Tokenization Pipeline
        token, token_label = self._tokenizer(data_X, data_y, plot_flag)

        # Sequencer
        sequence, sequence_label, token2sequence_id = self._sequencer(token, token_label)
        # ---------------------------------------------
        
        # Log dimension transformation for this experiment
        logger.debug(f"Experiment {self._experiment_id}: Raw {raw_shape} -> Processed {processed_shape} -> Tokens {token.shape} -> Sequences {sequence.shape}")

        return sequence, sequence_label, token2sequence_id




    def _tokenizer(self, data, joystick_data, plot_flag):
        # Determine number of sliding windows
        last_axis_len = data.shape[-1]
        num_windows = (last_axis_len - self._token_window) // self._token_stride + 1

        # Calculate window start and end indices - for metadata usage only
        starts = [i * self._token_stride for i in range(num_windows)]
        ends = [start + self._token_window for start in starts]

        # Extract sliding windows (tokens) from ultrasound data
        token = extract_sliding_windows(data, ax=2, window_size=self._token_window, stride=self._token_stride)
        token = token.swapaxes(0, 2).swapaxes(1, 2)  # (num_windows, chs, samples, window_size)

        # Create labels using the labeling pipeline
        token_label = self._create_token_labels(joystick_data, num_windows)

        # Add synthetic start/end markers to each token
        if self._set_token_startendID_flag:
            token = self.__add_startend_seq(token, self._minmax_normalization_values)

        # Clip excess tokens that don't fit into sequencer (for transformer mode only)
        if self._output_mode == 'transformer':
            clipfloor = int(np.floor(token.shape[0] / self._sequence_window) * self._sequence_window)
            token = token[:clipfloor]
            token_label = token_label[:clipfloor]
            starts = starts[:clipfloor]
            ends = ends[:clipfloor]

        # Create meta datafile
        self._token_len = token.shape[0]
        self.__create_meta(starts, ends, token_label)

        return token, token_label

    def _create_token_labels(self, joystick_data, num_tokens):
        """
        Create token labels using the labeling pipeline from labeling.py.

        Args:
            joystick_data: Joystick data array - either [4, pulses] or [1, 4, pulses] after stacking
            num_tokens: Expected number of tokens

        Returns:
            token_label: [num_tokens, num_classes] for soft labels or [num_tokens, 1] for hard labels
        """
        # Handle different joystick data shapes
        # After loading: joystick is [pulses, 4], transposed to [4, pulses]
        # After stacking (if single file): could be [1, 4, pulses]
        if joystick_data.ndim == 3:
            joystick_data = joystick_data[0]  # Remove batch dimension: [4, pulses]

        # Extract position data
        # Channels: 0=unused, 1=X position, 2=Y position, 3=trigger
        x_position = joystick_data[1, :]
        y_position = joystick_data[2, :]

        # Dual-axis mode: 5-class labels using amplitude voting
        if self._label_axis == 'dual':
            # Apply filters to both axes
            x_filtered = apply_joystick_filters(
                x_position.copy(), self._joystick_filters, 'position'
            )
            y_filtered = apply_joystick_filters(
                y_position.copy(), self._joystick_filters, 'position'
            )

            # Compute derivatives
            x_derivative = np.gradient(x_filtered)
            y_derivative = np.gradient(y_filtered)

            # Apply filters to derivatives
            x_derivative = apply_joystick_filters(
                x_derivative, self._joystick_filters, 'derivative'
            )
            y_derivative = apply_joystick_filters(
                y_derivative, self._joystick_filters, 'derivative'
            )

            # Create 5-class labels using position_peak + amplitude voting
            hard_labels, _, _ = create_5class_position_peak_labels(
                x_filtered, y_filtered, x_derivative, y_derivative,
                self._pp_deriv_thresh, self._pp_pos_thresh,
                self._pp_peak_window, self._pp_timeout
            )
        else:
            # Single-axis mode: 3-class labels (original behavior)
            if self._label_axis == 'x':
                raw_position = x_position
            elif self._label_axis == 'y':
                raw_position = y_position
            else:
                raw_position = x_position

            # Apply filters to position data (same as label_logic/visualize.py)
            position_data = apply_joystick_filters(
                raw_position.copy(), self._joystick_filters, 'position'
            )

            # Compute derivative
            derivative = np.gradient(position_data)

            # Apply filters to derivative
            derivative = apply_joystick_filters(
                derivative, self._joystick_filters, 'derivative'
            )

            # Create per-sample hard labels using position_peak method
            hard_labels, _, _ = create_position_peak_labels(
                position_data, derivative,
                self._pp_deriv_thresh, self._pp_pos_thresh,
                self._pp_peak_window, self._pp_timeout
            )

        # Validate label range
        label_min, label_max = hard_labels.min(), hard_labels.max()
        if label_min < 0 or label_max >= self._num_label_classes:
            raise ValueError(
                f"Invalid label range [{label_min}, {label_max}], "
                f"expected [0, {self._num_label_classes - 1}]"
            )

        # Convert to token-level labels
        if self._soft_labels_enabled:
            # Soft labels: probability distributions per token
            token_label = self._soft_label_gen.create_soft_labels(
                hard_labels, self._token_window, self._token_stride
            )
        else:
            # Hard labels: majority vote per token window
            token_labels_1d = window_hard_labels(
                hard_labels, self._token_window, self._token_stride, method='majority'
            )
            token_label = np.expand_dims(token_labels_1d, axis=1)

        # Ensure label count matches expected token count (joystick/US sample counts may differ slightly)
        if len(token_label) != num_tokens:
            logger.warning(f"Label count ({len(token_label)}) != token count ({num_tokens}), truncating/padding")
            if len(token_label) > num_tokens:
                token_label = token_label[:num_tokens]
            else:
                # Pad with zeros (noise class)
                pad_shape = (num_tokens - len(token_label), *token_label.shape[1:])
                padding = np.zeros(pad_shape, dtype=token_label.dtype)
                token_label = np.concatenate([token_label, padding], axis=0)

        return token_label


    def _sequencer(self, data, label):
        """
        Route to appropriate output format based on output_mode config.

        Transformer mode: Groups tokens into sequences
        Flat mode: Returns tokens directly (for CNN training)
        """
        if self._output_mode == 'transformer':
            return self._sequence_for_transformer(data, label)
        else:  # flat mode
            return self._output_flat(data, label)

    def _sequence_for_transformer(self, data, label):
        """Group tokens into sequences for transformer training."""
        num_sequences = int(data.shape[0] // self._sequence_window)
        sequence = data.reshape(num_sequences, self._sequence_window, data.shape[1], data.shape[2], data.shape[3])
        sequence_label = label.reshape(num_sequences, self._sequence_window, label.shape[-1])

        # Prepare sequence metadata for experimental accumulation
        sequence_ids = np.arange(num_sequences).astype(int)
        token2sequence_id = np.repeat(sequence_ids, self._sequence_window).astype(int)

        return sequence, sequence_label, token2sequence_id

    def _output_flat(self, data, label):
        """Direct token output for CNN training - no sequencing."""
        # Token IDs are just sequential indices
        token_ids = np.arange(data.shape[0]).astype(int)

        # Data and labels remain as-is (no reshaping into sequences)
        # data shape: [num_tokens, C, H, W]
        # label shape: [num_tokens, num_classes] for soft or [num_tokens, 1] for hard

        return data, label, token_ids



    def __add_startend_seq(self, token, minmax):
        # Add fixed values at start and end of every A-mode line in token
        stv = np.round(np.max(np.abs(minmax)), -1)
        spv = stv + 1
        start_val = np.full((token.shape[0], token.shape[1], 1, token.shape[3]), stv)
        end_val = np.full((token.shape[0], token.shape[1], 1, token.shape[3]), spv)
        token = np.concatenate([start_val, token, end_val], axis=2)
        return token


    def __create_meta(self, starts, ends, token_label):
        start_id = 0 if self._tokenid[-1] == 0 else self._tokenid[-1] + 1
        tid = start_id + np.arange(self._token_len)

        # Create per-token metadata entries
        participant_id = np.repeat(self._participant_id, self._token_len)
        session_id = np.repeat(self._session_id, self._token_len)
        experiment_id = np.repeat(self._experiment_id, self._token_len)

        # Handle soft labels (multi-column) vs hard labels (single column)
        if self._soft_labels_enabled:
            # For soft labels, store the argmax (dominant class) in metadata
            label_for_meta = np.argmax(token_label, axis=1)
        else:
            label_for_meta = np.squeeze(token_label)

        meta = pd.DataFrame(np.stack([tid, starts, ends, participant_id, session_id, experiment_id, label_for_meta], axis=1),
                            columns=self._metastructure)
        self._metadata = pd.concat([self._metadata, meta], axis=0, ignore_index=True)


    def __save_data(self, data=None, label=None):
        filename = f"S{int(self._session_id):04}_P{int(self._participant_id):04}_E{int(self._experiment_id):04}_Xy.h5"
        self._experiment_file_id = os.path.join(f'P{int(self._participant_id):04}', filename)
        experiment_path = os.path.join(self._output_folder, self._experiment_file_id)

        if not os.path.exists(os.path.dirname(experiment_path)):
            os.makedirs(os.path.dirname(experiment_path))

        backend = self._save_strategy

        # Determine label dtype based on soft/hard labels
        label_dtype = 'float32' if self._soft_labels_enabled else 'int64'

        file_attr = f"_{backend}file"
        if not hasattr(self, file_attr) or getattr(self, file_attr) is None:
            setattr(self, file_attr, init_dataset(
                path=experiment_path,
                data_size=data.shape[1:],
                backend=backend,
                label_shape=label.shape[1:],
                label_dtype=label_dtype
            ))
        append_and_save(experiment_path, getattr(self, file_attr), data, label, config=self._config, backend=backend)

    def __clean_datapath(self, path):
        normalized_path = os.path.normpath(self._experiment_file_id)
        path_parts = normalized_path.split(os.sep, 1)
        new_path = path_parts[1] if len(path_parts) > 1 else ""
        return new_path

    # =========================================================================
    # PUBLIC API FOR SINGLE EXPERIMENT PROCESSING (used by visualization)
    # =========================================================================

    def get_experiment_paths(self):
        """Load and return experiment paths based on config strategy."""
        if self._config.preprocess.data.strategy == "all":
            self._load_fstructure()
        elif self._config.preprocess.data.strategy == "selection_file":
            selection_path = self._config.preprocess.data.selection_file
            self._load_fstructure_from_selection(selection_path)
        return self._fstructure

    def process_single_experiment(self, exp_path):
        """
        Process a single experiment and return raw + processed data with labels.

        Args:
            exp_path: Path to experiment folder

        Returns:
            dict with keys:
                'raw_us': Raw ultrasound data [C, samples, pulses]
                'processed_us': Processed ultrasound data [C, samples, pulses]
                'joystick': Joystick data [4, pulses] (ch0=unused, ch1=X, ch2=Y, ch3=trigger)
                'labels': Per-sample labels array
                'markers': Dict with edge/peak indices for visualization
                'config_info': Dict with processing parameters used
        """
        # Load raw data
        files = sorted(glob.glob(os.path.join(exp_path, "_US_ch*")))
        if not files:
            raise FileNotFoundError(f"No US channel files found in {exp_path}")

        raw_us = np.stack([np.load(f).T for f in files], axis=0)

        joystick_files = glob.glob(os.path.join(exp_path, "_joystick*"))
        if not joystick_files:
            raise FileNotFoundError(f"No joystick file found in {exp_path}")
        joystick = np.load(joystick_files[0]).T

        # Process ultrasound
        processed_us = self._signal_processing(raw_us.copy())

        # Create labels with markers
        labels, markers = self._create_visualization_labels(joystick)

        # Config info for plot titles
        config_info = {
            'bandpass': self._bandpass_flag,
            'tgc': self._tgc_flag,
            'clip': self._clip_flag,
            'envelope': self._envelope_flag,
            'logcompression': self._logcompression_flag,
            'normalization': self._normalization_technique if self._normalization_flag else None,
            'differentiation': f"{self._differentiation_method}(order={self._differentiation_order})" if self._differentiation_flag else None,
            'percentile_clip': f"p{self._percentile_clip_value}" if self._percentile_clip_flag else None,
            'decimation_factor': self._decimation_factor if self._decimation_flag else None,
            'label_method': self._label_method,
            'label_axis': self._label_axis,
            'pp_deriv_thresh': self._pp_deriv_thresh,
            'pp_pos_thresh': self._pp_pos_thresh,
        }

        return {
            'raw_us': raw_us,
            'processed_us': processed_us,
            'joystick': joystick,
            'labels': labels,
            'markers': markers,
            'config_info': config_info
        }

    def _create_visualization_labels(self, joystick_data):
        """
        Create per-sample labels with visualization markers.

        Returns:
            labels: Per-sample label array
                - Single axis: 0=noise, 1=positive, 2=negative
                - Dual axis: 0=noise, 1=up, 2=down, 3=left, 4=right
            markers: Dict with indices for visualization
        """
        # Handle joystick shape
        if joystick_data.ndim == 3:
            joystick_data = joystick_data[0]

        # Extract position based on config axis
        x_position = joystick_data[1, :]
        y_position = joystick_data[2, :]

        # Apply filters to both axes
        x_filtered = apply_joystick_filters(x_position.copy(), self._joystick_filters, 'position')
        y_filtered = apply_joystick_filters(y_position.copy(), self._joystick_filters, 'position')
        x_derivative = apply_joystick_filters(np.gradient(x_filtered), self._joystick_filters, 'derivative')
        y_derivative = apply_joystick_filters(np.gradient(y_filtered), self._joystick_filters, 'derivative')

        if self._label_axis == 'dual':
            # 5-class dual-axis mode using position_peak
            labels, thresholds, dual_markers = create_5class_position_peak_labels(
                x_filtered, y_filtered, x_derivative, y_derivative,
                self._pp_deriv_thresh, self._pp_pos_thresh,
                self._pp_peak_window, self._pp_timeout
            )
            markers = {
                'method': 'position_peak_5class',
                'x_markers': dual_markers.get('x', {}),
                'y_markers': dual_markers.get('y', {}),
                'thresholds': thresholds
            }
        else:
            # Single-axis mode using position_peak
            if self._label_axis == 'x':
                position_data, derivative = x_filtered, x_derivative
            else:
                position_data, derivative = y_filtered, y_derivative

            labels, thresholds, axis_markers = create_position_peak_labels(
                position_data, derivative,
                self._pp_deriv_thresh, self._pp_pos_thresh,
                self._pp_peak_window, self._pp_timeout
            )
            markers = {
                'method': 'position_peak',
                'start': axis_markers.get('start', []),
                'peak': axis_markers.get('peak', []),
                'stop': axis_markers.get('stop', []),
                'rejected': axis_markers.get('rejected', []),
                'timeout': axis_markers.get('timeout', []),
                'thresholds': thresholds
            }

        # Add position and derivative to markers for plotting
        markers['x_filtered'] = x_filtered
        markers['y_filtered'] = y_filtered
        markers['x_derivative'] = x_derivative
        markers['y_derivative'] = y_derivative
        markers['x_position'] = x_position
        markers['y_position'] = y_position

        return labels, markers


if __name__ == '__main__':
    data = DataProcessor(config_file='/home/cleitner/code/lab/projects/ML/m-mode_nn/config/config.yaml')
