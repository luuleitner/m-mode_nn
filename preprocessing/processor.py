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
from visualization.plot_callback import plot_mmode
from utils.saving import init_dataset, append_and_save
from preprocessing.signal_utils import peak_normalization, Z_normalization, butter_bandpass_filter, Time_Gain_Compensation, extract_sliding_windows
# from preprocessing.signal_utils import analytic_signal

import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")


class DataProcessor():
    def __init__(self, config_file='config.yaml'):
        # Setup Config Parameters
        self._config_path = config_file  # Store for dimension checker
        self._config = load_config(config_file)
        self._np_seed_generator = setup_environment(self._config)

        # Set Operation Mode
        self._operation_mode = self._config.global_setting.run.mode
        self._debug_level = self._config.global_setting.run.config.debug.level if self._operation_mode == 'debug' else None

        # Set Processing Parameters
        # Signal Processing and Flags
        
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

        # ---Clipping
        self._clip_flag = self._config.preprocess.signal.clip.apply
        self._clip_initial_size = self._config.preprocess.signal.clip.initial_size
        self._clip_samples2remove_start = self._config.preprocess.signal.clip.samples2remove_start
        self._clip_samples2remove_end = self._config.preprocess.signal.clip.samples2remove_end

        #---Decimation
        self._decimation_flag = self._config.preprocess.signal.decimation.apply
        self._decimation_factor = self._config.preprocess.signal.decimation.factor

        #---Envelope
        self._envelope_flag = self._config.preprocess.signal.envelope.apply
        self._envelope_interp = self._config.preprocess.signal.envelope.interp
        self._envelope_padding_flag = self._config.preprocess.signal.envelope.padding.apply
        self._envelope_padding_mode = self._config.preprocess.signal.envelope.padding.mode
        self._envelope_padding_amount = self._config.preprocess.signal.envelope.padding.amount

        #---Log Compression
        self._logcompression_flag = self._config.preprocess.signal.logcompression.apply
        self._logcompression_dbrange = self._config.preprocess.signal.logcompression.db

        #---Normalization
        # In the standard mode the normalization is done channel wise for each exerpiment
        # One option that is not fully implemented is the possibility to normalize with precomputed values.
        # NOTE (20251228): The normalization using dataset wide precomputed values is not yet implemented.
        #  Currently norm is calculated channel-wise for each experiment even thought the infrastructure is already in place
        self._normalization_flag = self._config.preprocess.signal.normalization.apply
        self._normalization_technique = self._config.preprocess.signal.normalization.method
        self._normalization_minmax_path = self._config.preprocess.signal.normalization.minmax_path
        if self._normalization_minmax_path:
            self._minmax_normalization_values = np.loadtxt(self._normalization_minmax_path, delimiter=',')
            # Sanity check if signal processing and normalization fit together
            match_log = re.search(r"Log", self._normalization_minmax_path)
            match_env = re.search(r"Env", self._normalization_minmax_path)
            if match_log and not self._logcompression_flag:
                raise ValueError("Normalization file suggests log compression, but log compression is disabled.")
            if match_env and not self._envelope_flag:
                raise ValueError("Normalization file suggests envelope detection, but envelope detection is disabled.")

        # Tokens
        self._token_window = self._config.preprocess.tokenization.window
        self._token_stride = self._config.preprocess.tokenization.stride

        # Sequences
        self._sequence_window = self._config.preprocess.sequencing.window

        # Saving
        self._save_strategy = self._config.preprocess.data.save_ftype
        self._save_path_id = self._config.preprocess.data.id
        self._h5file = None

        # Setup metadata table and token ID tracker
        self._metastructure = ['token_id local', 'start', 'end', 'participant', 'session', 'experiment', 'token label']
        self._metadata = pd.DataFrame(columns=self._metastructure)
        self._tokenid = np.zeros(1)

        # Define paths to raw and processed data
        self._data_path_processed = os.path.join(self._config.preprocess.data.basepath, 'processed')
        self._data_path_raw = os.path.join(self._config.preprocess.data.basepath, 'raw')
        if self._data_path_raw is None:
            raise ValueError("Data path not found in the config file under 'experiment: path'")

        self._file_save_id = f'TokenWin{int(self._config.preprocess.tokenization.window):02}_TokenStr{int(self._config.preprocess.tokenization.stride):02}_SeqWin{int(self._config.preprocess.sequencing.window):02}'

        # Create unique output folder structure: dataset/params/run_timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_folder = os.path.join(self._data_path_processed, self._save_path_id)
        params_folder = os.path.join(dataset_folder, self._file_save_id)
        self._output_folder = os.path.join(params_folder, f"run_{timestamp}")
        
        # Create all directories
        os.makedirs(self._output_folder, exist_ok=True)
        logger.info(f"Created unique output folder: {self._output_folder}")

        # Other Flags
        self._save_flag = self._config.preprocess.tokenization.tokens2file
        self._set_token_startendID_flag = self._config.preprocess.tokenization.startendID


        # ---------------------------------------------
        # --------------PREPROCESS---------------------
        # ---------------------------------------------
        # (1) PREPROCESS: Load folder structure depending on selected strategy
        if self._config.preprocess.data.strategy == "all":
            self._load_fstructure()
        elif self._config.preprocess.data.strategy == "specific":
            self._load_fstructure(session=self._config.preprocess.configs.specific.selection.session,
                                  exp=self._config.preprocess.configs.specific.selection.experiment)
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
                    'label_key': 'y' if self._save_strategy == 'h5' else 'label'
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
                'token_window': self._token_window,
                'token_stride': self._token_stride,
                'sequence_window': self._sequence_window,
                'save_strategy': self._save_strategy
            },
            'input_data': {
                'data_path_raw': self._data_path_raw,
                'strategy': self._config.preprocess.data.strategy,
                'experiments_found': len(self._fstructure),
                'experiment_paths': self._fstructure
            }
        }
        
        # Copy normalization minmax file if it exists
        if self._normalization_minmax_path and os.path.exists(self._normalization_minmax_path):
            minmax_filename = os.path.basename(self._normalization_minmax_path)
            local_minmax_path = os.path.join(self._output_folder, minmax_filename)
            shutil.copy2(self._normalization_minmax_path, local_minmax_path)
            
            replication_info['processing_parameters']['normalization_minmax_file'] = minmax_filename
            replication_info['processing_parameters']['original_minmax_path'] = self._normalization_minmax_path
            logger.info(f"Copied normalization file: {minmax_filename}")
        
        replication_path = os.path.join(self._output_folder, 'replication_info.yaml')
        with open(replication_path, 'w') as f:
            yaml.dump(replication_info, f, default_flow_style=False, indent=2)
        
        logger.info(f"Replication info saved: {replication_path}")


    def _load_fstructure(self, session=None, exp=None):
        # Recursively search for valid experiment folders
        self._fstructure = [f for f in glob.glob(os.path.join(self._data_path_raw, "session*", "*"), recursive=True)
                            if os.path.isdir(f) and os.path.basename(f).isdigit()]

        # Apply session/experiment filters if strategy is 'specific'
        if exp and not session:
            logger.error("Cannot filter by 'experiment' without specifying 'session'.")
            return

        if session:
            self._fstructure = [f for f in self._fstructure
                                if any(f'session{s}' in os.path.split(os.path.dirname(f))[1]
                                       for s in session)]
            if exp:
                self._fstructure = [f for f in self._fstructure if int(os.path.basename(f)) in exp]


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

        # Parse participant and session info from folder path
        session_path = os.path.normpath(exp)
        self._experiment_id = os.path.basename(session_path)

        match = re.search(r"session(\d+)_W_(\d+)", session_path)
        if match:
            self._session_id = match.group(1)
            self._participant_id =  str(int(match.group(2)))
        else:
            self._session_id = None
            self._participant_id = None

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


    def _signal_processing(self, data):
        
        # Bandpass Filtering
        if self._bandpass_flag:
            data = butter_bandpass_filter(data,
                                          ax=1,
                                          lowcut=self._bandpass_lowcut,
                                          highcut=self._bandpass_highcut,
                                          fs=self._bandpass_fs,
                                          order=self._bandpass_order)
        
        # Time Gain Compensation (TGC)
        if self._tgc_flag:
            data = Time_Gain_Compensation(data, freq=self._tgc_fs, coef_att=self._tgc_coef_att)
        
        # Clipping
        if self._clip_flag:
            clip_end = data.shape[1] - self._clip_samples2remove_end
            data = data[:, self._clip_samples2remove_start:clip_end, :]

        #---Envelope
        if self._envelope_flag:
            # NOTE: @Bruno this introduces quite a significant signal lag i therefore converged back to the original implementation
            # data = envelope(analytic_signal(data, ax=1,
            #                                 interp=self._envelope_interp,
            #                                 padding=self._envelope_padding_flag,
            #                                 pad_mode=self._envelope_padding_mode,
            #                                 pad_amount=self._envelope_padding_amount))
            data = envelope(analytic_signal(data))

        #---Logcompression
        if self._logcompression_flag:
            data = logcompression(data, self._logcompression_dbrange)



        #---Normalization
        if self._normalization_flag:
            # Normalize Data
            if self._normalization_technique == 'peakZ':
                data  = Z_normalization(peak_normalization(data))

        # Decimation
        if self._decimation_flag:
            data = data[:, ::self._decimation_factor, :]

        return data


    def _tokenizer(self, data, label, plot_flag):
        # Determine number of sliding windows
        last_axis_len = data.shape[-1]
        num_windows = (last_axis_len - self._token_window) //  self._token_stride + 1

        # Calculate window start and end indices - for metadata usage only
        starts = [i * self._token_stride for i in range(num_windows)]
        ends = [start + self._token_window for start in starts]

        #Extracts sliding windows (tokens)
        token = extract_sliding_windows(data, ax=2, window_size=self._token_window, stride=self._token_stride) # (chs, samples, num_windows, window_size)
        token = token.swapaxes(0,2).swapaxes(1,2)  # (num_windows, chs, samples, window_size)
        token_label = np.squeeze(extract_sliding_windows(label, ax=2, window_size=self._token_window, stride=self._token_stride)) # (labels, num_windows, window_size)
        token_label = token_label.swapaxes(0,1)  # (num_windows, labels, window_size)

        token_label_mean = np.mean(token_label, axis=-1)
        x_label =  np.squeeze(token_label_mean[:, 1])
        y_label =  np.squeeze(token_label_mean[:, 2])

        token_label = (
                (x_label < 0).astype(int) * 2 +
                (y_label < 0).astype(int)
        )
        token_label = np.expand_dims(token_label, axis=1)

        # Add synthetic start/end markers to each token
        if self._set_token_startendID_flag:
            token = self.__add_startend_seq(token, self._minmax_normalization_values)
        else:
            pass

        # Clip accessive Tokens that do not fit into sequencer (depends on the window size)
        clipfloor = int(np.floor(token.shape[0] / self._sequence_window) * self._sequence_window)
        token = token[:clipfloor]
        token_label = token_label[:clipfloor]
        starts = starts[:clipfloor]
        ends = ends[:clipfloor]

        # Create meta datafile
        self._token_len = len(token_label)
        self.__create_meta(starts, ends, token_label)

        return token, token_label


    def _sequencer(self, data, label):
        num_sequences = int(data.shape[0] // self._sequence_window)
        sequence = data.reshape(num_sequences, self._sequence_window, data.shape[1], data.shape[2], data.shape[3])
        sequence_label = label.reshape(num_sequences, self._sequence_window, label.shape[-1])

        # Prepare sequence metadata for experimental accumulation
        sequence_ids = np.arange(num_sequences).astype(int)
        token2sequence_id = np.repeat(sequence_ids, self._sequence_window).astype(int)

        return sequence, sequence_label, token2sequence_id



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
        meta = pd.DataFrame(np.stack([tid, starts, ends, participant_id, session_id, experiment_id, np.squeeze(token_label)], axis=1),
                            columns=self._metastructure)
        self._metadata = pd.concat([self._metadata, meta], axis=0, ignore_index=True)


    def __save_data(self, data=None, label=None):
        filename = f"S{int(self._session_id):04}_P{int(self._participant_id):04}_E{int(self._experiment_id):04}_Xy.h5"
        self._experiment_file_id = os.path.join(f'P{int(self._participant_id):04}', filename)
        experiment_path = os.path.join(self._output_folder, self._experiment_file_id)

        if not os.path.exists(os.path.dirname(experiment_path)):
            os.makedirs(os.path.dirname(experiment_path))

        backend = self._save_strategy
        file_attr = f"_{backend}file"
        if not hasattr(self, file_attr) or getattr(self, file_attr) is None:
            setattr(self, file_attr, init_dataset(path=experiment_path, data_size=data.shape[1:], backend=backend))
        append_and_save(experiment_path, getattr(self, file_attr), data, label, config=self._config, backend=backend)   # [B, S, C, E] B...batch, S...sequence, C...channel, F...feature (flattend A-mode lines)

    def __clean_datapath(self, path):
        normalized_path = os.path.normpath(self._experiment_file_id)
        path_parts = normalized_path.split(os.sep, 1)
        new_path = path_parts[1] if len(path_parts) > 1 else ""
        return new_path


if __name__ == '__main__':
    data = DataProcessor(config_file='/home/cleitner/code/lab/projects/ML/m-mode_nn/config/config.yaml')
