import re

import numpy as np
import yaml

import os
import glob
import argparse

from scipy.signal import decimate


class DataLoader():
    def __init__(self, config_file='config.yaml'):
        self._config_file = config_file
        self._config = self.load_config()
        # The data path is now read from the config YAML under experiment.path.
        # self._data_path = self._config.get('experiment', {}).get('path', None)
        # self._data_path = os.path.join(self._config.get('experiment', {}).get('basepath', None), f"exp_{self._config.get('experiment', {}).get('exp_nbr', None)}")
        self._data_path = os.path.join(self._config.get('experiment', {}).get('basepath', None), f"{self._config.get('experiment', {}).get('exp_nbr', None)}")
        if self._data_path is None:
            raise ValueError("Data path not found in the config file under 'experiment: path'")

        self._qvar, self._qvar_id = self.load_csv(self.load_files())
        self._icl, self._icl_id = self.load_npy(self.load_files())


    def load_config(self):
        with open(self._config_file, 'r') as f:
            return yaml.safe_load(f)

    def load_files(self, fileext=['*.npy', '*.NPY', '*.csv', '*.CSV']):
        # Recursively search for files with the given extensions in the data path.
        files = []
        for ext in fileext:
            files.extend(glob.glob(os.path.join(self._data_path, '**', ext), recursive=True))
        return np.unique(files)



    def load_csv(self, files):
        csv_data = []
        id_qvar = []
        qvar_channels = self._config.get('qvar_channels', [])

        # If qvar_channels is explicitly an empty list, return empty lists
        if qvar_channels == []:
            print("qvar_channels is empty in the YAML file. No data will be loaded.")
            return csv_data, id_qvar

        pattern = re.compile(r"qvar(\d+)", re.IGNORECASE)

        for idx, file in enumerate(files):
            if file is None:
                print(f"Skipping None entry at index {idx}.")
                continue

            match = pattern.search(file)
            if not match:
                print(f"Skipping CSV file {os.path.basename(file)}: no 'qvar' channel number found.")
                continue

            try:
                channel_num = int(match.group(1))
            except ValueError:
                print(f"Skipping CSV file {os.path.basename(file)}: unable to convert channel number to int.")
                continue

            if qvar_channels and channel_num not in qvar_channels:
                print(
                    f"Skipping CSV file {os.path.basename(file)}: channel {channel_num} not in specified qvar channels {qvar_channels}.")
                continue

            try:
                data = np.genfromtxt(file, delimiter=',', skip_header=1)
                print(f"Loading CSV file: {os.path.basename(file)} (channel {channel_num})")
                csv_data.append(data)
                id_qvar.append(f'qvar_channel_{channel_num}')
            except Exception as e:
                print(f"Error reading CSV file {file}: {e}")

        return csv_data, id_qvar

    def load_npy(self, files):
        # Handle the case when files is None
        if files is None:
            return [], []

        us_data = []
        icl_eq_data = []
        id_icl_us = []
        id_icl_equipment = []
        data_keys = self._config.get('data', []) or []

        # Iterate over files and load those that are NPY and whose base name contains any key from the config.
        us_channels = [f'ch{c}' for c in self._config['us_channels']]
        us_files = [f for f in files if
                    f.lower().endswith('.npy') and any(ch in os.path.basename(f).casefold() for ch in us_channels)]
        icl_eq_files = [f for f in files if
                    f.lower().endswith('.npy') and any(key.casefold() in os.path.basename(f).casefold() for key in data_keys)
                        and not any(ch in os.path.basename(f).casefold() for ch in us_channels)]

        for file in us_files:
            basename = os.path.basename(file)
            try:
                d = np.load(file, allow_pickle=True)
                d = d[::self._config['decimation'],:]
                print(f"Loading NPY file: {basename}")
                us_data.append(d)
                id_icl_us.append(os.path.splitext(basename)[0])
            except Exception as e:
                print(f"Error reading NPY file {basename}: {e}")
        us_data = np.stack(us_data)
        us_data = np.moveaxis(us_data, 2, 0)

        for file in icl_eq_files:
            basename = os.path.basename(file)
            try:
                d = np.load(file, allow_pickle=True)
                d = d[::self._config['decimation'],:]
                print(f"Loading NPY file: {basename}")
                icl_eq_data.append(d)
                id_icl_equipment.append(os.path.splitext(basename)[0])
            except Exception as e:
                print(f"Error reading NPY file {basename}: {e}")

        npy_data = [us_data, np.squeeze(np.stack(icl_eq_data))]
        id_icl_equipment = [id_icl_us, id_icl_equipment]
        return npy_data, id_icl_equipment


    def load_all_data(self):
        files = self.load_files()
        csv_list, _ = self.load_csv(files)
        npy_list, _ = self.load_npy(files)
        # Return a tuple containing lists for CSV and NPY data.
        return csv_list, npy_list

    def run(self):
        print(f"Start loading data using config '{os.path.basename(self._config_file)}'...")
        csv_list, npy_list = self.load_all_data()
        print(f"Loaded {len(csv_list)} CSV file(s) and {len(npy_list)} NPY file(s).")
        return csv_list, npy_list

    @property
    def qvar(self):
        return self._qvar

    @property
    def icl(self):
        return self._icl, self._icl_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load CSV and NPY files (filtered by YAML config) from a directory into pandas DataFrames."
    )
    # Only an optional argument for the config file is required.
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    loader = DataLoader(args.config)
    loader.run()
