import os
import glob
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from include.dasIT.dasIT.features.signal import analytic_signal, envelope, logcompression

from preprocessing.signal_utils import peak_normalization, Z_normalization


def precompute_normalization_values(config_path: str, output_filename: str = 'minmax_Envdata.csv'):
    """
    Precompute min/max normalization values for the dataset.

    Args:
        config_path: Path to the config.yaml file
        output_filename: Name of the output CSV file
    """
    config = OmegaConf.load(config_path)

    data_path_raw = os.path.join(config.preprocess.data.basepath, 'raw')

    fstructure = [f for f in glob.glob(os.path.join(data_path_raw, "session*", "*"), recursive=True)
                        if os.path.isdir(f) and os.path.basename(f).isdigit()]

    minmax_values_postnorm = []
    for i, e in enumerate(tqdm(fstructure, desc="Processing experiments")):
        # Find ultrasound and joystick files
        files = glob.glob(os.path.join(e, "_US_ch*"))
        nbr_us_channels = len(files)
        files += glob.glob(os.path.join(e, "_joystick*"))

        # Load and min/max normalize data
        data = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            data.append(d.T)

        data = np.stack(data[:nbr_us_channels], axis=0)

        # envelope
        data = envelope(analytic_signal(data, ax=1))
        # #logcompress
        # data = logcompression(data, self._logcompression_dbrange)
        # normalize
        data, _ = peak_normalization(data, precompute=True)
        _, mima_v = Z_normalization(data, static=True)
        minmax_values_postnorm.append(mima_v)
    minmax_values_postnorm = np.array(minmax_values_postnorm)

    # Save to config directory
    config_dir = os.path.dirname(config_path)
    output_path = os.path.join(config_dir, output_filename)
    np.savetxt(output_path, minmax_values_postnorm, delimiter=',')
    print(f"Saved normalization values to: {output_path}")


if __name__ == '__main__':
    config_path = '/home/cleitner/code/lab/projects/ML/m-mode_nn/config/config.yaml'
    precompute_normalization_values(config_path)
