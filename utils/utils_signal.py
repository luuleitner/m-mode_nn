import h5py
import numpy as np
from omegaconf import OmegaConf




def peak_normalization(data, maximum=None, static=False):
    data = data.astype(float)
    if maximum is None:
        maximum = np.array([np.abs(channel).max() for channel in data])
    maximum = np.repeat(np.expand_dims(maximum, 1), data.shape[1], axis=-1)
    maximum = np.repeat(np.expand_dims(maximum, 2), data.shape[2], axis=-1)
    data /= maximum

    #NOTE: use to precompute the min/max values of the complete dataset
    if static:
        minmax = np.array([np.min(data), np.max(data)])
        minmax = smart_round(minmax)
        return data, minmax
    else:
        return data

def Z_normalization(data, sigma=None, static=False):
    data = data.astype(float)
    if sigma is None:
        sigma = np.array([np.std(data) for channel in data])
    sigma = np.repeat(np.expand_dims(sigma, 1), data.shape[1], axis=-1)
    sigma = np.repeat(np.expand_dims(sigma, 2), data.shape[2], axis=-1)
    data /= sigma

    #NOTE: use to precompute the min/max values of the complete dataset
    if static:
        minmax = np.array([np.min(data), np.max(data)])
        minmax = smart_round(minmax)
        return data, minmax
    else:
        return data

def smart_round(x):
    # Create Boolean array with all values closer to 0
    closer_to_zero = np.abs(x) < 0.5
    # Otherwise, apply floor or ceil
    y = np.where(x < 0, np.floor(x), np.ceil(x))
    # Filter
    y[closer_to_zero] = 0
    return y


def openhd5(path):
    data = h5py.File(path, 'r')
    return data

if __name__ == '__main__':
    import os
    import glob
    from tqdm import tqdm
    from include.dasIT.dasIT.features.signal import analytic_signal, envelope, logcompression

    config = OmegaConf.load('/home/cleitner/code/lab/projects/ML/m-mode_nn/config/config.yaml')

    data_path_raw = os.path.join(config.preprocess.data.basepath, 'raw')
    clip_samples2keep = config.preprocess.signal.clip.samples2keep

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
        data, _ = peak_normalization(data, static=True)
        _, mima_v = Z_normalization(data, static=True)
        minmax_values_postnorm.append(mima_v)
    minmax_values_postnorm = np.array(minmax_values_postnorm)

    #HACK: This the computation and file path saving is ghardcoded! check before saving
    filename = 'minmax_Envdata.csv'
    np.savetxt(os.path.join('/home/cleitner/code/lab/projects/ML/m-mode_nn/config', filename), minmax_values_postnorm, delimiter=',')







