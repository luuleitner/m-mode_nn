import h5py
import numpy as np
from omegaconf import OmegaConf
from scipy.signal import butter, sosfiltfilt, hilbert, resample
from numpy.lib.stride_tricks import sliding_window_view



def butter_bandpass_filter(data, ax, lowcut, highcut, fs, order) -> np.ndarray:
    sos = butter(order, [lowcut,highcut], fs=fs, btype='bandpass', output="sos")
    return sosfiltfilt(sos, data, axis=ax)

def Time_Gain_Compensation(US, freq, coef_att): # TODO: this could be speed up by making the multiplication inplace, but care needs to be taken on the dtype of the input array
    Sequence_depth = np.arange(0, US.shape[1], 1) * (1/freq) * (1e2 * 1540 / 2)
    attenuation = np.exp(coef_att * (freq/10e6) * Sequence_depth)
     
    return US * attenuation[np.newaxis, :, np.newaxis]

def analytic_signal(signal, ax, interp=False, padding=False, padding_mode = None, pad_amount= 0, *kwargs):
    if padding: # Pad signal to reduce edge effects if requested
        padding_list = [(0,0) for x in range(len(signal.shape))]
        padding_list[ax] = (pad_amount, pad_amount)
        signal = np.pad(signal, padding_list, mode= padding_mode, *kwargs)
        
    hilbert_transformed_signal = hilbert(signal, axis=ax)
    
    if padding: # Remove padding after Hilbert transform if padding was applied
        hilbert_transformed_signal = hilbert_transformed_signal.take(indices=range(pad_amount, hilbert_transformed_signal.shape[ax] - pad_amount), axis=ax)
        
    if interp: # Interpolate signal if requested
        hilbert_transformed_signal_interp = resample(hilbert_transformed_signal, hilbert_transformed_signal.shape[0] * 3, axis=-1)
        return hilbert_transformed_signal_interp
    else:
        return hilbert_transformed_signal


def peak_normalization(data, maximum=None, static=False):
    data = data.astype(float)
    if maximum is None:
        maximum = np.amax(data, axis=(1,2)) # (frames, channels)
        minimum = np.amin(data, axis=(1,2)) # (frames, channels)

    data -= minimum[:, np.newaxis, np.newaxis]  # Broadcasting over samples (frames, sampels, channels)
    data /= (maximum[:, np.newaxis, np.newaxis] - minimum[:, np.newaxis, np.newaxis])  # Broadcasting over samples (frames, sampels, channels)

    #NOTE: use to precompute the min/max values of the complete dataset
    if static: #TODO: Currently this does not perform the same normalization approach as the dynamic . It only saves one 
        minmax = np.array([minimum, maximum])
        minmax = smart_round(minmax)
        return data, minmax
    else:
        return data

def Z_normalization(data, sigma=None, static=False):
    data = data.astype(float)
    if sigma is None:
        sigma = np.std(data, axis=(1,2))  # (frames, channels)
        mean = np.mean(data, axis=(1,2))  # (frames, channels)
        
    data -= mean[:, np.newaxis, np.newaxis]  # Broadcasting over samples (frames, sampels, channels)
    data /= sigma[:, np.newaxis, np.newaxis]  # Broadcasting over samples (frames, sampels, channels)

    #NOTE: use to precompute the mean/sigma values of the complete dataset
    if static:
        meansigma = np.array([mean, sigma])
        meansigma = smart_round(meansigma)
        return data, meansigma
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







