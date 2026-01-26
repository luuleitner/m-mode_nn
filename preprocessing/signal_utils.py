import h5py
import numpy as np
import spm1d
from scipy.signal import butter, sosfiltfilt, hilbert, resample, medfilt, savgol_filter
from numpy.lib.stride_tricks import sliding_window_view


# ============== Filter Functions ==============

def butter_bandpass_filter(data, ax, lowcut, highcut, fs, order) -> np.ndarray:
    """Apply butterworth bandpass filter."""
    sos = butter(order, [lowcut,highcut], fs=fs, btype='bandpass', output="sos")
    return sosfiltfilt(sos, data, axis=ax)

def butter_lowpass_filter(data, ax, cutoff, fs, order):
    """Apply butterworth lowpass filter."""
    sos = butter(order, cutoff, fs=fs, btype='lowpass', output="sos")
    return sosfiltfilt(sos, data, axis=ax)

def gaussian_smooth(data, fwhm):
    """Apply gaussian kernel smoothing using spm1d."""
    return spm1d.util.smooth(data, fwhm=fwhm)

def moving_average(data, window_size):
    """Apply moving average filter."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def median_filter(data, kernel_size):
    """Apply median filter."""
    return medfilt(data, kernel_size=kernel_size)

def savitzky_golay(data, window_length, polyorder):
    """Apply Savitzky-Golay filter."""
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

def apply_joystick_filters(data, filter_config, signal_type):
    """
    Apply all enabled filters for the given signal type (raw or derivative).
    """
    signal_config = filter_config.get(signal_type, {})

    # Apply lowpass filter first
    lowpass_cfg = signal_config.get('lowpass', {})
    if lowpass_cfg.get('enabled', False):
        cutoff = lowpass_cfg.get('cutoff', 10)
        fs = lowpass_cfg.get('fs', 100)
        order = lowpass_cfg.get('order', 4)
        ax = lowpass_cfg.get('ax', 0)
        data = butter_lowpass_filter(data, ax, cutoff, fs, order)

    # Apply gaussian filter second
    gaussian_cfg = signal_config.get('gaussian', {})
    if gaussian_cfg.get('enabled', False):
        fwhm = gaussian_cfg.get('fwhm', 5.0)
        data = gaussian_smooth(data, fwhm)

    # Apply moving average filter third
    moving_avg_cfg = signal_config.get('moving_average', {})
    if moving_avg_cfg.get('enabled', False):
        window_size = moving_avg_cfg.get('window_size', 5)
        data = moving_average(data, window_size)

    # Apply median filter fourth
    median_cfg = signal_config.get('median', {})
    if median_cfg.get('enabled', False):
        kernel_size = median_cfg.get('kernel_size', 5)
        data = median_filter(data, kernel_size)

    # Apply Savitzky-Golay filter fifth
    savgol_cfg = signal_config.get('savitzky_golay', {})
    if savgol_cfg.get('enabled', False):
        window_length = savgol_cfg.get('window_length', 5)
        polyorder = savgol_cfg.get('polyorder', 2)
        data = savitzky_golay(data, window_length, polyorder)

    return data


# ============== Ultrasound Specific Functions ==============


def Time_Gain_Compensation(US, freq, coef_att):
    # TODO: this could be speed up by making the multiplication inplace, but care needs to be taken on the dtype of the input array
    Sequence_depth = np.arange(0, US.shape[1], 1) * (1/freq) * (1e2 * 1540 / 2)
    attenuation = np.exp(coef_att * (freq/10e6) * Sequence_depth)
     
    return US * attenuation[np.newaxis, :, np.newaxis]


def analytic_signal(signal, ax, interp=False, padding=False, pad_mode = None, pad_amount= 0, *kwargs):
    if padding: # Pad signal to reduce edge effects if requested
        padding_list = [(0,0) for x in range(len(signal.shape))]
        padding_list[ax] = (pad_amount, pad_amount)
        signal = np.pad(signal, padding_list, mode= pad_mode, *kwargs)
        
    hilbert_transformed_signal = hilbert(signal, axis=ax)
    
    if padding: # Remove padding after Hilbert transform if padding was applied
        hilbert_transformed_signal = hilbert_transformed_signal.take(indices=range(pad_amount, hilbert_transformed_signal.shape[ax] - pad_amount), axis=ax)
        
    if interp: # Interpolate signal if requested
        hilbert_transformed_signal_interp = resample(hilbert_transformed_signal, hilbert_transformed_signal.shape[0] * 3, axis=-1)
        return hilbert_transformed_signal_interp
    else:
        return hilbert_transformed_signal


# ============== Normalization Functions ==============

def peak_normalization(data, minmax=None, precompute=False):
    data = data.astype(float)
    if minmax is None:
        maximum = np.amax(data, axis=(1,2)) # (frames, channels)
        minimum = np.amin(data, axis=(1,2)) # (frames, channels)
    else:
        #TODO: normalization on precomputed values is not yet implemented albeit infrastructure (eg. below exists)
        # minimum = minmax[:,0]
        # maximum = minmax[:,1]
        pass

    data -= minimum[:, np.newaxis, np.newaxis]  # Broadcasting over samples (frames, sampels, channels)
    data /= (maximum[:, np.newaxis, np.newaxis] - minimum[:, np.newaxis, np.newaxis])  # Broadcasting over samples (frames, sampels, channels)

    #NOTE: precompute command is used to precompute the min/max values of the complete dataset
    # to precompute use the file in ./data/precompute_normalization_values.py
    if precompute:
        minmax = np.array([minimum, maximum])
        minmax = smart_round(minmax)
        return data, minmax
    else:
        return data

def Z_normalization(data, meansigma=None, precompute=False):
    data = data.astype(float)
    if meansigma is None:
        sigma = np.std(data, axis=(1,2))  # (frames, channels)
        mean = np.mean(data, axis=(1,2))  # (frames, channels)
    else:
        # TODO: normalization on precomputed values is not yet implemented albeit infrastructure (eg. below exists)
        # mean = meansigma[:,0]
        # sigma = meansigma[:,1]
        pass
        
    data -= mean[:, np.newaxis, np.newaxis]  # Broadcasting over samples (frames, sampels, channels)
    data /= sigma[:, np.newaxis, np.newaxis]  # Broadcasting over samples (frames, sampels, channels)

    #NOTE: precompute command is used to precompute the mean/sigma values of the complete dataset
    # to precompute use the file in ./data/precompute_normalization_values.py
    if precompute:
        meansigma = np.array([mean, sigma])
        meansigma = smart_round(meansigma)
        return data, meansigma
    else:
        return data


# ============== Math Functions ==============

def smart_round(x):
    # Create Boolean array with all values closer to 0
    closer_to_zero = np.abs(x) < 0.5
    # Otherwise, apply floor or ceil
    y = np.where(x < 0, np.floor(x), np.ceil(x))
    # Filter
    y[closer_to_zero] = 0
    return y

def extract_sliding_windows(data, ax, window_size, stride):
    data = sliding_window_view(data, axis=ax, window_shape=window_size)
    indexes = []
    for x in range(len(data.shape)):
        indexes.append(slice(0, data.shape[x]))
    indexes[ax] = slice(0, data.shape[ax], stride)
    data = data[tuple(indexes)]
    
    return data


# ============== Data Functions ==============

def openhd5(path):
    data = h5py.File(path, 'r')
    return data



