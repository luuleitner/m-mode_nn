import numpy as np
import scipy.signal as ss
import yaml

import plotly.express as px
import matplotlib.pyplot as plt

from loader import DataLoader

from include.dasIT.dasIT.features import signal
from include.emg.emg_processing import filter

# Load configuration file
with open("../qvar_config.yaml", "r") as file:
    config = yaml.safe_load(file)

complete_data = DataLoader(config_file='../qvar_config.yaml')

########################################################################
########### ---- Qvar Data Processing
Qvar_data = [complete_data.qvar[channel][:, 1] for channel in range(len(complete_data.qvar))]

if config["qvar"]["use_QVnotchFilter"]:
    notch_config = config["qvar"]["notch_filter"]
    b_notch, a_notch = ss.iirnotch(w0=notch_config["w0"], Q=notch_config["Q"], fs=config["qvar"]["f_sampling"])
    Qvar_data = [ss.filtfilt(b_notch, a_notch, Qvar_data[channel]) for channel in range(len(Qvar_data))]

if config["qvar"]["use_QVbpFilter"]:
    bp_config = config["qvar"]["bandpass_filter"]
    Qvar_data = [filter.Butter(Qvar_data[channel], bp_config["low"], bp_config["high"], config["qvar"]["f_sampling"], bp_config["order"])
                 for channel in range(len(Qvar_data))]
    Qvar_data = [d.data_filtered for d in Qvar_data]

if config["qvar"]["use_SmoothSavGol"]:
    savgol_config = config["qvar"]["savgol_filter"]
    Qvar_data = [ss.savgol_filter(Qvar_data[channel], savgol_config['window'], savgol_config['order']) for channel in range(len(Qvar_data))]


########################################################################
########### ---- Plotting
num_channels = len(Qvar_data)  # each column corresponds to one channel

fig, axs = plt.subplots(num_channels, 1, figsize=(6, 7), dpi=300, sharex=True)

# In case there is only one channel, ensure axs is iterable
if num_channels == 1:
    axs = [axs]

for ch in range(num_channels):
    axs[ch].plot(Qvar_data[ch])
    axs[ch].set_title(f"Channel {ch+1}")

plt.tight_layout()
plt.show()

print()