import numpy as np
import scipy.signal as ss
from scipy.spatial.distance import cdist
import yaml

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from loader import DataLoader
from include.dasIT.dasIT.features import signal
from include.emg.emg_processing import filter

# Load configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load Data
ICL_data = DataLoader(config_file='config.yaml')

########################################################################
########### ---- Ultrasound Data Processing
US_data = ICL_data.icl[0][0]

if config["ultrasound"]["use_USfilter"]:
    US_filter = config["ultrasound"]["US_filter"]
    US_data = signal.RFfilter(signals=US_data,
                              fcutoff_band=US_filter["fcutoff_band"],
                              fsampling=US_filter["fsampling"],
                              type=US_filter["type"],
                              order=US_filter["order"])

if config["ultrasound"]["clipUSat"]:
    US_data = np.squeeze(US_data.signal[:config["ultrasound"]["clipUSat"], :, :])

if config["ultrasound"]["use_USenvelope"]:
    US_data = signal.envelope(signal.analytic_signal(US_data))

if config["ultrasound"]["use_USlogcompression"]:
    US_data = signal.logcompression(US_data, config["ultrasound"]["dbrange_compression"])

########################################################################
########### ---- Qvar Data Processing
if config["qvar_channels"] != []:

    Qvar_data = [ICL_data.qvar[channel][:, 1] for channel in range(len(ICL_data.qvar))]

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

# ########################################################################
# ########### ---- Joystick
if 'joystick' in config["data"]:
    print(f'Using Joystick as Ground Truth.')
    groundtruth = ICL_data.icl[0][1][:,[1,2]]
    plot_names = ['Joystick x-axis', 'Joystick y-axis']
    show_legends = [True, True]

########################################################################
########### ---- Leap Markers Processing
elif 'leap_markers' in config["data"]:
    Marker_data = ICL_data.icl[0][1]
    # Leap Motion MarkerSet
    # R is on the wrist
    # M is the first finger joint on the hand
    # P is the proximal second finger joint
    # d is the most distal third finger joint
    # ["RS", "M1", "M2", "M3", "M4", "M5", "P1", "P2", "P3", "P4", "P5", "D1", "D2", "D3","D4","D5"]
    thumb_tip = np.stack((Marker_data[:,33], Marker_data[:,34], Marker_data[:,35])).T
    index_tip = np.stack((Marker_data[:,36], Marker_data[:,37], Marker_data[:,38])).T

    groundtruth = np.squeeze(np.sqrt(np.sum((thumb_tip - index_tip) ** 2, axis=1)))
    groundtruth = np.stack((groundtruth, groundtruth)).T
    plot_names = ['Euclidean Distance', '']
    show_legends = [True, False]


else:
    print('WARNING: No Ground Truth available.')


########################################################################
# Visualization
if not config["plotting"]["use_plotly"]:
    num_datalines = US_data.shape[1]  # Each column corresponds to one channel
    fig1, axs = plt.subplots(num_datalines, 1, figsize=(6, 7), dpi=300, sharex=True)
    if num_datalines == 1:
        axs = [axs]
    for ch in range(US_data.shape[1]):
        axs[ch].imshow(US_data[:, ch, :], cmap="gray")
        axs[ch].set_title(f"US Channel {ch + 1}", fontsize=14, fontweight='bold')
        axs[ch].tick_params(axis='both', which='major', labelsize=12, width=2)
    plt.tight_layout()
    plt.show()

    num_datalines = len(Qvar_data)
    fig2, axs = plt.subplots(num_datalines, 1, figsize=(6, 7), dpi=300, sharex=True)
    if num_datalines == 1:
        axs = [axs]
    for ch in range(len(Qvar_data)):
        axs[ch].plot(Qvar_data[ch])
        axs[ch].set_title(f"Channel {ch + 1}", fontsize=14, fontweight='bold')
        axs[ch].tick_params(axis='both', which='major', labelsize=12, width=2)

    plt.tight_layout()
    plt.show()

else:
    # Ensure Plotly renders in the browser
    pio.renderers.default = "browser"

    # Plot US_data as images
    num_datalines = US_data.shape[1]
    ch_pos = config["experiment"]["US_channel_positions"]
    fig1 = make_subplots(rows=num_datalines, cols=1,
                         subplot_titles=[f"US Channel Position: {pos}" for pos in ch_pos],
                         specs=[[{"secondary_y": True}] for _ in range(num_datalines)])

    for ch in range(num_datalines):
        show_colorbar = ch == 0
        show_legend = ch == 0

        fig1.add_trace(
            go.Scatter(y=groundtruth[:,0], mode="lines", name=plot_names[0], line=dict(color='blue'), opacity = 0.5, showlegend=show_legend),
            row=ch + 1, col=1, secondary_y=False
        )
        fig1.add_trace(
            go.Scatter(y=groundtruth[:,1], mode="lines", name=plot_names[1], line=dict(color='orange'), opacity = 0.5, showlegend=show_legend),
            row=ch + 1, col=1, secondary_y=False
        )
        fig1.add_trace(
            go.Heatmap(z=US_data[:, ch, :],
                       colorscale="gray",
                       showscale=show_colorbar,
                       colorbar=dict(
                           len=0.5,
                           yanchor="middle",
                           y=0.5
                        ),
                       ),
            row=ch + 1, col=1, secondary_y=False
        )

    fig1.update_layout(
        height=500 * num_datalines,
        width=2500,
        title_text="US Data Visualization",
        font=dict(size=16, family="Arial", weight='bold'),
        margin=dict(l=10, r=10, t=50, b=10),  # Reduce margins
        autosize=False  # Prevent unwanted stretching
    )
    fig1.show()

    if config["qvar_channels"] != []:
        # Plot Qvar_data as line charts
        num_datalines = len(Qvar_data)
        fig2 = make_subplots(rows=num_datalines, cols=1,
                             subplot_titles=[f"Channel {ch + 1}" for ch in range(num_datalines)])

        for ch in range(num_datalines):
            fig2.add_trace(
                go.Scatter(y=Qvar_data[ch], mode="lines", name=f"Channel {ch + 1}"),
                row=ch + 1, col=1
            )

        fig2.update_layout(height=300 * num_datalines, width=2500, title_text="Qvar Data Visualization", font=dict(size=16, family="Arial", weight='bold'))
        fig2.show()
