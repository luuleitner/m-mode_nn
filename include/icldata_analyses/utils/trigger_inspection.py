import os
import numpy as np
from matplotlib import pyplot as plt


path = 'C:/Users/chule/OneDrive/002_PROJECTS/010_datasets/int/2025_wristUS/day3/session019_w006/e002'
file_joy = '_joystick.npy'
file_us = '_US_ch1.npy'
file_emg = '2.csv'

plot_emg = True

##### Ultrasound
data_us = np.load(os.path.join(path, file_us))

##### Joystick
data_joy = np.load(os.path.join(path, file_joy))
plot_joy_col = 3
data_joy_norm = (data_joy[:,plot_joy_col] - np.min(data_joy[:,plot_joy_col])) / (np.max(data_joy[:,plot_joy_col]) - np.min(data_joy[:,plot_joy_col]))

##### EMG
data_emg = np.loadtxt(os.path.join(path, file_emg), delimiter=',')
emg_channel = 1
emg_trigger = 3



#####################################################
##### PLOT
fig = plt.figure()

# Plot Ultrasound
ax1 = fig.add_subplot(211)
im = ax1.imshow(data_us.T, aspect='auto', cmap='gray')
ax1.set_xlabel('Slow Time')
ax1.set_ylabel('Fast Time')

# US trigger
ax2 = ax1.twinx()
ax2.plot(data_joy_norm[:-1], color='red')
ax2.set_ylabel('Normalized joy trigger')

# Plot EMG
ax3 = fig.add_subplot(212)
ax3.plot(data_emg[:,emg_channel], color='blue')
ax3.set_xlabel('Sample')
ax3.set_ylabel('EMG')

# EMG trigger
ax4 = ax3.twinx()
ax4.plot(data_emg[:, emg_trigger], color='red')
ax4.set_ylabel('EMG trigger')

fig.tight_layout()
# plt.savefig('signal&trigger.png')
plt.show()