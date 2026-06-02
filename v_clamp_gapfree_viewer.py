import matplotlib.pyplot as plt
import pyabf
import os
import numpy as np
from scipy.signal import butter, filtfilt

############################## Load the ABF file ##############################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/Ephys/Patch and LFP/Analyses/Patch/RD_SCLC_TumorCells_MonoCulture/Control'
recording = '2026_04_22_0015'

abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

############################## Globals ########################################
sweep = 0
print("total sweeps =", abf.sweepCount)

# sampling rate
fs = abf.dataRate

############################## Filter Functions ###############################
def bandpass_filter(data, fs, lowcut=1, highcut=1000, order=2):
    nyq = fs / 2

    # high-pass
    b_high, a_high = butter(order, lowcut / nyq, btype='high')
    data = filtfilt(b_high, a_high, data)

    # low-pass
    b_low, a_low = butter(order, highcut / nyq, btype='low')
    data = filtfilt(b_low, a_low, data)

    return data

############################## Create Plot ####################################
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# initial Vm trace
abf.setSweep(sweep, channel=0)
vm_filtered = bandpass_filter(abf.sweepY, fs)

line_vm, = axs[0].plot(abf.sweepX, vm_filtered, color='maroon')
axs[0].axhline(0, color='k', ls='--')
axs[0].set_xlabel(abf.sweepLabelX)
axs[0].set_ylabel('Vm (mV)')
axs[0].set_title("Filtered Membrane Potential (1–1000 Hz)")
axs[0].set_ylim(-100, 200)

# initial current trace
abf.setSweep(sweep, channel=1)
i_filtered = bandpass_filter(abf.sweepY, fs)

line_i, = axs[1].plot(abf.sweepX, i_filtered, color='olive')
axs[1].set_xlabel(abf.sweepLabelX)
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Filtered Current (1–1000 Hz)")
axs[1].set_ylim(-100, 100)

plt.subplots_adjust(hspace=0.4)

############################## Update Function ################################
def update_plot():
    global sweep

    # Vm
    abf.setSweep(sweep, channel=0)
    vm_filtered = bandpass_filter(abf.sweepY, fs)
    line_vm.set_ydata(vm_filtered)

    # Current
    abf.setSweep(sweep, channel=1)
    i_filtered = bandpass_filter(abf.sweepY, fs)
    line_i.set_ydata(i_filtered)

    fig.suptitle(f"Sweep {sweep+1} / {abf.sweepCount}")
    fig.canvas.draw_idle()

############################## Key Press Handler ##############################
def on_key(event):
    global sweep

    if event.key == "right":
        sweep = min(sweep + 1, abf.sweepCount - 1)
        update_plot()

    elif event.key == "left":
        sweep = max(sweep - 1, 0)
        update_plot()

fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()