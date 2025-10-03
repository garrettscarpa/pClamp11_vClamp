import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pyabf
import pyabf.filter
import numpy as np
from scipy.signal import butter, filtfilt
import csv
import os
import pandas as pd

############################## Load the ABF file ##############################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data'
recording = '2025_09_12_0024'

display_window = 50  # ms
isi_latency = 3  # ms
window_ms = 10
amp_threshold = 10
duration_threshold = (1.5, 8)

abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

############################# User variables ##################################
hpf_freq = 1      # High-pass cutoff freq in Hz
lpf_freq = 1000   # Low-pass cutoff freq in Hz
sigma = 0.5       # Gaussian smoothing (ms)

save_dir = os.path.join(root, 'mini_timestamps')
csv_filename = os.path.join(save_dir, "threshold_crossings_" + recording + ".csv")

######################## Filter Functions #####################################

def highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

######################## Adaptive Threshold ###################################

def adaptive_threshold(signal, multiplier=-8):
    mad = np.median(np.abs(signal - np.median(signal)))
    noise_std_estimate = mad / 0.6745
    return multiplier * noise_std_estimate

######################## Data Processing ######################################

print("total sweeps =", abf.sweepCount)
abf.setSweep(0, channel=0)
fs = abf.dataRate  # Sampling rate in Hz

raw_data = abf.sweepY

# Apply filters
filtered_data = highpass_filter(raw_data, hpf_freq, fs)
filtered_data = lowpass_filter(filtered_data, lpf_freq, fs)

# Gaussian smoothing applied in-place to abf.sweepY (works on abf object)
pyabf.filter.gaussian(abf, sigma)

######################## Initialize Detection Variables #######################

crossing_times = []

######################## Event Detection Functions ############################

def measure_event_properties(signal, time, crossing_times, fs, window_ms=window_ms, amp_threshold=amp_threshold, duration_threshold = duration_threshold):
    filtered_events = []
    window_samples = int(window_ms / 1000 * fs)
    n_crossings = len(crossing_times)
    print(f"Checking {n_crossings} crossings")

    for i, ct in enumerate(crossing_times):
        if i % 50 == 0:
            print(f"Processing crossing {i+1} / {n_crossings}")

        idx = np.argmin(np.abs(time - ct))
        if idx + window_samples > len(signal):
            continue

        segment = signal[idx : idx + window_samples]
        if len(segment) == 0:
            continue

        peak_amp = np.max(np.abs(segment))
        if peak_amp < amp_threshold:
            continue

        threshold_level = 0.1 * peak_amp
        above_thresh = np.where(np.abs(segment) > threshold_level)[0]
        if len(above_thresh) == 0:
            continue

        duration_samples = above_thresh[-1] - above_thresh[0]
        duration_ms = duration_samples / fs * 1000

        if duration_threshold[0] <= duration_ms <= duration_threshold[1]:
            filtered_events.append(ct)

    print(f"Filtered down to {len(filtered_events)} events")
    return filtered_events

def detect_crossings(threshold_value):
    global crossing_times
    crossing_times.clear()

    if os.path.exists(csv_filename):
        print(f"CSV file {csv_filename} exists, loading crossings from file instead of recalculating.")
        df = pd.read_csv(csv_filename)
        crossing_times = df['Timestamp'].tolist()
        return

    raw_crossings = []
    for i in range(1, len(filtered_data)):
        y1 = filtered_data[i-1]
        y2 = filtered_data[i]
        t = abf.sweepX[i]
        if (y1 < threshold_value <= y2) or (y1 >= threshold_value > y2):
            raw_crossings.append(t)

    filtered_crossings = measure_event_properties(
        signal=filtered_data,
        time=abf.sweepX,
        crossing_times=raw_crossings,
        fs=fs,
        window_ms=window_ms,
        amp_threshold=amp_threshold,
        duration_threshold=(duration_threshold)
    )
    crossing_times = filtered_crossings

auto_threshold = adaptive_threshold(filtered_data)

def save_data(event=None):
    threshold_value = auto_threshold
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "threshold_crossings_" + recording + ".csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Threshold"])
        for crossing in crossing_times:
            writer.writerow([crossing, threshold_value])
    print(f"Data saved to {filename}")

########################### Plot Setup ########################################

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(abf.sweepX, filtered_data, color='maroon')
ax.axhline(0, color='k', ls='--')

line, = ax.plot(abf.sweepX, np.ones_like(abf.sweepX) * auto_threshold,
               color='red', linestyle='--', label="Detection Threshold")

ax.set_xlabel(abf.sweepLabelX)
ax.set_ylabel("Current (pA)")
ax.set_title("Filtered & Smoothed Current Trace")
ax.set_ylim(-200, 200)
ax.legend()

plt.subplots_adjust(bottom=0.3)


######################## Event Visualization ###################################

current_event = 0
crossings_data = []

def remove_duplicate_crossings(crossings_df, time_window_ms=display_window):
    crossings_df = crossings_df.sort_values(by='Timestamp').reset_index(drop=True)
    filtered_crossings = []
    last_crossing_time = -np.inf
    for _, row in crossings_df.iterrows():
        crossing_time = row['Timestamp']
        if crossing_time - last_crossing_time >= time_window_ms / 1000:
            filtered_crossings.append(row)
            last_crossing_time = crossing_time
    return pd.DataFrame(filtered_crossings)

def view_saved_detections(csv_filename, abf, fs, time_window_ms=display_window):
    global current_event, crossings_data

    crossings_data = pd.read_csv(csv_filename)
    crossings_data = remove_duplicate_crossings(crossings_data, time_window_ms=isi_latency).reset_index(drop=True)

    time_window_samples = int(time_window_ms / 1000 * fs)
    rejected_indices = set()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)
    
    def update_plot(event_idx):
        ax.clear()
        if 0 <= event_idx < len(crossings_data):
            crossing_time = crossings_data['Timestamp'][event_idx]
            closest_index = np.argmin(np.abs(abf.sweepX - crossing_time))
            
            start_idx = max(0, closest_index - time_window_samples // 2)
            end_idx = min(len(abf.sweepX), closest_index + time_window_samples // 2)
            signal_segment = filtered_data[start_idx:end_idx]
            time_segment = abf.sweepX[start_idx:end_idx]
            
            # Find most negative point (min y) in the window
            min_idx_local = np.argmin(signal_segment)
            min_time = time_segment[min_idx_local]
            min_value = signal_segment[min_idx_local]
            
            if event_idx in rejected_indices:
                ax.set_facecolor('mistyrose')  # light red
                status = "REJECTED"
            else:
                ax.set_facecolor('honeydew')   # light green
                status = "Accepted (default)"
            
            ax.plot(time_segment, signal_segment, color='maroon')
            ax.axvline(min_time, color='red', linestyle='--', label='Event (min point)')
            ax.axhline(0, color='k', ls='--')
            
            ax.set_title(f"Event {event_idx + 1}/{len(crossings_data)} at {min_time:.4f}s — {status}")
            ax.set_xlabel(abf.sweepLabelX)
            ax.set_ylabel("Current (pA)")
            ax.legend()
            fig.canvas.draw_idle()
    
        


    def on_key(event):
        global current_event
        if event.key == 'right':
            current_event = (current_event + 1) % len(crossings_data)
            update_plot(current_event)
        elif event.key == 'left':
            current_event = (current_event - 1) % len(crossings_data)
            update_plot(current_event)
        elif event.key == 'r':
            toggle_rejection(None)
        elif event.key == 's':
            save_filtered_events(None)

    def toggle_rejection(event):
        global current_event
        if current_event in rejected_indices:
            rejected_indices.remove(current_event)
            print(f"Un-rejected event {current_event + 1}")
        else:
            rejected_indices.add(current_event)
            print(f"Rejected event {current_event + 1}")
        update_plot(current_event)


    def save_filtered_events(event):
        # Exclude rejected events
        filtered_df = crossings_data.drop(index=list(rejected_indices)).reset_index(drop=True)
        
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save to the predefined csv_filename path
        filtered_df['Threshold'] = auto_threshold
        filtered_df.to_csv(csv_filename, index=False)  # Save directly to the pre-defined filename
        print(f"Overwritten original CSV with {len(filtered_df)} accepted events at {csv_filename}")
    



    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot(current_event)
    plt.show()

########################### Main Execution ####################################

if not os.path.exists(csv_filename):
    print(f"No CSV file found at {csv_filename}. Running detection and saving data...")
    detect_crossings(auto_threshold)
    save_data()
else:
    print(f"CSV file found at {csv_filename}, loading saved events.")

view_saved_detections(csv_filename, abf, fs)
