import pandas as pd
import matplotlib.pyplot as plt

# Load the exported CSV
csv_path = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data/mini_timestamps/threshold_crossings_2025_09_12_0024.csv'
df = pd.read_csv(csv_path)

# Define treatment windows and labels
treatment_windows = [
    (60, 120),
    (120, 180),
    (540, 600),
]
window_labels = ['acsf', 'ach', 'hex']  # Must match number of windows

# Count events in each window
event_counts = []
for start, end in treatment_windows:
    count = df[(df['Timestamp'] >= start) & (df['Timestamp'] < end)].shape[0]
    event_counts.append(count)

# Plot
plt.figure(figsize=(8, 5))
plt.bar(window_labels, event_counts, color='maroon')
plt.xlabel('Treatment')
plt.ylabel('Event Count')
plt.title('Event Frequency per Treatment Window')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
