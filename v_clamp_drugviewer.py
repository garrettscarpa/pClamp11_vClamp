import matplotlib.pyplot as plt
import pyabf
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle

############################## User Inputs ####################################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data'
recording = '2025_10_01_0016'
config = 'iclamp'  # or 'lfp'
LJP_CORRECTION_MV = 0  # 14.681 for patch clamp, 0 for LFP

# Optional x-axis limits in seconds (set to None to auto-scale)
xlim_start = 670
xlim_end = 900 

############################## Load ABF file ##################################
abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

print("total sweeps =", abf.sweepCount)
abf.setSweep(0, channel=0)
fs = abf.dataRate

if config == 'iclamp':
    corrected_vm = abf.sweepY - LJP_CORRECTION_MV
else:
    corrected_vm = abf.sweepY

############################## Set treatments #################################
recording_end = abf.sweepX[-1]
drug_treatments = [
    (900, 940, 'skyblue', 'aCSF'),
#    (150, 152, 'lightgreen', 'ACh'),
    (690, 692, 'k', 'Glutamate'),
    (780, 795, 'k', 'Glutamate'), 
    (845, 850, 'lightgreen', 'ACh'),
]

############################## Bar Parameters #################################
# Define bar height and vertical offset logic
base_y = 10  # Base height above data trace
bar_height = 5
bar_spacing = 2

############################## Create Plot ####################################

fig, ax = plt.subplots(figsize=(10, 6))

# Plot membrane potential
ax.plot(abf.sweepX, corrected_vm, color='maroon', label='Membrane Potential')
ax.axhline(0, color='k', ls='--')
ax.set_xlabel(abf.sweepLabelX)
ax.set_ylabel('pA')
ax.set_title("Membrane Potential" + (" (LJP Corrected)" if config == 'iclamp' else ""))
ax.set_ylim(-100, 60)

# Set x-axis limits if provided
if xlim_start is not None and xlim_end is not None:
    ax.set_xlim(xlim_start, xlim_end)


# Track overlapping bars using y-offset indexing
bar_levels = []  # list of [start, end, level_index]

for start, end, color, label in drug_treatments:
    # Skip if outside of xlim
    if (xlim_start is not None and end < xlim_start) or \
       (xlim_end is not None and start > xlim_end):
        continue

    # Determine offset level to avoid overlap
    level = 0
    while any((max(start, s) < min(end, e) and l == level) for s, e, l in bar_levels):
        level += 1

    # Record this treatment for future overlap checks
    bar_levels.append((start, end, level))

    y_bottom = base_y + level * (bar_height + bar_spacing)

    # Draw rectangle bar
    rect = Rectangle(
        (start, y_bottom),
        end - start,
        bar_height,
        facecolor=color,
        edgecolor='k',
        alpha=0.7,
        label=label
    )
    ax.add_patch(rect)

# Add legend with unique labels only
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc='upper right')

# Format axes
ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

plt.tight_layout()
plt.show()
