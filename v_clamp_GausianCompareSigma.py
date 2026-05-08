#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pyabf
import pyabf.filter
import numpy as np

# Load the ABF file
abf = pyabf.ABF('/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data/2026_04_23_0021.abf')

# Number of sweeps
x = abf.sweepCount

# Smoothing levels
sigma_values = [0, 0.5, 1, 2, 5]

# Time window
time_start = 9.5
time_end = 10

# Create subplots
fig, axes = plt.subplots(len(sigma_values), 1, figsize=(10, 12), sharex=True)

# Loop through sigma values FIRST
for j, sigma in enumerate(sigma_values):

    for i in range(x):

        # Remove previous filter
        pyabf.filter.gaussian(abf, 0)

        # Apply current filter
        pyabf.filter.gaussian(abf, sigma)

        # Load sweep AFTER filtering
        abf.setSweep(i)

        # Extract filtered data
        time_data = abf.sweepX
        signal_data = abf.sweepY

        # Restrict time range
        mask = (time_data >= time_start) & (time_data <= time_end)

        axes[j].plot(
            time_data[mask],
            signal_data[mask],
            alpha=0.8
        )

    axes[j].set_title(f'Gaussian sigma = {sigma}')
    axes[j].set_ylabel('Signal (mV)')

axes[-1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()