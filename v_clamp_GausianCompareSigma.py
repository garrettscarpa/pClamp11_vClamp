#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:15:08 2025

@author: gs075
"""
import matplotlib.pyplot as plt
import pyabf
import pyabf.filter
import numpy as np

root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data'
recording = '2025_08_08_0005'

# Load the ABF file
abf = pyabf.ABF('/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data/2025_08_08_0005.abf')

# Get the sweep count
x = abf.sweepCount

# Define sigma values for different smoothing levels
sigma_values = [0, 0.5, 1, 2, 5]

# Define the specific time range (from 29.34 to 29.48 seconds)
time_start = 29.34
time_end = 29.48

# Create a figure and a grid of subplots (one for each sigma value)
fig, axes = plt.subplots(len(sigma_values), 1, figsize=(8, 10))

# Loop over each sweep and plot in separate panels for each sigma
for i in range(x):
    abf.setSweep(i)  # Set sweep to i

    # Get the time data for this sweep
    time_data = abf.sweepX

    # Limit data to the specific time range (29.34 to 29.48 seconds)
    time_mask = (time_data >= time_start) & (time_data <= time_end)
    time_data = time_data[time_mask]
    signal_data = abf.sweepY[time_mask]

    # Apply different sigma values and plot them in separate subplots
    for j, sigma in enumerate(sigma_values):
        pyabf.filter.gaussian(abf, 0)  # Remove old filter
        pyabf.filter.gaussian(abf, sigma)  # Apply custom sigma
        abf.setSweep(i)  # Reload sweep with new filter
        axes[j].plot(time_data, signal_data, alpha=.8, label=f"sigma: {sigma:.02f}")
        axes[j].set_title(f'Sweep {i+1} - sigma: {sigma:.02f}')
        axes[j].set_xlabel('Time (Seconds)')
        axes[j].set_ylabel('Signal (mV)')
        axes[j].legend()

# Link the axes to sync zooming and panning
for ax in axes:
    ax.set_xlim([time_start, time_end])  # Set x-axis limits for the range of interest

# Adjust layout to prevent overlap between subplots
plt.tight_layout()

# Show the plots
plt.show()
