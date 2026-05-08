#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
from scipy.stats import ttest_ind
import pyabf
from collections import defaultdict
from itertools import combinations


# =============================================================================
# ROOT DIRECTORY
# =============================================================================

root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Analyses/RD_SCLC_TumorCells'


# =============================================================================
# SET GLOBALS
# =============================================================================

event_features = defaultdict(lambda: {
    "prominence": [],
    "width": [],
    "ttp": [],
    "rec": [],
    "base_dur": []   # right_t − left_t in ms; should equal ttp + rec
})

# =============================================================================
# PARAMETERS
# =============================================================================

analysis_start_s = 0
analysis_end_s = 60

window_ms_pre = 100
total_window_ms = 250

highpass_cutoff = 1
lowpass_cutoff = 1000

accepted_only = True

ylim = (-40, 15)


# =============================================================================
# FILTERS
# =============================================================================

def highpass(data, cutoff, fs):
    b, a = butter(4, cutoff / (0.5 * fs), btype='high')
    return filtfilt(b, a, data)


def lowpass(data, cutoff, fs):
    b, a = butter(4, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)


# =============================================================================
# CACHE
# =============================================================================

trace_cache = {}

def load_filtered_trace(recording_path, recording_file):

    key = (recording_path, recording_file)

    if key in trace_cache:
        return trace_cache[key]

    abf = pyabf.ABF(os.path.join(recording_path, recording_file))
    abf.setSweep(0)

    fs = abf.dataRate

    time = abf.sweepX
    signal = abf.sweepY

    mask = (time >= analysis_start_s) & (time <= analysis_end_s)

    time = time[mask]
    signal = signal[mask]

    signal = highpass(signal, highpass_cutoff, fs)
    signal = lowpass(signal, lowpass_cutoff, fs)

    trace_cache[key] = (time, signal, fs)
    return time, signal, fs


# =============================================================================
# FEATURES
# =============================================================================

def extract_features(wf, t, left_base_val=None, right_base_val=None, right_t_ms=None):
    """
    Extract event features from a baseline-subtracted waveform.
    Peak is searched only within [0, right_t_ms] — i.e. between left and right base.
    """

    # ------------------------------------------------------------------
    # 0. CONSTRAIN PEAK SEARCH to the event interval [left_base, right_base]
    #    t=0 is the left base; right_t_ms is the right base in ms.
    #    Only search for the peak within this interval.
    # ------------------------------------------------------------------
    if right_t_ms is not None and right_t_ms > 0:
        event_mask = (t >= 0) & (t <= right_t_ms)
    else:
        event_mask = (t >= 0)   # fallback: at least don't search pre-window

    if not np.any(event_mask):
        # Edge case: no samples in range — fall back to full waveform
        event_mask = np.ones(len(t), dtype=bool)

    # argmin within the masked region, then map back to full-array index
    peak_idx_local = np.argmin(wf[event_mask])
    peak_idx = np.where(event_mask)[0][peak_idx_local]
    peak_val = wf[peak_idx]

    # ------------------------------------------------------------------
    # 1. TRUE PROMINENCE
    # ------------------------------------------------------------------
    if left_base_val is not None and right_base_val is not None:
        base_level = (left_base_val + right_base_val) / 2.0
    else:
        base_level = 0.0

    prominence = np.abs(peak_val - base_level)

    # ------------------------------------------------------------------
    # 2. WIDTH AT HALF-MAXIMUM  (searched only within event interval)
    # ------------------------------------------------------------------
    half_max = peak_val / 2.0
    below = wf <= half_max

    # Only consider crossings within the event interval
    below_masked = below & event_mask

    if np.any(below_masked):
        indices = np.where(below_masked)[0]
        left_cross  = indices[0]
        right_cross = indices[-1]
        width = t[right_cross] - t[left_cross]
    else:
        width = np.nan

    # ------------------------------------------------------------------
    # 3. TIME TO PEAK  (left-base → peak; always within [0, right_t_ms])
    # ------------------------------------------------------------------
    ttp = t[peak_idx] - 0.0
    ttp = max(ttp, 0.0)

    # ------------------------------------------------------------------
    # 4. RECOVERY  (peak → right-base)
    # ------------------------------------------------------------------
    if right_t_ms is not None:
        recovery = max(right_t_ms - t[peak_idx], 0.0)
    else:
        recovery = np.nan

    return prominence, width, ttp, recovery

# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def sem(vals):
    """Standard error of the mean, ignoring NaNs."""
    arr = np.array([v for v in vals if not np.isnan(v)])
    if len(arr) < 2:
        return 0.0
    return np.nanstd(arr, ddof=1) / np.sqrt(len(arr))


def pvalue_stars(p):
    """Convert p-value to significance star string."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# =============================================================================
# LOAD EVENTS
# =============================================================================
condition_recordings = defaultdict(set)
condition_waveforms = defaultdict(list)
recording_features = []

event_files = [
    os.path.join(subdir, "event_status.npy")
    for subdir, _, files in os.walk(root)
    if "event_status.npy" in files
]

print(f"\nFound {len(event_files)} saved event files")


for event_file in event_files:

    try:
        saved = np.load(event_file, allow_pickle=True).item()
    except Exception:
        continue

    events = saved.get("events", [])
    if len(events) == 0:
        continue

    condition = os.path.basename(os.path.dirname(event_file))

    if accepted_only:
        events = [e for e in events if e.get("status", True)]

    rec_groups = defaultdict(list)

    for e in events:
        rec_groups[(e["recording_path"], e["file"])].append(e)

    for (rec_path, rec_file), rec_events in rec_groups.items():

        time, signal, fs = load_filtered_trace(rec_path, rec_file)

        pre_samples   = int(window_ms_pre   / 1000 * fs)
        total_samples = int(total_window_ms / 1000 * fs)

        waveforms = []

        for e in rec_events:

            if e.get("base") is None:
                continue

            # base = [left_t, right_t]
            left_t  = e["base"][0]
            right_t = e["base"][1]

            left_idx  = np.argmin(np.abs(time - left_t))
            right_idx = np.argmin(np.abs(time - right_t))

            # Raw signal values at the base markers (for true prominence)
            left_base_val  = float(signal[left_idx])  if left_idx  < len(signal) else None
            right_base_val = float(signal[right_idx]) if right_idx < len(signal) else None

            start = left_idx - pre_samples
            end   = start + total_samples

            if start < 0 or end >= len(signal):
                continue

            wf = signal[start:end].copy()
            wf -= np.mean(wf[:pre_samples])   # baseline subtract

            # Time axis: 0 ms at the left base (index = pre_samples)
            t_wf = np.arange(wf.size)
            t_wf = (t_wf - pre_samples) / fs * 1000

            # right base position in the same ms units as t_wf
            # (right_t and left_t are in seconds; difference → ms)
            right_t_ms = (right_t - left_t) * 1000

            prom, width, ttp, rec = extract_features(
                wf, t_wf,
                left_base_val  = left_base_val,
                right_base_val = right_base_val,
                right_t_ms     = right_t_ms
            )

            event_features[condition]["prominence"].append(prom)
            event_features[condition]["width"].append(width)
            event_features[condition]["ttp"].append(ttp)
            event_features[condition]["rec"].append(rec)
            event_features[condition]["base_dur"].append(right_t_ms)

            waveforms.append(wf)
            
            
            
            # ── DEBUG: verify TTP + Rec == base_dur ──────────────────────────
            sum_ttp_rec = ttp + rec
            residual_debug = right_t_ms - sum_ttp_rec

            if abs(residual_debug) > 0.5:   # flag anything > 0.5 ms off
                print(
                    f"[RESIDUAL DEBUG] {condition} | {rec_file}\n"
                    f"  left_t        = {left_t:.6f} s\n"
                    f"  right_t       = {right_t:.6f} s\n"
                    f"  right_t_ms    = {right_t_ms:.4f} ms   (right_t - left_t)*1000\n"
                    f"  pre_samples   = {pre_samples}  ({window_ms_pre} ms @ {fs} Hz)\n"
                    f"  left_idx      = {left_idx}  → time[left_idx]  = {time[left_idx]:.6f} s\n"
                    f"  right_idx     = {right_idx} → time[right_idx] = {time[right_idx]:.6f} s\n"
                    f"  peak_idx (in wf) = {np.argmin(wf)}\n"
                    f"  t_wf[peak]    = {t_wf[np.argmin(wf)]:.4f} ms\n"
                    f"  TTP           = {ttp:.4f} ms\n"
                    f"  Recovery      = {rec:.4f} ms\n"
                    f"  TTP + Rec     = {sum_ttp_rec:.4f} ms\n"
                    f"  base_dur      = {right_t_ms:.4f} ms\n"
                    f"  residual      = {residual_debug:.4f} ms  ← should be ~0\n"
                )
            # ── END DEBUG ─────────────────────────────────────────────────────
            
            

        if len(waveforms) == 0:
            continue

        waveforms = np.array(waveforms)

        duration_s = analysis_end_s - analysis_start_s
        recording_features.append({
            "condition": condition,
            "frequency": len(waveforms) / duration_s
        })

        condition_waveforms[condition].extend(waveforms)
        condition_recordings[condition].add((rec_path, rec_file))


# =============================================================================
# SUMMARY  (means ± SEM per condition)
# =============================================================================

raw_frequency = defaultdict(list)
for r in recording_features:
    raw_frequency[r["condition"]].append(r["frequency"])

conditions = sorted(
    set(event_features.keys()) | set(raw_frequency.keys())
)
condition_N = {cond: len(condition_recordings[cond]) for cond in conditions}

if len(conditions) == 0:
    raise SystemExit("No valid conditions found")

feature_summary = {k: [] for k in ("condition", "frequency", "frequency_sem",
                                    "prominence", "prominence_sem",
                                    "width",      "width_sem",
                                    "ttp",        "ttp_sem",
                                    "rec",        "rec_sem")}

for cond in conditions:
    feature_summary["condition"].append(cond)

    freq_vals = raw_frequency.get(cond, [])
    feature_summary["frequency"].append(np.nanmean(freq_vals) if freq_vals else np.nan)
    feature_summary["frequency_sem"].append(sem(freq_vals))

    for key in ("prominence", "width", "ttp", "rec"):
        vals = event_features[cond][key]
        feature_summary[key].append(np.nanmean(vals) if vals else np.nan)
        feature_summary[f"{key}_sem"].append(sem(vals))

raw_event_features = event_features   # alias for plotting


# =============================================================================
# PLOTTING STYLE  (clean, publication-ready)
# =============================================================================

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "figure.dpi":         150,
    "pdf.fonttype":       42,   # editable text in Illustrator
    "svg.fonttype":       "none",
})

# Color palette – one distinct color per condition
_palette = ["#1d1d1d", "#7B2CBF", "#9D4EDD", "#5A189A", "#C77DFF",
            "#2196F3", "#FF5722", "#00897B"]
colors = _palette[:len(conditions)]

# ------------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------------
n_wf_rows = len(conditions)
fig = plt.figure(figsize=(14, 3.2 + 2.8 * n_wf_rows))

outer = gridspec.GridSpec(
    2, 1,
    height_ratios=[2.8, 2.8 * n_wf_rows],
    hspace=0.55,
    figure=fig
)

top_gs    = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.45)
bottom_gs = gridspec.GridSpecFromSubplotSpec(n_wf_rows, 1,
                                             subplot_spec=outer[1], hspace=0.55)


# ------------------------------------------------------------------
# Helper: draw significance brackets between bar pairs
# ------------------------------------------------------------------
def add_significance(ax, x1, x2, y_top, label, dy_frac=0.06):
    """Draw a bracket + label above two bars."""
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_range * dy_frac
    bar_tip = y_top + h * 0.3
    ax.plot([x1, x1, x2, x2],
            [bar_tip, bar_tip + h, bar_tip + h, bar_tip],
            lw=0.8, color="k")
    ax.text((x1 + x2) / 2, bar_tip + h * 1.1, label,
            ha="center", va="bottom", fontsize=7)


# ------------------------------------------------------------------
# Feature bar plots (row 0)
# ------------------------------------------------------------------
keys   = ["frequency",  "prominence",  "width",  "ttp",    "rec"]
labels = ["Frequency (Hz)", "Prominence (pA)", "Half-width (ms)",
          "Time-to-peak (ms)", "Recovery (ms)"]
sem_keys = [k + "_sem" for k in keys]

x = np.arange(len(conditions))
rng = np.random.default_rng(0)

for col_i, (key, label, sem_key) in enumerate(zip(keys, labels, sem_keys)):

    ax = fig.add_subplot(top_gs[col_i])

    means = np.array(feature_summary[key],     dtype=float)
    sems  = np.array(feature_summary[sem_key], dtype=float)

    # --- bars ---
    bars = ax.bar(x, means, color=colors, alpha=0.85,
                  width=0.55, zorder=2,
                  linewidth=0.6, edgecolor="white")

    # --- error bars ---
    ax.errorbar(x, means, yerr=sems,
                fmt="none", color="black",
                capsize=3, capthick=0.8,
                elinewidth=0.8, zorder=3)

    # --- jittered scatter ---
    for xi, cond, color in zip(x, conditions, colors):
        if key == "frequency":
            vals = [v for v in raw_frequency.get(cond, []) if not np.isnan(v)]
        else:
            vals = [v for v in raw_event_features[cond][key] if not np.isnan(v)]

        if len(vals) == 0:
            continue

        jitter = rng.normal(0, 0.07, size=len(vals))
        ax.scatter(
            np.full(len(vals), xi) + jitter,
            vals,
            color="white",
            edgecolors=color,
            linewidths=0.6,
            s=14,
            alpha=0.75,
            zorder=4
        )

    # --- pairwise t-tests (all pairs of conditions) ---
    if len(conditions) >= 2:
        # Collect per-condition data arrays
        data_by_cond = []
        for cond in conditions:
            if key == "frequency":
                d = [v for v in raw_frequency.get(cond, []) if not np.isnan(v)]
            else:
                d = [v for v in raw_event_features[cond][key] if not np.isnan(v)]
            data_by_cond.append(d)

        pairs = list(combinations(range(len(conditions)), 2))
        # Compute auto-positioned brackets
        y_max = np.nanmax(means + sems)
        bracket_step = (ax.get_ylim()[1] - y_max) / max(len(pairs), 1)
        # recompute after knowing data range
        data_max = np.nanmax([np.nanmax(d) if len(d) else 0
                              for d in data_by_cond])
        bracket_base = data_max * 1.08
        bracket_gap  = data_max * 0.12 if data_max != 0 else 0.12

        for pair_i, (i, j) in enumerate(pairs):
            d1, d2 = data_by_cond[i], data_by_cond[j]
            if len(d1) < 2 or len(d2) < 2:
                continue
            _, p = ttest_ind(d1, d2, equal_var=False)
            stars = pvalue_stars(p)

            y_bracket = bracket_base + pair_i * bracket_gap
            h = bracket_gap * 0.35
            ax.plot([i, i, j, j],
                    [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
                    lw=0.7, color="#444444")
            ax.text((i + j) / 2, y_bracket + h * 1.1, stars,
                    ha="center", va="bottom", fontsize=6.5, color="#222222")

    ax.set_title(label, fontsize=8.5, fontweight="bold", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=7.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    ax.tick_params(length=3)
    ax.set_ylabel("")

# ------------------------------------------------------------------
# Section headers with underlines
# ------------------------------------------------------------------

# Grab the 5 bar-plot axes (added in order: frequency, prominence, width, ttp, rec)
top_axes = [fig.axes[i] for i in range(5)]

def ax_pos(ax):
    return ax.get_position()

bb_freq  = ax_pos(top_axes[0])          # frequency (Per-Recording)
bb_left  = ax_pos(top_axes[1])          # prominence (leftmost Per-Event)
bb_right = ax_pos(top_axes[4])          # recovery   (rightmost Per-Event)

header_y      = bb_freq.y1 + 0.045      # just above the bar axes
line_y_offset = 0.005                  # gap between text bottom and line

# --- Per-Recording ---
x_freq_mid = (bb_freq.x0 + bb_freq.x1) / 2
fig.text(x_freq_mid, header_y, "Per-Recording",
         ha="center", va="bottom", fontsize=10, fontweight="bold",
         transform=fig.transFigure, color="#111111")

line_y = header_y - line_y_offset
fig.add_artist(Line2D(
    [bb_freq.x0, bb_freq.x1], [line_y, line_y],
    transform=fig.transFigure, color="black", linewidth=1.2, clip_on=False
))

# --- Per-Event ---
x_event_mid = (bb_left.x0 + bb_right.x1) / 2
fig.text(x_event_mid, header_y, "Per-Event",
         ha="center", va="bottom", fontsize=10, fontweight="bold",
         transform=fig.transFigure, color="#111111")

fig.add_artist(Line2D(
    [bb_left.x0, bb_right.x1], [line_y, line_y],
    transform=fig.transFigure, color="black", linewidth=1.2, clip_on=False
))
# ------------------------------------------------------------------
# Waveform plots (rows 1…N)
# ------------------------------------------------------------------
# Recompute a shared time axis for labelling (uses last fs from loop above)
# (pre_samples / fs are still in scope from the data-loading loop)

for row_idx, cond in enumerate(conditions):

    ax = fig.add_subplot(bottom_gs[row_idx])

    wfs = np.array(condition_waveforms[cond])
    n_events = len(wfs)

    t = (np.arange(wfs.shape[1]) - pre_samples) / fs * 1000

    color = colors[row_idx]

    # individual traces
    for wf in wfs:
        ax.plot(t, wf, color=color, alpha=max(0.03, min(0.12, 3 / n_events)),
                lw=0.6, rasterized=True)

    # mean ± SEM
    mean_wf = np.mean(wfs, axis=0)
    sem_wf  = np.std(wfs, axis=0) / np.sqrt(n_events)

    ax.fill_between(t, mean_wf - sem_wf, mean_wf + sem_wf,
                    color=color, alpha=0.18, zorder=2)
    ax.plot(t, mean_wf, color=color, lw=2.2, zorder=3)

    # left-base marker
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)

    ax.set_title(
        f"{cond}   (n = {n_events} events,  N = {condition_N[cond]} recordings)",
        fontsize=8.5, fontweight="bold", color="#222222", pad=4
    )
    ax.set_ylim(ylim)
    ax.set_ylabel("Current (pA)", fontsize=8.5)

    if row_idx < n_wf_rows - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time from left base (ms)", fontsize=8.5)

    ax.tick_params(length=3)

# ------------------------------------------------------------------
# Save / show
# ------------------------------------------------------------------
fig.patch.set_facecolor("white")

plt.show()
print("Done.")

# =============================================================================
# CONFIRMATION FIGURE
# Verifies that TTP + Recovery ≈ base-to-base duration for every event.
#
# For each condition, three values are plotted side-by-side per event:
#   • TTP + Rec   (sum of the two annotated segments)
#   • base_dur    (right_t − left_t, computed independently from timestamps)
#   • residual    (base_dur − (TTP + Rec); should be ≈ 0)
#
# If the arithmetic is correct, the first two columns will be identical
# and the third will scatter tightly around zero.
# =============================================================================

fig2, axes2 = plt.subplots(
    1, len(conditions),
    figsize=(3.5 * len(conditions), 4),
    sharey=False
)

if len(conditions) == 1:
    axes2 = [axes2]

for ax, cond, color in zip(axes2, conditions, colors):

    ttp_vals  = np.array(raw_event_features[cond]["ttp"],      dtype=float)
    rec_vals  = np.array(raw_event_features[cond]["rec"],      dtype=float)
    base_vals = np.array(raw_event_features[cond]["base_dur"], dtype=float)

    # drop events where any value is NaN
    valid = ~(np.isnan(ttp_vals) | np.isnan(rec_vals) | np.isnan(base_vals))
    ttp_v  = ttp_vals[valid]
    rec_v  = rec_vals[valid]
    base_v = base_vals[valid]

    summed   = ttp_v + rec_v
    residual = base_v - summed          # should be ≈ 0 everywhere

    x_pos  = np.array([0, 1, 2])
    x_labs = ["TTP + Rec", "Base dur.", "Residual\n(base \u2212 sum)"]
    data   = [summed, base_v, residual]

    for xi, vals_i in zip(x_pos[:2], data[:2]):
        m = np.nanmean(vals_i)
        s = np.nanstd(vals_i, ddof=1) / np.sqrt(len(vals_i))
        ax.bar(xi, m, color=color, alpha=0.8, width=0.55,
               edgecolor="white", linewidth=0.6, zorder=2)
        ax.errorbar(xi, m, yerr=s, fmt="none", color="black",
                    capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)
        jitter = np.random.default_rng(xi).normal(0, 0.06, size=len(vals_i))
        ax.scatter(np.full(len(vals_i), xi) + jitter, vals_i,
                   color="white", edgecolors=color, linewidths=0.6,
                   s=14, alpha=0.75, zorder=4)

    # residual: zero-centred, show as scatter + mean line
    jitter = np.random.default_rng(99).normal(0, 0.06, size=len(residual))
    ax.scatter(np.full(len(residual), 2) + jitter, residual,
               color=color, s=14, alpha=0.6, linewidths=0, zorder=4)
    ax.axhline(np.nanmean(residual), color="black", lw=1.2,
               xmin=2/3 + 0.02, xmax=1.0 - 0.02, zorder=5)
    ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.6, zorder=3,
               xmin=2/3, xmax=1.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labs, fontsize=8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_title(cond, fontsize=9, fontweight="bold", pad=4)
    ax.set_ylabel("Time (ms)", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)

    # annotate mean residual
    ylo, yhi = ax.get_ylim()
    ax.text(2, ylo + (yhi - ylo) * 0.05,
            f"mean = {np.nanmean(residual):.3f} ms",
            ha="center", va="bottom", fontsize=7, color="#444444")

fig2.suptitle("Confirmation: TTP + Recovery vs. Base-to-Base Duration",
              fontsize=10, fontweight="bold", y=1.01)
fig2.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig(os.path.join(_script_dir, "mPSC_confirmation.pdf"),
            bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(_script_dir, "mPSC_confirmation.png"),
            bbox_inches="tight", dpi=300)
plt.show()