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

root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/Ephys/Patch and LFP/Analyses/Patch/RD_SCLC_TumorCells_MonoCultur_May27_2026_Update'


# =============================================================================
# SET GLOBALS
# =============================================================================

event_features = defaultdict(lambda: {
    "prominence": [],
    "width": [],
    "ttp": [],
    "rec": [],
    "base_dur": [],
    "auc": []
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

halfwidth_threshold_ms = 3.0   # events with width <= threshold → "narrow", else "wide"

ylim = (-40, 15)

accepted_only = True
split_by_halfwidth = False
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
    # ------------------------------------------------------------------
    if right_t_ms is not None and right_t_ms > 0:
        event_mask = (t >= 0) & (t <= right_t_ms)
    else:
        event_mask = (t >= 0)

    if not np.any(event_mask):
        event_mask = np.ones(len(t), dtype=bool)

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
    # 2. WIDTH AT HALF-MAXIMUM
    # ------------------------------------------------------------------
    half_max = peak_val / 2.0
    below = wf <= half_max
    below_masked = below & event_mask

    if np.any(below_masked):
        indices = np.where(below_masked)[0]
        left_cross  = indices[0]
        right_cross = indices[-1]
        width = t[right_cross] - t[left_cross]
    else:
        width = np.nan

    # ------------------------------------------------------------------
    # 3. TIME TO PEAK
    # ------------------------------------------------------------------
    ttp = t[peak_idx] - 0.0
    ttp = max(ttp, 0.0)

    # ------------------------------------------------------------------
    # 4. RECOVERY
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
            right_t_ms = (right_t - left_t) * 1000

            prom, width, ttp, rec = extract_features(
                wf, t_wf,
                left_base_val  = left_base_val,
                right_base_val = right_base_val,
                right_t_ms     = right_t_ms
            )

            auc = e.get("auc", np.nan)
            if auc is None:
                auc = np.nan

            # ── Assign condition label by half-width population ──
            if split_by_halfwidth:
                if np.isnan(width):
                    continue          # skip events that can't be classified
                pop = "narrow" if width <= halfwidth_threshold_ms else "wide"
                cond_label = f"{condition}_{pop}"
            else:
                cond_label = condition

            event_features[cond_label]["prominence"].append(prom)
            event_features[cond_label]["width"].append(width)
            event_features[cond_label]["ttp"].append(ttp)
            event_features[cond_label]["rec"].append(rec)
            event_features[cond_label]["base_dur"].append(right_t_ms)
            event_features[cond_label]["auc"].append(auc)

            waveforms.append((cond_label, wf))

            # ── DEBUG: verify TTP + Rec == base_dur ──────────────────────────
            sum_ttp_rec = ttp + rec
            residual_debug = right_t_ms - sum_ttp_rec

            if abs(residual_debug) > 0.5:
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

        # Group this recording's waveforms by population label
        wf_by_label = defaultdict(list)
        for lbl, wf in waveforms:
            wf_by_label[lbl].append(wf)

        duration_s = analysis_end_s - analysis_start_s

        for lbl, wf_list in wf_by_label.items():
            recording_features.append({
                "condition": lbl,
                "frequency": len(wf_list) / duration_s
            })
            condition_waveforms[lbl].extend(wf_list)
            condition_recordings[lbl].add((rec_path, rec_file))


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
                                    "rec",        "rec_sem",
                                    "auc",        "auc_sem")}

for cond in conditions:
    feature_summary["condition"].append(cond)

    freq_vals = raw_frequency.get(cond, [])
    feature_summary["frequency"].append(np.nanmean(freq_vals) if freq_vals else np.nan)
    feature_summary["frequency_sem"].append(sem(freq_vals))

    for key in ("prominence", "width", "ttp", "rec", "auc"):
        vals = [v for v in event_features[cond][key] if v is not None]
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
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

_palette = ["#1d1d1d", "#7B2CBF", "#9D4EDD", "#5A189A", "#C77DFF",
            "#2196F3", "#FF5722", "#00897B"]


# =============================================================================
# MAIN PLOTTING FUNCTION  (one full figure set per condition group)
# =============================================================================

def make_figures(group_conditions, group_name=""):
    """
    Build the feature/waveform figure and the confirmation figure for a
    given subset of conditions.
    """

    if len(group_conditions) == 0:
        return

    colors = _palette[:len(group_conditions)]

    # ------------------------------------------------------------------
    # Figure layout: 6 columns (1 per-recording + 5 per-event)
    # ------------------------------------------------------------------
    n_wf_rows = len(group_conditions)
    fig = plt.figure(figsize=(16, 3.2 + 2.8 * n_wf_rows))

    if group_name:
        fig.suptitle(group_name, fontsize=13, fontweight="bold", y=0.995)

    outer = gridspec.GridSpec(
        2, 1,
        height_ratios=[2.8, 2.8 * n_wf_rows],
        hspace=0.55,
        figure=fig
    )

    top_gs    = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.45)
    bottom_gs = gridspec.GridSpecFromSubplotSpec(n_wf_rows, 1,
                                                 subplot_spec=outer[1], hspace=0.55)

    # ------------------------------------------------------------------
    # Feature bar plots — 6 panels
    # col 0:     frequency   (per-recording)
    # cols 1–5:  prominence, width, ttp, rec, auc   (per-event)
    # ------------------------------------------------------------------
    keys     = ["frequency",     "prominence",     "width",          "ttp",              "rec",          "auc"]
    labels   = ["Frequency\n(Hz)", "Prominence\n(pA)", "Half-width\n(ms)",
                "Time-to-peak\n(ms)", "Recovery\n(ms)", "AUC\n(pA·ms)"]

    x   = np.arange(len(group_conditions))
    rng = np.random.default_rng(0)

    for col_i, (key, label) in enumerate(zip(keys, labels)):

        ax = fig.add_subplot(top_gs[col_i])

        means = np.array([feature_lookup[cond][key]            for cond in group_conditions], dtype=float)
        sems  = np.array([feature_lookup[cond][f"{key}_sem"]   for cond in group_conditions], dtype=float)

        ax.bar(x, means, color=colors, alpha=0.85,
               width=0.55, zorder=2, linewidth=0.6, edgecolor="white")

        ax.errorbar(x, means, yerr=sems,
                    fmt="none", color="black",
                    capsize=3, capthick=0.8, elinewidth=0.8, zorder=3)

        for xi, cond, color in zip(x, group_conditions, colors):
            if key == "frequency":
                vals = [v for v in raw_frequency.get(cond, []) if not np.isnan(v)]
            else:
                vals = [v for v in raw_event_features[cond][key]
                        if v is not None and not np.isnan(v)]

            if len(vals) == 0:
                continue

            jitter = rng.normal(0, 0.07, size=len(vals))
            ax.scatter(
                np.full(len(vals), xi) + jitter, vals,
                color="white", edgecolors=color,
                linewidths=0.6, s=14, alpha=0.75, zorder=4
            )

        if len(group_conditions) >= 2:
            data_by_cond = []
            for cond in group_conditions:
                if key == "frequency":
                    d = [v for v in raw_frequency.get(cond, []) if not np.isnan(v)]
                else:
                    d = [v for v in raw_event_features[cond][key]
                         if v is not None and not np.isnan(v)]
                data_by_cond.append(d)

            pairs    = list(combinations(range(len(group_conditions)), 2))
            data_max = np.nanmax([np.nanmax(d) if len(d) else 0 for d in data_by_cond])
            bracket_base = data_max * 1.08
            bracket_gap  = data_max * 0.12 if data_max != 0 else 0.12

            for pair_i, (i, j) in enumerate(pairs):
                d1, d2 = data_by_cond[i], data_by_cond[j]
                if len(d1) < 2 or len(d2) < 2:
                    continue
                _, p  = ttest_ind(d1, d2, equal_var=False)
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
        ax.set_xticklabels(group_conditions, rotation=35, ha="right", fontsize=7.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        ax.tick_params(length=3)
        ax.set_ylabel("")

    # ------------------------------------------------------------------
    # Section headers with underlines
    # ------------------------------------------------------------------
    top_axes = fig.axes[:6]

    bb_freq  = top_axes[0].get_position()   # frequency  → Per-Recording
    bb_left  = top_axes[1].get_position()   # prominence → leftmost Per-Event
    bb_right = top_axes[5].get_position()   # auc        → rightmost Per-Event

    header_y = bb_freq.y1 + 0.055
    line_y   = header_y - 0.005

    fig.text((bb_freq.x0 + bb_freq.x1) / 2, header_y, "Per-Recording",
             ha="center", va="bottom", fontsize=10, fontweight="bold",
             transform=fig.transFigure, color="#111111")
    fig.add_artist(Line2D(
        [bb_freq.x0, bb_freq.x1], [line_y, line_y],
        transform=fig.transFigure, color="black", linewidth=1.2, clip_on=False
    ))

    fig.text((bb_left.x0 + bb_right.x1) / 2, header_y, "Per-Event",
             ha="center", va="bottom", fontsize=10, fontweight="bold",
             transform=fig.transFigure, color="#111111")
    fig.add_artist(Line2D(
        [bb_left.x0, bb_right.x1], [line_y, line_y],
        transform=fig.transFigure, color="black", linewidth=1.2, clip_on=False
    ))

    # ------------------------------------------------------------------
    # Waveform plots
    # ------------------------------------------------------------------
    for row_idx, cond in enumerate(group_conditions):

        ax = fig.add_subplot(bottom_gs[row_idx])

        wfs      = np.array(condition_waveforms[cond])
        n_events = len(wfs)

        if n_events == 0:
            ax.set_title(f"{cond}   (no events)", fontsize=8.5,
                         fontweight="bold", color="#222222", pad=4)
            ax.set_ylim(ylim)
            continue

        t     = (np.arange(wfs.shape[1]) - pre_samples) / fs * 1000
        color = colors[row_idx]

        for wf in wfs:
            ax.plot(t, wf, color=color,
                    alpha=max(0.03, min(0.12, 3 / n_events)),
                    lw=0.6, rasterized=True)

        mean_wf = np.mean(wfs, axis=0)
        sem_wf  = np.std(wfs, axis=0) / np.sqrt(n_events)

        ax.fill_between(t, mean_wf - sem_wf, mean_wf + sem_wf,
                        color=color, alpha=0.18, zorder=2)
        ax.plot(t, mean_wf, color=color, lw=2.2, zorder=3)

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

    fig.patch.set_facecolor("white")
    plt.show()

    # ------------------------------------------------------------------
    # CONFIRMATION FIGURE
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(
        1, len(group_conditions),
        figsize=(3.5 * len(group_conditions), 4),
        sharey=False
    )

    if len(group_conditions) == 1:
        axes2 = [axes2]

    for ax, cond, color in zip(axes2, group_conditions, colors):

        ttp_vals  = np.array(raw_event_features[cond]["ttp"],      dtype=float)
        rec_vals  = np.array(raw_event_features[cond]["rec"],      dtype=float)
        base_vals = np.array(raw_event_features[cond]["base_dur"], dtype=float)

        valid  = ~(np.isnan(ttp_vals) | np.isnan(rec_vals) | np.isnan(base_vals))
        ttp_v  = ttp_vals[valid]
        rec_v  = rec_vals[valid]
        base_v = base_vals[valid]

        if len(base_v) == 0:
            ax.set_title(f"{cond} (no events)", fontsize=9, fontweight="bold")
            continue

        summed   = ttp_v + rec_v
        residual = base_v - summed

        x_pos  = np.array([0, 1, 2])
        x_labs = ["Onset + Offsert", "Base dur.", "Residual\n(base \u2212 sum)"]

        for xi, vals_i in zip(x_pos[:2], [summed, base_v]):
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

        ylo, yhi = ax.get_ylim()
        ax.text(2, ylo + (yhi - ylo) * 0.05,
                f"mean = {np.nanmean(residual):.3f} ms",
                ha="center", va="bottom", fontsize=7, color="#444444")

    title_suffix = f"  —  {group_name}" if group_name else ""
    fig2.suptitle("Confirmation: TTP + Recovery vs. Base-to-Base Duration" + title_suffix,
                  fontsize=10, fontweight="bold", y=1.01)
    fig2.patch.set_facecolor("white")
    plt.tight_layout()
    plt.show()


# =============================================================================
# BUILD A LOOKUP FROM THE SUMMARY  (so make_figures can index by condition)
# =============================================================================

feature_lookup = {}
for i, cond in enumerate(feature_summary["condition"]):
    feature_lookup[cond] = {
        k: feature_summary[k][i]
        for k in feature_summary if k != "condition"
    }


# =============================================================================
# SPLIT CONDITIONS INTO GROUPS AND PLOT SEPARATELY
# =============================================================================

if split_by_halfwidth:
    narrow_conditions = sorted(c for c in conditions if c.endswith("_narrow"))
    wide_conditions   = sorted(c for c in conditions if c.endswith("_wide"))
    other_conditions  = sorted(c for c in conditions
                               if not (c.endswith("_narrow") or c.endswith("_wide")))

    make_figures(narrow_conditions, group_name="Narrow events")
    make_figures(wide_conditions,   group_name="Wide events")

    # Any conditions that never got split (e.g. NaN-width fallbacks)
    if other_conditions:
        make_figures(other_conditions, group_name="Unsplit events")
else:
    make_figures(conditions)

print("Done.")