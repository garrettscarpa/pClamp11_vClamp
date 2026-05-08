#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:39:32 2026

@author: gs075
"""
from scipy.signal import butter, filtfilt
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyabf
import sys
import os


# Load data
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Analyses/RD_SCLC_TumorCells'


# Detection metrics
min_amp = 5      # pA (your existing threshold)
max_amp = 20    # pA (set based on your dataset; adjust as needed)
threshold_sigma = 5

# Temporal filtering
analysis_start_s = 0   # e.g. 10   (seconds) # Set to None to use full recording
analysis_end_s   = 60  # e.g. 60   (seconds) # Set to None to use full recording
window_samples_view_ms = 200
ylim = (-40, 20)


recordings = []
for subdir, _, files in os.walk(root):
    for f in files:
        if f.endswith(".abf"):
            recordings.append((subdir, f))
print(f"Found {len(recordings)} recordings")


# Apply hpf and lpf and validate
def highpass(data, cutoff, fs):
    b, a = butter(4, cutoff/(0.5*fs), btype='high')
    return filtfilt(b, a, data)

def lowpass(data, cutoff, fs):
    b, a = butter(4, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, data)


events = []

def process_recording(rec_dir, rec_file):

    abf = pyabf.ABF(os.path.join(rec_dir, rec_file))
    abf.setSweep(0)

    fs = abf.dataRate
    time = abf.sweepX
    signal = abf.sweepY

    # -----------------------------------
    # ANALYSIS WINDOW
    # -----------------------------------

    start_s = analysis_start_s if analysis_start_s is not None else time[0]
    end_s = analysis_end_s if analysis_end_s is not None else time[-1]

    analysis_mask = (time >= start_s) & (time <= end_s)

    time_f = time[analysis_mask]
    signal_f = signal[analysis_mask]

    # -----------------------------------
    # FILTERING
    # -----------------------------------

    filtered = highpass(signal_f, 1, fs)
    filtered = lowpass(filtered, 1000, fs)

    # -----------------------------------
    # THRESHOLD
    # -----------------------------------

    mad = np.median(np.abs(filtered - np.median(filtered)))
    noise = mad / 0.6745

    threshold = -threshold_sigma * noise

    # -----------------------------------
    # LOAD PREVIOUS SESSION
    # -----------------------------------

    save_file = os.path.join(rec_dir, "event_status.npy")

    saved_events = None

    if os.path.exists(save_file):

        try:
            saved = np.load(save_file, allow_pickle=True).item()
            saved_events = saved.get("events", None)

            print(f"Loaded previous session: {rec_file}")

        except Exception as e:

            print(f"Could not load previous session for {rec_file}")
            print(e)

            saved_events = None

    # -----------------------------------
    # DETECT CROSSINGS
    # -----------------------------------

    crossings = []

    for i in range(1, len(filtered)):

        if filtered[i - 1] > threshold and filtered[i] <= threshold:
            crossings.append(time_f[i])

    # -----------------------------------
    # MIN EVENT DURATION FILTER
    # -----------------------------------

    min_event_ms = 0
    min_event_samples = int(min_event_ms / 1000 * fs)

    filtered_crossings = []

    for ct in crossings:

        idx = np.argmin(np.abs(time_f - ct))

        search_win = int(2e-3 * fs)

        start = max(0, idx - search_win)
        end = min(len(filtered), idx + search_win)

        segment = filtered[start:end]

        below = segment <= threshold

        max_run = 0
        run = 0

        for b in below:

            if b:
                run += 1
                max_run = max(max_run, run)

            else:
                run = 0

        if max_run >= min_event_samples:
            filtered_crossings.append(ct)

    # -----------------------------------
    # EXTRACT EVENTS
    # -----------------------------------

    window_ms = 15
    window_samples = int(window_ms / 1000 * fs)

    for ct in filtered_crossings:

        idx = np.argmin(np.abs(time_f - ct))

        start = max(0, idx - int(2e-3 * fs))
        end = start + window_samples

        if end >= len(filtered):
            continue

        segment = filtered[start:end]

        baseline = np.median(segment[:5])
        peak = np.min(segment)

        amp = baseline - peak

        # amplitude filter

        if not (min_amp < amp < max_amp):
            continue

        # -----------------------------------
        # RESTORE PREVIOUS STATE IF POSSIBLE
        # -----------------------------------

        status = True
        base = None
        peak_t = None   # saved peak time (set after base drag or on restore)

        if saved_events is not None:

            for old_event in saved_events:

                if abs(old_event["ct"] - ct) < 1e-6:

                    status   = old_event.get("status", True)
                    base     = old_event.get("base",   None)
                    peak_t   = old_event.get("peak_t", None)

                    break

        # -----------------------------------
        # STORE EVENT
        # -----------------------------------

        events.append({

            "recording_path": rec_dir,
            "file": rec_file,

            "ct": ct,

            "status": status,
            "base":   base,
            "peak_t": peak_t,   # None → will be computed from search on first display

            "amp":       amp,
            "threshold": threshold
        })

    print(f"{rec_file}: {len(filtered_crossings)} crossings")


for rec_dir, rec_file in recordings:
    print(f"Processing {rec_file}")
    process_recording(rec_dir, rec_file)

print(f"Total events detected: {len(events)}")



trace_cache = {}

def load_trace(event):

    key = (event["recording_path"], event["file"])

    if key in trace_cache:
        return trace_cache[key]

    abf = pyabf.ABF(os.path.join(event["recording_path"], event["file"]))
    abf.setSweep(0)

    fs = abf.dataRate

    time = abf.sweepX
    signal = abf.sweepY

    analysis_mask = (time >= analysis_start_s) & (time <= analysis_end_s)

    time = time[analysis_mask]
    signal = signal[analysis_mask]

    filtered = highpass(signal, 1, fs)
    filtered = lowpass(filtered, 1000, fs)

    trace_cache[key] = (time, filtered, fs)

    return time, filtered, fs


# ----------------------------
# PEAK HELPERS
# ----------------------------

def compute_peak_between_bases(event):
    """Return (peak_time, peak_value) as the most negative sample between the
    two baseline markers.  Updates event['peak_t'] in place."""
    time, filtered, fs = load_trace(event)
    left_x, right_x = event["base"]
    mask = (time >= left_x) & (time <= right_x)
    if not np.any(mask):
        # fall back to the full search-window approach
        return _search_peak(event)
    seg_t = time[mask]
    seg_y = filtered[mask]
    peak_rel = np.argmin(seg_y)
    t_peak = seg_t[peak_rel]
    event["peak_t"] = float(t_peak)
    return t_peak, seg_y[peak_rel]


def _search_peak(event):
    """Original ±5 ms search around the crossing time (fallback / first display)."""
    time, filtered, fs = load_trace(event)
    ct  = event["ct"]
    idx = np.argmin(np.abs(time - ct))
    half = int(5e-3 * fs)
    s = max(0, idx - half)
    e = min(len(filtered), idx + half)
    peak_rel = np.argmin(filtered[s:e])
    peak_idx = s + peak_rel
    t_peak   = time[peak_idx]
    event["peak_t"] = float(t_peak)
    return t_peak, filtered[peak_idx]


def get_peak(event):
    """Return (peak_time, peak_value), using saved peak_t when available."""
    time, filtered, _ = load_trace(event)
    if event["peak_t"] is not None:
        t_peak = event["peak_t"]
        y_peak = np.interp(t_peak, time, filtered)
        return t_peak, y_peak
    return _search_peak(event)


# ----------------------------
# EVENT VIEWER — DUAL PANEL
# ----------------------------

current_event = 0

fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor("#1a1a2e")

gs = gridspec.GridSpec(
    2, 1,
    height_ratios=[1.6, 1],
    hspace=0.45,
    left=0.08, right=0.97,
    top=0.93, bottom=0.1
)

ax_top = fig.add_subplot(gs[0])   # full recording overview
ax_bot = fig.add_subplot(gs[1])   # zoomed event detail

for ax in (ax_top, ax_bot):
    ax.set_facecolor("#0d0d1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("#ddddff")

dragging = None      # "left" or "right" or None
drag_threshold = 0.002  # seconds proximity for selecting marker

# Track which recording is currently shown in the top panel
_top_panel_key = [None]

if len(events) == 0:
    print("No events detected.")
    sys.exit()


for i, event in enumerate(events):

    if event["base"] is None:

        ct = event["ct"]

        time, filtered, fs = load_trace(event)

        left  = max(time[0],  ct - 0.001)
        right = min(time[-1], ct + 0.001)

        event["base"] = [left, right]


def get_recording_key(event):
    return (event["recording_path"], event["file"])


def update_top_panel(force=False):
    """Redraw the full-recording overview. Only redraws if the recording changed
    or force=True, then always updates the current-event marker."""

    event   = events[current_event]
    rec_key = get_recording_key(event)

    time, filtered, fs = load_trace(event)

    # --- collect every event that belongs to this recording ---
    rec_events = [e for e in events if get_recording_key(e) == rec_key]

    if _top_panel_key[0] != rec_key or force:

        _top_panel_key[0] = rec_key

        ax_top.clear()
        ax_top.set_facecolor("#0d0d1a")
        for spine in ax_top.spines.values():
            spine.set_edgecolor("#444466")
        ax_top.tick_params(colors="#aaaacc", labelsize=8)
        ax_top.xaxis.label.set_color("#aaaacc")
        ax_top.yaxis.label.set_color("#aaaacc")

        # draw full trace (downsampled for speed if very long)
        ds = max(1, len(time) // 20000)
        ax_top.plot(time[::ds], filtered[::ds], color="#4a6fa5", lw=0.6, alpha=0.9)

        # threshold line
        thr = event["threshold"]
        ax_top.axhline(thr, color="#e05c5c", lw=0.8, ls="--", alpha=0.7,
                       label=f"Threshold ({thr:.2f} pA)")

        # tick marks for all detected events in this recording
        for e in rec_events:
            color = "#55dd88" if e["status"] else "#dd5566"
            ax_top.axvline(e["ct"], color=color, lw=0.6, alpha=0.5, zorder=2)

        ax_top.set_ylabel("Current (pA)", fontsize=8)
        ax_top.set_xlabel("Time (s)", fontsize=8)
        ax_top.set_ylim(ylim)
        ax_top.legend(fontsize=7, loc="upper right",
                      facecolor="#1a1a2e", edgecolor="#444466",
                      labelcolor="#aaaacc")

    # --- update / draw current-event marker ---
    # remove old marker artists tagged with _cur_marker
    for artist in ax_top.lines[:]:
        if getattr(artist, "_cur_marker", False):
            artist.remove()
    for artist in ax_top.collections[:]:
        if getattr(artist, "_cur_marker", False):
            artist.remove()
    for artist in ax_top.patches[:]:
        if getattr(artist, "_cur_marker", False):
            artist.remove()
    for artist in ax_top.texts[:]:
        if getattr(artist, "_cur_marker", False):
            artist.remove()

    ct  = event["ct"]
    ylims = ax_top.get_ylim()

    # vertical highlight line
    vline = ax_top.axvline(ct, color="#ffdd55", lw=1.5, zorder=5)
    vline._cur_marker = True

    # downward-pointing triangle at the top
    tri = ax_top.scatter([ct], [ylims[1] * 0.97], marker="v",
                         color="#ffdd55", s=60, zorder=6, clip_on=False)
    tri._cur_marker = True

    rec_n = list(dict.fromkeys(get_recording_key(e) for e in events)).index(rec_key) + 1
    n_rec = len(dict.fromkeys(get_recording_key(e) for e in events))

    ax_top.set_title(
        f"Recording {rec_n}/{n_rec}  —  {event['file']}   "
        f"({len(rec_events)} events detected)",
        fontsize=9, color="#ddddff"
    )


def update_plot():

    ax_bot.clear()
    ax_bot.set_facecolor("#0d0d1a")
    for spine in ax_bot.spines.values():
        spine.set_edgecolor("#444466")
    ax_bot.tick_params(colors="#aaaacc", labelsize=8)
    ax_bot.xaxis.label.set_color("#aaaacc")
    ax_bot.yaxis.label.set_color("#aaaacc")

    event = events[current_event]

    time, filtered, fs = load_trace(event)

    # ----------------------------
    # RESOLVE PEAK
    # ----------------------------

    t_peak, y_peak = get_peak(event)

    # build view window centred on the peak
    peak_idx = np.argmin(np.abs(time - t_peak))

    window_samples_view = int(window_samples_view_ms / 1000 * fs)

    start = max(0, peak_idx - window_samples_view // 2)
    end   = min(len(filtered), peak_idx + window_samples_view // 2)

    t = time[start:end]
    y = filtered[start:end]

    if len(t) < 2:
        return

    # plot signal
    ax_bot.plot(t, y, color="#7ecfff", lw=1.2)

    # peak marker
    ax_bot.scatter(t_peak, y_peak, color="#ff6b6b", s=60, zorder=5)

    # ----------------------------
    # BASELINES (DRAGGABLE)
    # ----------------------------

    left_x, right_x = event["base"]

    left_y  = np.interp(left_x,  t, y)
    right_y = np.interp(right_x, t, y)

    ax_bot.scatter(left_x,  left_y,  color="#55ccff", s=60, zorder=6)
    ax_bot.scatter(right_x, right_y, color="#55ccff", s=60, zorder=6)

    mask = (t >= left_x) & (t <= right_x)

    if np.any(mask):

        t_seg = t[mask]
        y_seg = y[mask]

        left_y  = np.interp(left_x,  t, y)
        right_y = np.interp(right_x, t, y)

        top    = np.column_stack([t_seg, y_seg])
        bottom = np.array([[right_x, right_y], [left_x, left_y]])
        verts  = np.vstack([top, bottom])

        poly = Polygon(
            verts,
            closed=True,
            facecolor="#aa55ff",
            alpha=0.25,
            edgecolor=None
        )
        ax_bot.add_patch(poly)

    # status styling
    if event["status"]:
        ax_bot.set_facecolor("#0a1f0a")
        label = "ACCEPTED ✓"
        title_color = "#88ff99"
    else:
        ax_bot.set_facecolor("#1f0a0a")
        label = "REJECTED ✗"
        title_color = "#ff8888"

    for spine in ax_bot.spines.values():
        spine.set_edgecolor("#444466")

    ax_bot.set_title(
        f"Event {current_event+1}/{len(events)}  —  {label}   "
        f"[← → navigate  |  ↑ accept  |  ↓ reject  |  Q save & quit]",
        fontsize=8.5, color=title_color
    )
    ax_bot.set_xlabel("Time (s)", fontsize=8)
    ax_bot.set_ylabel("Current (pA)", fontsize=8)
    ax_bot.set_ylim(ylim)

    # also refresh top panel marker / recording
    update_top_panel()

    fig.canvas.draw_idle()


def on_key(key_event):

    global current_event

    # navigate
    if key_event.key == "right":
        current_event = (current_event + 1) % len(events)

    elif key_event.key == "left":
        current_event = (current_event - 1) % len(events)

    # accept
    elif key_event.key == "up":
        events[current_event]["status"] = True
        print(f"Accepted event {current_event+1}")

    # reject
    elif key_event.key == "down":
        events[current_event]["status"] = False
        print(f"Rejected event {current_event+1}")

    # SAVE + EXIT
    elif key_event.key == "q":

        print("Saving per-recording outputs...")

        from collections import defaultdict
        grouped = defaultdict(list)

        for e in events:
            grouped[e["recording_path"]].append(e)

        for rec_path, rec_events in grouped.items():

            out_file = os.path.join(rec_path, "event_status.npy")

            np.save(out_file, {
                "events": rec_events
            }, allow_pickle=True)

            print(f"Saved: {out_file}")

        plt.close(fig)
        return

    update_plot()


def on_press(mouse_event):

    global dragging

    if mouse_event.inaxes != ax_bot:
        return

    current = events[current_event]

    left_x, right_x = current["base"]

    if abs(mouse_event.xdata - left_x) < drag_threshold:
        dragging = "left"

    elif abs(mouse_event.xdata - right_x) < drag_threshold:
        dragging = "right"


def on_release(mouse_event):
    global dragging
    dragging = None


def on_motion(mouse_event):

    if dragging is None or mouse_event.inaxes != ax_bot:
        return

    if mouse_event.xdata is None:
        return

    current = events[current_event]

    left_x, right_x = current["base"]

    if dragging == "left":
        current["base"][0] = min(mouse_event.xdata, right_x - 1e-4)

    elif dragging == "right":
        current["base"][1] = max(mouse_event.xdata, left_x + 1e-4)

    # recompute peak between the updated bases and store it on the event
    compute_peak_between_bases(current)

    update_plot()


fig.canvas.mpl_connect("key_press_event",      on_key)
fig.canvas.mpl_connect("button_press_event",   on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event",  on_motion)


# force full draw on first render
update_top_panel(force=True)
update_plot()
plt.show()