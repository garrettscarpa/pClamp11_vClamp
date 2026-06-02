#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:39:32 2026
@author: gs075
"""
from scipy.signal import butter, filtfilt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, TextBox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pyabf
import sys
import os


# Load data
root = '/Users/gs075/Desktop/Voltage Clamp'

# Detection metrics
min_amp = 6
max_amp = 80
threshold_sigma = 5

# Temporal filtering
analysis_start_s = 0
analysis_end_s   = 60
window_samples_view_ms = 400
ylim = (-40, 5)

noise_filter_pos_voltage = 10
noise_filter_window_ms = 5

recordings = []
for subdir, _, files in os.walk(root):
    for f in files:
        if f.endswith(".abf"):
            recordings.append((subdir, f))
print(f"Found {len(recordings)} recordings")


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

    start_s = analysis_start_s if analysis_start_s is not None else time[0]
    end_s   = analysis_end_s   if analysis_end_s   is not None else time[-1]

    analysis_mask = (time >= start_s) & (time <= end_s)
    time_f   = time[analysis_mask]
    signal_f = signal[analysis_mask]

    filtered = highpass(signal_f, 1, fs)
    filtered = lowpass(filtered, 1000, fs)

    mad       = np.median(np.abs(filtered - np.median(filtered)))
    noise     = mad / 0.6745
    threshold = -threshold_sigma * noise

    save_file    = os.path.join(rec_dir, "event_status.npy")
    saved_events = None

    if os.path.exists(save_file):
        try:
            saved        = np.load(save_file, allow_pickle=True).item()
            saved_events = saved.get("events", None)
            print(f"Loaded previous session: {rec_file}")
        except Exception as e:
            print(f"Could not load previous session for {rec_file}: {e}")
            saved_events = None

    # If a previous session exists, restore it verbatim and skip re-detection.
    # This preserves manually added peaks, adjusted bases, accept/reject
    # status, peak times, and AUC exactly as last saved.
    if saved_events is not None:
            restored = 0
            for ev in saved_events:
                if ev.get("file") != rec_file:
                    continue
                ev = dict(ev)
                ev["recording_path"] = rec_dir
                ev["file"]           = rec_file
                ev.setdefault("manually_added", False)
                ev.setdefault("auc", None)
                events.append(ev)
                restored += 1

            if restored > 0:
                print(f"Restored {restored} saved events for {rec_file} "
                      f"(detection skipped)")
                return
            # No saved events for THIS file → fall through to fresh detection
            print(f"No saved events for {rec_file}; running detection.")

    crossings = []
    for i in range(1, len(filtered)):
        if filtered[i - 1] > threshold and filtered[i] <= threshold:
            crossings.append(time_f[i])

    min_event_ms      = 0
    min_event_samples = int(min_event_ms / 1000 * fs)
    filtered_crossings = []

    for ct in crossings:
        idx        = np.argmin(np.abs(time_f - ct))
        search_win = int(2e-3 * fs)
        start      = max(0, idx - search_win)
        end        = min(len(filtered), idx + search_win)
        segment    = filtered[start:end]
        below      = segment <= threshold
        max_run = run = 0
        for b in below:
            if b:
                run     += 1
                max_run  = max(max_run, run)
            else:
                run = 0
        if max_run >= min_event_samples:
            filtered_crossings.append(ct)

    window_ms      = 15
    window_samples = int(window_ms / 1000 * fs)

    for ct in filtered_crossings:
        idx   = np.argmin(np.abs(time_f - ct))
        start = max(0, idx - int(2e-3 * fs))
        end   = start + window_samples

        if end >= len(filtered):
            continue

        segment  = filtered[start:end]
        baseline = np.median(segment[:5])
        peak     = np.min(segment)
        amp      = baseline - peak

        if not (min_amp < amp < max_amp):
            continue

        peak_idx_local = np.argmin(segment)
        peak_abs_idx   = start + peak_idx_local
        artifact_window = int(noise_filter_window_ms * fs)
        art_start = max(0, peak_abs_idx - artifact_window // 2)
        art_end   = min(len(filtered), peak_abs_idx + artifact_window)

        if np.any(filtered[art_start:art_end] >= noise_filter_pos_voltage):
            continue

        status = True
        base   = None
        peak_t = None

        if saved_events is not None:
            for old_event in saved_events:
                if abs(old_event["ct"] - ct) < 1e-6:
                    status = old_event.get("status", True)
                    base   = old_event.get("base",   None)
                    peak_t = old_event.get("peak_t", None)
                    break

        events.append({
            "recording_path": rec_dir,
            "file":           rec_file,
            "ct":             ct,
            "status":         status,
            "base":           base,
            "peak_t":         peak_t,
            "amp":            amp,
            "threshold":      threshold,
            "auc":            None,
            "manually_added": False,
        })


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

    fs     = abf.dataRate
    time   = abf.sweepX
    signal = abf.sweepY

    analysis_mask = (time >= analysis_start_s) & (time <= analysis_end_s)
    time   = time[analysis_mask]
    signal = signal[analysis_mask]

    filtered = highpass(signal, 1, fs)
    filtered = lowpass(filtered, 1000, fs)

    trace_cache[key] = (time, filtered, fs)
    return time, filtered, fs


# ----------------------------
# PEAK HELPERS
# ----------------------------
def compute_auc(event):
    time, filtered, fs = load_trace(event)
    left_x, right_x = event["base"]
    mask = (time >= left_x) & (time <= right_x)
    if not np.any(mask):
        event["auc"] = None
        return None
    seg_t    = time[mask]
    seg_y    = filtered[mask]
    y_left   = np.interp(left_x,  time, filtered)
    y_right  = np.interp(right_x, time, filtered)
    baseline = np.interp(seg_t, [left_x, right_x], [y_left, y_right])
    auc      = float(np.abs(np.trapz(seg_y - baseline, seg_t)))
    event["auc"] = auc
    return auc


def compute_peak_between_bases(event):
    time, filtered, fs = load_trace(event)
    left_x, right_x = event["base"]
    mask = (time >= left_x) & (time <= right_x)
    if not np.any(mask):
        return _search_peak(event)
    seg_t    = time[mask]
    seg_y    = filtered[mask]
    peak_rel = np.argmin(seg_y)
    t_peak   = seg_t[peak_rel]
    event["peak_t"] = float(t_peak)
    return t_peak, seg_y[peak_rel]


def _search_peak(event):
    time, filtered, fs = load_trace(event)
    ct       = event["ct"]
    idx      = np.argmin(np.abs(time - ct))
    half     = int(5e-3 * fs)
    s        = max(0, idx - half)
    e        = min(len(filtered), idx + half)
    peak_rel = np.argmin(filtered[s:e])
    peak_idx = s + peak_rel
    t_peak   = time[peak_idx]
    event["peak_t"] = float(t_peak)
    return t_peak, filtered[peak_idx]


def get_peak(event):
    time, filtered, _ = load_trace(event)
    if event["peak_t"] is not None:
        t_peak = event["peak_t"]
        y_peak = np.interp(t_peak, time, filtered)
        return t_peak, y_peak
    return _search_peak(event)


def get_recording_key(event):
    return (event["recording_path"], event["file"])


# ----------------------------
# ADD PEAK — helper
# ----------------------------
def insert_manual_peak(click_time):
    """
    Find the local minimum within ±5 ms of click_time in the current recording,
    create a new event, and insert it into `events` in chronological order
    relative to other events from the same recording.

    The key fix vs the previous version: we find the correct global insertion
    index by scanning only same-recording events and tracking their global
    indices, rather than searching the entire (mixed-recording) list and
    potentially falling off the end.
    """
    global current_event

    ref_event          = events[current_event]
    rec_key            = get_recording_key(ref_event)
    rec_dir            = ref_event["recording_path"]
    rec_file           = ref_event["file"]
    threshold_val      = ref_event["threshold"]
    time, filtered, fs = load_trace(ref_event)

    # --- find local minimum near click ---
    search_half   = int(5e-3 * fs)
    click_idx     = np.argmin(np.abs(time - click_time))
    s             = max(0, click_idx - search_half)
    e             = min(len(filtered), click_idx + search_half)
    local_min_rel = np.argmin(filtered[s:e])
    peak_abs_idx  = s + local_min_rel
    t_peak        = float(time[peak_abs_idx])

    # default bases ±3 ms around the peak
    base_half = int(3e-3 * fs)
    left_x  = float(time[max(0, peak_abs_idx - base_half)])
    right_x = float(time[min(len(time) - 1, peak_abs_idx + base_half)])

    new_event = {
        "recording_path": rec_dir,
        "file":           rec_file,
        "ct":             t_peak,
        "status":         True,
        "base":           [left_x, right_x],
        "peak_t":         t_peak,
        "amp":            0.0,
        "threshold":      threshold_val,
        "auc":            None,
        "manually_added": True,
    }

    # compute amp
    y_peak = filtered[peak_abs_idx]
    baseline_est = np.median(
        filtered[max(0, peak_abs_idx - int(2e-3 * fs)): peak_abs_idx + 1]
    )
    new_event["amp"] = float(abs(baseline_est - y_peak))
    compute_auc(new_event)

    # ------------------------------------------------------------------
    # Find the correct global insertion index.
    #
    # Strategy: collect (global_index, peak_time) for every event that
    # belongs to this recording, then find the first one whose peak time
    # is later than t_peak.  Insert immediately before that global index.
    # If no same-recording event has a later time, insert just after the
    # last same-recording event (not at the very end of the full list).
    # ------------------------------------------------------------------
    same_rec_indices = [
        i for i, ev in enumerate(events)
        if get_recording_key(ev) == rec_key
    ]

    insert_pos = None
    for gi in same_rec_indices:
        tp, _ = get_peak(events[gi])
        if tp > t_peak:
            insert_pos = gi   # insert before this global index
            break

    if insert_pos is None:
        # New peak is later than all existing same-recording events —
        # place it just after the last same-recording event.
        insert_pos = same_rec_indices[-1] + 1 if same_rec_indices else len(events)

    events.insert(insert_pos, new_event)
    current_event = insert_pos

    print(
        f"Manually added peak at t={t_peak:.4f} s  "
        f"(event {current_event + 1}/{len(events)})"
    )


# ----------------------------
# LAYOUT
# ----------------------------
current_event = 0

fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor("#1a1a2e")

gs = gridspec.GridSpec(
    2, 1,
    height_ratios=[1.6, 1],
    hspace=0.45,
    left=0.08, right=0.97,
    top=0.93, bottom=0.13
)

ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1])

for ax in (ax_top, ax_bot):
    ax.set_facecolor("#0d0d1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("#ddddff")


# ----------------------------
# ADD PEAK — button & state
# ----------------------------
_add_peak_active = [False]

ax_btn = fig.add_axes([0.08, 0.02, 0.15, 0.045])
btn_add_peak = Button(
    ax_btn, "➕  Add Peak",
    color="#1e2240", hovercolor="#2e3460"
)
btn_add_peak.label.set_color("#aaddff")
btn_add_peak.label.set_fontsize(9)

def toggle_add_peak(mouse_event):
    _add_peak_active[0] = not _add_peak_active[0]
    if _add_peak_active[0]:
        btn_add_peak.color      = "#3a5a3a"
        btn_add_peak.hovercolor = "#4a7a4a"
        btn_add_peak.label.set_text("🔴  Click on Signal")
        btn_add_peak.label.set_color("#88ff99")
        fig.canvas.set_cursor(2)
    else:
        btn_add_peak.color      = "#1e2240"
        btn_add_peak.hovercolor = "#2e3460"
        btn_add_peak.label.set_text("➕  Add Peak")
        btn_add_peak.label.set_color("#aaddff")
        fig.canvas.set_cursor(1)
    ax_btn.figure.canvas.draw_idle()

btn_add_peak.on_clicked(toggle_add_peak)


# ----------------------------
# JUMP TO PEAK — text box
# ----------------------------
ax_jump = fig.add_axes([0.35, 0.02, 0.10, 0.045])
txt_jump = TextBox(
    ax_jump, "Go to # ",
    initial="",
    color="#1e2240", hovercolor="#2e3460"
)
txt_jump.label.set_color("#aaddff")
txt_jump.label.set_fontsize(9)
txt_jump.text_disp.set_color("#aaddff")

def jump_to_peak(text):
    global current_event
    try:
        n = int(text.strip())
    except ValueError:
        print(f"Invalid peak number: '{text}'")
        return
    if not (1 <= n <= len(events)):
        print(f"Peak number out of range: {n} (valid 1-{len(events)})")
        return
    current_event = n - 1
    _top_panel_key[0] = None   # force top panel rebuild if recording changed
    update_plot()
    print(f"Jumped to event {current_event + 1}/{len(events)}")

txt_jump.on_submit(jump_to_peak)

# ----------------------------
# DRAG STATE
# ----------------------------
dragging       = None
drag_threshold = 0.002
_top_panel_key = [None]
_view_xlim     = [None, None]


def update_top_panel(force=False):
    event   = events[current_event]
    rec_key = get_recording_key(event)

    time, filtered, fs = load_trace(event)
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

        ds = max(1, len(time) // 20000)
        if ds > 1:
            n_blocks    = len(time) // ds
            t_ds        = time[:n_blocks * ds].reshape(n_blocks, ds)[:, ds // 2]
            y_ds        = filtered[:n_blocks * ds].reshape(n_blocks, ds)
            y_min       = y_ds.min(axis=1)
            y_max       = y_ds.max(axis=1)
            t_env       = np.repeat(t_ds, 2)
            y_env       = np.empty(len(t_ds) * 2)
            y_env[0::2] = y_min
            y_env[1::2] = y_max
            ax_top.plot(t_env, y_env, color="#4a6fa5", lw=0.5, alpha=0.9)
        else:
            ax_top.plot(time, filtered, color="#4a6fa5", lw=0.6, alpha=0.9)

        thr = event["threshold"]
        ax_top.axhline(thr, color="#e05c5c", lw=0.8, ls="--", alpha=0.7,
                       label=f"Threshold ({thr:.2f} pA)")

        for e in rec_events:
            color = "#55dd88" if e["status"] else "#dd5566"
            if e.get("manually_added", False):
                color = "#ffaa33" if e["status"] else "#dd5566"
            t_pk, _ = get_peak(e)
            ax_top.axvline(t_pk, color=color, lw=0.6, alpha=0.5, zorder=2)

        ax_top.set_ylabel("Current (pA)", fontsize=8)
        ax_top.set_xlabel("Time (s)", fontsize=8)
        ax_top.set_ylim(ylim)
        ax_top.legend(fontsize=7, loc="upper right",
                      facecolor="#1a1a2e", edgecolor="#444466",
                      labelcolor="#aaaacc")

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

    t_peak, _ = get_peak(event)
    ylims     = ax_top.get_ylim()

    vline = ax_top.axvline(t_peak, color="#ffdd55", lw=1.5, zorder=5)
    vline._cur_marker = True

    tri = ax_top.scatter([t_peak], [ylims[1] * 0.97], marker="v",
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
    event = events[current_event]
    time, filtered, fs = load_trace(event)

    t_peak, y_peak = get_peak(event)
    compute_auc(event)

    peak_idx            = np.argmin(np.abs(time - t_peak))
    window_samples_view = int(window_samples_view_ms / 1000 * fs)

    default_start = max(0, peak_idx - window_samples_view // 2)
    default_end   = min(len(filtered), peak_idx + window_samples_view // 2)
    default_xlim  = (time[default_start], time[default_end - 1])

    if dragging is not None and _view_xlim[0] is not None:
        x_left  = _view_xlim[0]
        x_right = _view_xlim[1]
        right_base = events[current_event]["base"][1]
        if right_base > x_right - (x_right - x_left) * 0.05:
            x_right = min(time[-1], right_base + (x_right - x_left) * 0.1)
        left_base = events[current_event]["base"][0]
        if left_base < x_left + (x_right - x_left) * 0.05:
            x_left = max(time[0], left_base - (x_right - x_left) * 0.1)
        new_xlim = (x_left, x_right)
    else:
        new_xlim = default_xlim

    _view_xlim[0], _view_xlim[1] = new_xlim

    ax_bot.clear()
    ax_bot.set_facecolor("#0d0d1a")
    for spine in ax_bot.spines.values():
        spine.set_edgecolor("#444466")
    ax_bot.tick_params(colors="#aaaacc", labelsize=8)
    ax_bot.xaxis.label.set_color("#aaaacc")
    ax_bot.yaxis.label.set_color("#aaaacc")

    view_mask = (time >= new_xlim[0] - time[1]) & (time <= new_xlim[1] + time[1])
    t = time[view_mask]
    y = filtered[view_mask]

    if len(t) < 2:
        return

    ax_bot.plot(t, y, color="#7ecfff", lw=1.2)
    ax_bot.scatter(t_peak, y_peak, color="#ff6b6b", s=60, zorder=5)

    left_x, right_x = event["base"]
    left_y  = np.interp(left_x,  t, y)
    right_y = np.interp(right_x, t, y)

    ax_bot.scatter(left_x,  left_y,  color="#55ccff", s=60, zorder=6)
    ax_bot.scatter(right_x, right_y, color="#55ccff", s=60, zorder=6)

    mask = (t >= left_x) & (t <= right_x)
    if np.any(mask):
        t_seg  = t[mask]
        y_seg  = y[mask]
        top    = np.column_stack([t_seg, y_seg])
        bottom = np.array([[right_x, right_y], [left_x, left_y]])
        verts  = np.vstack([top, bottom])
        poly   = Polygon(verts, closed=True, facecolor="#aa55ff", alpha=0.25, edgecolor=None)
        ax_bot.add_patch(poly)

    if _add_peak_active[0]:
        ax_bot.set_facecolor("#0a1a0a")
        ax_bot.text(
            0.5, 0.97, "🖱  Click on the signal to place a new peak",
            transform=ax_bot.transAxes, ha="center", va="top",
            color="#88ff99", fontsize=8, alpha=0.85
        )

    if event["status"]:
        label       = "ACCEPTED"
        title_color = "#88ff99"
        if not _add_peak_active[0]:
            ax_bot.set_facecolor("#0a1f0a")
    else:
        label       = "REJECTED"
        title_color = "#ff8888"
        if not _add_peak_active[0]:
            ax_bot.set_facecolor("#1f0a0a")

    manual_tag = "  [manual]" if event.get("manually_added", False) else ""
    ax_bot.set_title(
        f"Event {current_event+1}/{len(events)}  —  {label}{manual_tag}   "
        f"[<- -> : navigate  |  up : accept  |  down : reject  |  Q : save & quit]",
        fontsize=8.5, color=title_color
    )
    ax_bot.set_xlabel("Time (s)", fontsize=8)
    ax_bot.set_ylabel("Current (pA)", fontsize=8)
    ax_bot.set_ylim(ylim)
    ax_bot.set_xlim(new_xlim)

    update_top_panel()
    fig.canvas.draw_idle()


# ----------------------------
# INIT BASES
# ----------------------------
if len(events) == 0:
    print("No events detected.")
    sys.exit()

for i, event in enumerate(events):
    if event["base"] is None:
        ct     = event["ct"]
        time, filtered, fs = load_trace(event)
        left   = max(time[0],  ct - 0.001)
        right  = min(time[-1], ct + 0.001)
        event["base"] = [left, right]
        event["auc"]  = None


# ----------------------------
# EVENT CALLBACKS
# ----------------------------
def on_key(key_event):
    global current_event

    if key_event.key == "right":
        current_event = (current_event + 1) % len(events)
    elif key_event.key == "left":
        current_event = (current_event - 1) % len(events)
    elif key_event.key == "up":
        events[current_event]["status"] = True
        print(f"Accepted event {current_event+1}")
    elif key_event.key == "down":
        events[current_event]["status"] = False
        print(f"Rejected event {current_event+1}")
    elif key_event.key == "q":
        print("Saving per-recording outputs...")
        from collections import defaultdict
        grouped = defaultdict(list)
        for e in events:
            grouped[e["recording_path"]].append(e) 
        for rec_path, rec_events in grouped.items():
            out_file = os.path.join(rec_path, "event_status.npy")
            np.save(out_file, {"events": rec_events}, allow_pickle=True)
            print(f"Saved: {out_file}")
        plt.close(fig)
        return

    update_plot()


def on_press(mouse_event):
    global dragging

    if _add_peak_active[0]:
        if mouse_event.inaxes == ax_bot and mouse_event.xdata is not None:
            toggle_add_peak(None)
            insert_manual_peak(mouse_event.xdata)
            _top_panel_key[0] = None
            update_plot()
        return

    if mouse_event.inaxes != ax_bot:
        return

    current        = events[current_event]
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

    current        = events[current_event]
    left_x, right_x = current["base"]

    if dragging == "left":
        current["base"][0] = min(mouse_event.xdata, right_x - 1e-4)
    elif dragging == "right":
        current["base"][1] = max(mouse_event.xdata, left_x + 1e-4)

    compute_peak_between_bases(current)
    compute_auc(current)
    update_plot()


fig.canvas.mpl_connect("key_press_event",      on_key)
fig.canvas.mpl_connect("button_press_event",   on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event",  on_motion)

update_top_panel(force=True)
update_plot()
plt.show()