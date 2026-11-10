"""Visualization helpers for range profiles and extracted respiration signals.

This module translates parsed captures and extracted signal dictionaries into
the figures shown in the notebook. It also reports what a given capture can be
expected to visualize based on its recorded TLV inventory.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .capture_pipeline import CaptureBundle, TLV_NAMES
from .parser_mmw_demo import TLV_TYPE_AZIMUT_STATIC_HEAT_MAP, TLV_TYPE_RANGE_DOPPLER_HEAT_MAP, TLV_TYPE_RANGE_PROFILE
from .respiration_pipeline import build_range_profile_matrix, choose_respiration_window_seconds


def _gui_monitor_command(bundle: CaptureBundle) -> str | None:
    """Rebuild the `guiMonitor` command string from parsed config commands."""

    for command in bundle.config.get("commands", []):
        if isinstance(command, str) and command.strip().startswith("guiMonitor"):
            return command.strip()
        if isinstance(command, dict) and command.get("command") == "guiMonitor":
            args = [str(value) for value in command.get("args", [])]
            return " ".join(["guiMonitor", *args])
    return None


def describe_capture_visualizations(bundle: CaptureBundle) -> dict[str, Any]:
    """Report which plots are valid for this capture's recorded TLVs."""

    tlv_types = sorted({tlv_type for frame in bundle.frames for tlv_type, _ in frame.tlvs})
    tlv_names = [TLV_NAMES.get(tlv_type, f"unknown_{tlv_type}") for tlv_type in tlv_types]
    has_range_doppler = TLV_TYPE_RANGE_DOPPLER_HEAT_MAP in tlv_types
    has_azimuth_heatmap = TLV_TYPE_AZIMUT_STATIC_HEAT_MAP in tlv_types

    return {
        "gui_monitor_command": _gui_monitor_command(bundle),
        "tlv_types_present": tlv_types,
        "tlv_names_present": tlv_names,
        "can_plot_range_profile_waterfall": TLV_TYPE_RANGE_PROFILE in tlv_types,
        "can_plot_respiration_signal": TLV_TYPE_RANGE_PROFILE in tlv_types,
        "can_plot_true_range_doppler_heatmap": has_range_doppler,
        "can_plot_true_azimuth_heatmap": has_azimuth_heatmap,
        "doppler_heatmap_note": (
            "This recording does not contain a true range-Doppler heatmap TLV. "
            "You can still plot range-vs-time heatmaps and respiration spectrograms from the recorded range profiles."
            if not has_range_doppler
            else "This recording contains the range-Doppler heatmap TLV."
        ),
    }


def _prepare_range_profile_image(
    bundle: CaptureBundle,
    *,
    max_range_m: float | None = None,
    log_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert range profiles into an image-ready array and matching axes."""

    matrix_data = build_range_profile_matrix(bundle)
    range_axis = np.asarray(matrix_data["range_axis_m"], dtype=float)
    times_s = np.asarray(matrix_data["times_s"], dtype=float)
    profiles = np.asarray(matrix_data["profiles"], dtype=float)

    if max_range_m is not None:
        keep_mask = range_axis <= max_range_m
        if not np.any(keep_mask):
            raise ValueError(f"max_range_m={max_range_m} leaves no range bins to plot")
        range_axis = range_axis[keep_mask]
        profiles = profiles[:, keep_mask]

    image_data = np.log10(np.maximum(profiles, 1.0)) if log_scale else profiles
    return range_axis, times_s, image_data, matrix_data


def plot_range_profile_waterfall(
    bundle: CaptureBundle,
    *,
    max_range_m: float | None = None,
    log_scale: bool = True,
    cmap: str = "magma",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot range-profile magnitude as a range-vs-time waterfall."""

    range_axis, times_s, image_data, matrix_data = _prepare_range_profile_image(
        bundle,
        max_range_m=max_range_m,
        log_scale=log_scale,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    time_start = float(times_s[0]) if len(times_s) else 0.0
    time_end = float(times_s[-1] + matrix_data["frame_period_s"]) if len(times_s) else matrix_data["frame_period_s"]
    image = ax.imshow(
        image_data,
        aspect="auto",
        origin="lower",
        extent=[float(range_axis[0]), float(range_axis[-1]), time_start, time_end],
        cmap=cmap,
    )
    ax.set_title("Range-Time Waterfall")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Time (s)")
    fig.colorbar(image, ax=ax, label="log10 magnitude" if log_scale else "magnitude")
    return fig, ax


def plot_respiration_signal(
    signal: dict[str, Any],
    *,
    use_zscore: bool = True,
    color: str = "#0b7285",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the extracted respiration signal as a simple time trace."""

    times_s = np.asarray(signal["times_s"], dtype=float)
    series_key = "zscore_signal" if use_zscore else "centered_signal"
    values = np.asarray(signal[series_key], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    ax.plot(times_s, values, color=color, linewidth=1.5)
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_title(f"Respiration Signal at {signal['center_range_m']:.3f} m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score" if use_zscore else "centered magnitude")
    return fig, ax


def choose_respiration_visual_window_seconds(
    duration_s: float,
    *,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    preferred_hop_fraction: float = 0.20,
    min_window_seconds: float = 15.0,
    max_window_seconds: float = 20.0,
) -> dict[str, Any]:
    """Choose a spectrogram window intended for visual trend inspection.

    This is deliberately shorter than the estimation window from
    choose_respiration_window_seconds. It gives more time slices for mixed-rate
    experiments while reporting when the window is not reliable at the slow BPM
    limit.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if preferred_hop_fraction <= 0.0:
        raise ValueError("preferred_hop_fraction must be positive")

    duration = float(duration_s)
    if duration <= min_window_seconds:
        window_seconds = duration
    else:
        window_seconds = min(max_window_seconds, max(min_window_seconds, duration / 3.0))
        window_seconds = min(window_seconds, duration)

    hop_seconds = max(1e-9, window_seconds * preferred_hop_fraction)
    cycles_at_min_bpm = window_seconds * min_bpm / 60.0
    cycles_at_typical_slow_bpm = window_seconds * max(min_bpm, 8.0) / 60.0
    frequency_resolution_hz = 1.0 / window_seconds
    frequency_resolution_bpm = 60.0 * frequency_resolution_hz
    reliable_for_min_bpm = bool(cycles_at_min_bpm >= 3.0)
    decision_reason = (
        f"Visual spectrogram window uses {window_seconds:.1f} s with {hop_seconds:.1f} s hop "
        f"to show more time slices across mixed BPM recordings. It contains "
        f"{cycles_at_min_bpm:.2f} cycles at {min_bpm:.1f} BPM and "
        f"{cycles_at_typical_slow_bpm:.2f} cycles at {max(min_bpm, 8.0):.1f} BPM."
    )
    if not reliable_for_min_bpm:
        decision_reason += " Use the longer respiration-aware window for primary slow-BPM estimation."

    return {
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "frequency_resolution_hz": float(frequency_resolution_hz),
        "frequency_resolution_bpm": float(frequency_resolution_bpm),
        "cycles_at_min_bpm": float(cycles_at_min_bpm),
        "cycles_at_typical_slow_bpm": float(cycles_at_typical_slow_bpm),
        "reliable_for_min_bpm": reliable_for_min_bpm,
        "decision_reason": decision_reason,
        "window_mode": "visual",
    }


def compute_respiration_spectrogram(
    signal: dict[str, Any],
    *,
    window_seconds: float | None = None,
    hop_seconds: float | None = None,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = 2.0,
    use_respiration_window: bool = False,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
) -> dict[str, Any]:
    """Build a band-limited spectrogram plus metadata about the chosen window."""

    samples = np.asarray(signal["centered_signal"], dtype=float)
    frame_period_s = float(signal["frame_period_s"])
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if samples.size < 16:
        raise ValueError("Respiration signal is too short to build a spectrogram")

    sample_rate_hz = 1.0 / frame_period_s
    duration_s = float(signal["times_s"][-1] + frame_period_s) if signal.get("times_s") else float(samples.size * frame_period_s)
    window_choice: dict[str, Any] | None = None
    if use_respiration_window:
        window_choice = choose_respiration_window_seconds(
            duration_s,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            cycles_per_window=3.0,
            preferred_hop_fraction=0.25,
            max_window_fraction=0.75,
        )
        window_seconds = float(window_choice["window_seconds"])
        if hop_seconds is None:
            hop_seconds = float(window_choice["hop_seconds"])
        window_choice["window_mode"] = "estimation"
    elif window_seconds is None:
        window_choice = choose_respiration_visual_window_seconds(
            duration_s,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            preferred_hop_fraction=0.20,
        )
        window_seconds = float(window_choice["window_seconds"])
        if hop_seconds is None:
            hop_seconds = float(window_choice["hop_seconds"])
    if window_seconds is None:
        window_seconds = 8.0
    if hop_seconds is None:
        hop_seconds = max(1.0, window_seconds * 0.25)

    window_size = max(16, int(round(window_seconds * sample_rate_hz)))
    window_size = min(window_size, samples.size)
    hop_size = max(1, int(round(hop_seconds * sample_rate_hz)))
    actual_window_seconds = float(window_size * frame_period_s)
    actual_hop_seconds = float(hop_size * frame_period_s)

    if window_choice is None:
        cycles_at_min_bpm = actual_window_seconds * min_bpm / 60.0
        window_choice = {
            "window_seconds": actual_window_seconds,
            "hop_seconds": actual_hop_seconds,
            "frequency_resolution_hz": float(1.0 / actual_window_seconds),
            "frequency_resolution_bpm": float(60.0 / actual_window_seconds),
            "cycles_at_min_bpm": float(cycles_at_min_bpm),
            "reliable_for_min_bpm": bool(cycles_at_min_bpm >= 3.0),
            "decision_reason": (
                f"Manual spectrogram window contains {cycles_at_min_bpm:.2f} cycles at {min_bpm:.1f} BPM."
            ),
            "window_mode": "manual",
        }

    start_indices = np.arange(0, max(samples.size - window_size + 1, 1), hop_size)
    if start_indices.size == 0:
        start_indices = np.asarray([0])
    if start_indices[-1] != samples.size - window_size:
        start_indices = np.append(start_indices, samples.size - window_size)

    window = np.hanning(window_size)
    frequencies_hz = np.fft.rfftfreq(window_size, d=frame_period_s)
    power_rows: list[np.ndarray] = []
    times_s: list[float] = []

    for start_index in start_indices:
        segment = samples[start_index : start_index + window_size]
        segment = segment - np.mean(segment)
        spectrum = np.fft.rfft(segment * window)
        power_rows.append(np.abs(spectrum) ** 2)
        times_s.append((start_index + (window_size / 2.0)) * frame_period_s)

    power_db = 10.0 * np.log10(np.maximum(np.asarray(power_rows, dtype=float).T, 1e-12))
    keep_mask = np.ones_like(frequencies_hz, dtype=bool)
    if min_frequency_hz is not None:
        keep_mask &= frequencies_hz >= min_frequency_hz
    if max_frequency_hz is not None:
        keep_mask &= frequencies_hz <= max_frequency_hz
    if not np.any(keep_mask):
        raise ValueError("No spectrogram frequency bins remain after applying the requested frequency limits")

    if min_frequency_hz is not None or max_frequency_hz is not None:
        frequencies_hz = frequencies_hz[keep_mask]
        power_db = power_db[keep_mask, :]

    return {
        "times_s": times_s,
        "frequencies_hz": frequencies_hz.tolist(),
        "power_db": power_db,
        "sample_rate_hz": sample_rate_hz,
        "window_size": window_size,
        "hop_size": hop_size,
        "window_seconds": actual_window_seconds,
        "hop_seconds": actual_hop_seconds,
        "window_choice": window_choice,
        "window_mode": window_choice.get("window_mode"),
        "frequency_resolution_hz": float(1.0 / actual_window_seconds),
        "frequency_resolution_bpm": float(60.0 / actual_window_seconds),
        "reliable_for_min_bpm": bool(window_choice.get("reliable_for_min_bpm")),
    }


def _overlay_respiration_frequency_markers(
    ax: Axes,
    features: dict[str, Any] | None,
    *,
    orientation: str = "horizontal",
) -> None:
    if not features:
        return

    markers = [
        ("selected", features.get("selected_frequency_hz") or features.get("global_frequency_hz"), "#2b8a3e", "-"),
        (
            "raw peak",
            features.get("raw_spectral_frequency_hz") or features.get("global_raw_peak_frequency_hz"),
            "#c92a2a",
            "--",
        ),
        ("subharmonic", features.get("subharmonic_candidate_bpm", 0.0) / 60.0 if features.get("subharmonic_candidate_bpm") else None, "#5f3dc4", ":"),
    ]
    for label, frequency_hz, color, linestyle in markers:
        if frequency_hz is None:
            continue
        frequency = float(frequency_hz)
        if frequency <= 0.0:
            continue
        if orientation == "vertical":
            ax.axvline(frequency, color=color, linewidth=1.1, linestyle=linestyle, alpha=0.9)
            ax.text(
                frequency,
                0.98,
                label,
                color=color,
                fontsize=8,
                ha="right",
                va="top",
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )
        else:
            ax.axhline(frequency, color=color, linewidth=1.1, linestyle=linestyle, alpha=0.9)
            ax.text(
                0.99,
                frequency,
                label,
                color=color,
                fontsize=8,
                ha="right",
                va="bottom",
                transform=ax.get_yaxis_transform(),
            )


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.size == 0:
        return np.asarray([], dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first_edge = centers[0] - (midpoints[0] - centers[0])
    last_edge = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first_edge], midpoints, [last_edge]]).astype(float)


def plot_respiration_spectrogram(
    signal: dict[str, Any],
    *,
    window_seconds: float | None = None,
    hop_seconds: float | None = None,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = 2.0,
    use_respiration_window: bool = False,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    features: dict[str, Any] | None = None,
    cmap: str = "viridis",
    show_window_centers: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, dict[str, Any]]:
    """Plot the respiration spectrogram and return the computed spectrogram data."""

    spec = compute_respiration_spectrogram(
        signal,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        use_respiration_window=use_respiration_window,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    times_s = np.asarray(spec["times_s"], dtype=float)
    frequencies_hz = np.asarray(spec["frequencies_hz"], dtype=float)
    power_db = spec["power_db"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    # When there is only one valid window, fall back to a single spectrum so the
    # caller still gets a useful frequency-domain view instead of an empty plot.
    if power_db.shape[1] < 2:
        peak_index = int(np.argmax(power_db[:, 0]))
        peak_frequency_hz = float(frequencies_hz[peak_index])
        peak_power_db = float(power_db[peak_index, 0])

        ax.plot(frequencies_hz, power_db[:, 0], color="#0b7285", linewidth=1.8)
        ax.scatter([peak_frequency_hz], [peak_power_db], color="#c92a2a", zorder=3)
        ax.annotate(
            f"Peak {peak_frequency_hz * 60.0:.2f} BPM",
            xy=(peak_frequency_hz, peak_power_db),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )
        title = "Respiration Spectrum (single window)"
        if not spec.get("reliable_for_min_bpm"):
            title += " - lower confidence for slow RR"
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.grid(True, alpha=0.25)
        _overlay_respiration_frequency_markers(ax, features, orientation="vertical")
        return fig, ax, spec

    time_edges = _centers_to_edges(times_s)
    frequency_edges = _centers_to_edges(frequencies_hz)
    if time_edges.size:
        time_edges[0] = max(0.0, time_edges[0])
    image = ax.pcolormesh(
        time_edges,
        frequency_edges,
        power_db,
        shading="flat",
        cmap=cmap,
    )
    title = "Respiration Spectrogram"
    if not spec.get("reliable_for_min_bpm"):
        title += " - lower confidence for slow RR"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if show_window_centers and times_s.size > 1:
        ax.vlines(
            times_s,
            ymin=float(frequency_edges[0]),
            ymax=float(frequency_edges[-1]),
            colors="white",
            linewidth=0.45,
            alpha=0.22,
        )
    _overlay_respiration_frequency_markers(ax, features, orientation="horizontal")
    fig.colorbar(image, ax=ax, label="Power (dB)")
    return fig, ax, spec


def plot_capture_dashboard(
    bundle: CaptureBundle,
    signal: dict[str, Any],
    *,
    max_range_m: float | None = 2.5,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = 2.0,
    spectrogram_window_seconds: float | None = None,
    spectrogram_hop_seconds: float | None = None,
    use_respiration_window: bool | None = None,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    features: dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Create the notebook's standard 3-panel capture dashboard."""

    visualization_info = describe_capture_visualizations(bundle)
    figure, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    if use_respiration_window is None:
        use_respiration_window = False

    plot_range_profile_waterfall(bundle, max_range_m=max_range_m, ax=axes[0])
    plot_respiration_signal(signal, ax=axes[1])
    _, _, spectrogram = plot_respiration_spectrogram(
        signal,
        window_seconds=spectrogram_window_seconds,
        hop_seconds=spectrogram_hop_seconds,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        use_respiration_window=use_respiration_window,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        features=features,
        show_window_centers=True,
        ax=axes[2],
    )

    figure.suptitle("Capture Visualizations", fontsize=16)
    return figure, axes, visualization_info, spectrogram
