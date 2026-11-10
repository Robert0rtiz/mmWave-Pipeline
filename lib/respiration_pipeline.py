"""Respiration signal extraction, estimation, and quality scoring.

This module contains the project-specific heuristics that turn per-frame range
profiles into a breathing signal, then into rate, trend, and quality
summaries. The notebook calls a small set of public entrypoints here, while
many internal helpers support harmonic handling, transition detection, and
motion-aware range tracking.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from .capture_pipeline import CaptureBundle


RESPIRATION_BAND_HZ = (0.08, 1.0)


def _get_metadata_float(bundle: CaptureBundle, key: str, default: float | None = None) -> float | None:
    """Read one numeric config metadata field without failing on bad values."""

    value = bundle.config.get("metadata", {}).get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def frame_period_seconds(bundle: CaptureBundle) -> float:
    """Return the frame period declared in the paired config metadata."""

    frame_duration_ms = _get_metadata_float(bundle, "Frame Duration(msec)", 0.0) or 0.0
    return frame_duration_ms / 1000.0


def range_resolution_m(bundle: CaptureBundle) -> float:
    """Return the configured range-bin spacing in meters."""

    resolution = _get_metadata_float(bundle, "Range Resolution(m)", None)
    if resolution is None:
        raise ValueError("Range Resolution(m) is missing from config metadata")
    return resolution


def choose_respiration_window_seconds(
    duration_s: float,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    cycles_per_window: float = 3.0,
    preferred_hop_fraction: float = 0.25,
    min_window_seconds: float | None = None,
    max_window_fraction: float = 0.75,
) -> dict[str, Any]:
    """Choose a spectrogram/analysis window that is honest about slow breathing.

    A window that contains only about one slow breath can be useful for a
    responsive plot, but it is weak evidence for an estimator.  This helper
    returns both the chosen parameters and a reliability flag so callers do not
    silently treat a short window as high-confidence evidence.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if min_bpm <= 0.0 or max_bpm <= 0.0 or min_bpm >= max_bpm:
        raise ValueError("min_bpm and max_bpm must be positive with min_bpm < max_bpm")
    if cycles_per_window <= 0.0:
        raise ValueError("cycles_per_window must be positive")
    if preferred_hop_fraction <= 0.0:
        raise ValueError("preferred_hop_fraction must be positive")

    target_window_s = cycles_per_window * 60.0 / min_bpm
    minimum_window_s = 0.0 if min_window_seconds is None else max(0.0, float(min_window_seconds))

    if duration_s < target_window_s:
        window_seconds = duration_s
        reason = (
            f"Signal is shorter than the {target_window_s:.1f} s needed for "
            f"{cycles_per_window:.1f} cycles at {min_bpm:.1f} BPM; using the full record."
        )
    else:
        fraction_limit_s = duration_s
        if max_window_fraction is not None and max_window_fraction > 0.0:
            fraction_limit_s = max(minimum_window_s, duration_s * max_window_fraction)
        window_seconds = min(target_window_s, fraction_limit_s, duration_s)
        if window_seconds < target_window_s:
            reason = (
                f"Window was capped at {window_seconds:.1f} s by the available duration/window fraction; "
                f"target was {target_window_s:.1f} s for {cycles_per_window:.1f} cycles at {min_bpm:.1f} BPM."
            )
        else:
            reason = (
                f"Window contains about {cycles_per_window:.1f} cycles at the slow limit "
                f"({min_bpm:.1f} BPM)."
            )

    if minimum_window_s > 0.0 and window_seconds < minimum_window_s:
        window_seconds = min(duration_s, minimum_window_s)
        reason += f" Raised to the requested minimum window of {minimum_window_s:.1f} s."

    window_seconds = max(min(window_seconds, duration_s), min(duration_s, 1e-9))
    hop_seconds = max(window_seconds * preferred_hop_fraction, 1e-9)
    frequency_resolution_hz = 1.0 / window_seconds
    frequency_resolution_bpm = 60.0 * frequency_resolution_hz
    cycles_at_min_bpm = window_seconds * min_bpm / 60.0
    reliable_for_min_bpm = cycles_at_min_bpm >= (0.95 * cycles_per_window)
    if not reliable_for_min_bpm:
        reason += (
            f" This is lower confidence for {min_bpm:.1f} BPM because it contains only "
            f"{cycles_at_min_bpm:.2f} cycles at that rate."
        )

    return {
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "frequency_resolution_hz": float(frequency_resolution_hz),
        "frequency_resolution_bpm": float(frequency_resolution_bpm),
        "cycles_at_min_bpm": float(cycles_at_min_bpm),
        "reliable_for_min_bpm": bool(reliable_for_min_bpm),
        "decision_reason": reason,
    }


def build_range_profile_matrix(bundle: CaptureBundle) -> dict[str, Any]:
    """Build the frame-by-range matrix used throughout the respiration pipeline."""

    profiles = [frame.range_profile for frame in bundle.frames if frame.range_profile]
    if not profiles:
        raise ValueError("No range-profile TLVs were found in this capture")

    num_bins = len(profiles[0])
    frame_dt = frame_period_seconds(bundle)
    range_step = range_resolution_m(bundle)

    matrix: list[list[int]] = []
    times_s: list[float] = []
    frame_numbers: list[int] = []

    for frame_index, frame in enumerate(bundle.frames):
        profile = frame.range_profile if frame.range_profile else [0] * num_bins
        if len(profile) != num_bins:
            raise ValueError("Inconsistent range-profile bin count across frames")
        matrix.append(profile)
        times_s.append(frame_index * frame_dt)
        frame_numbers.append(frame.header.frame_number)

    range_axis_m = [bin_index * range_step for bin_index in range(num_bins)]
    return {
        "times_s": times_s,
        "frame_numbers": frame_numbers,
        "range_axis_m": range_axis_m,
        "profiles": matrix,
        "frame_period_s": frame_dt,
        "range_resolution_m": range_step,
    }


def summarize_range_bins(
    bundle: CaptureBundle,
    *,
    min_bin: int = 3,
    max_bin: int | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Rank range bins by simple amplitude statistics for quick inspection."""

    matrix_data = build_range_profile_matrix(bundle)
    matrix = matrix_data["profiles"]
    num_bins = len(matrix_data["range_axis_m"])
    upper = num_bins if max_bin is None else min(max_bin, num_bins)

    columns = list(zip(*matrix))
    bin_rows: list[dict[str, float | int]] = []
    for bin_index in range(max(0, min_bin), upper):
        column = columns[bin_index]
        bin_rows.append(
            {
                "bin_index": bin_index,
                "range_m": matrix_data["range_axis_m"][bin_index],
                "mean": mean(column),
                "std": pstdev(column),
            }
        )

    top_variable_bins = sorted(bin_rows, key=lambda row: row["std"], reverse=True)[:top_k]
    top_mean_bins = sorted(bin_rows, key=lambda row: row["mean"], reverse=True)[:top_k]
    return {
        "num_bins": num_bins,
        "frame_period_s": matrix_data["frame_period_s"],
        "range_resolution_m": matrix_data["range_resolution_m"],
        "top_variable_bins": top_variable_bins,
        "top_mean_bins": top_mean_bins,
    }


def score_respiration_range_bins(
    bundle: CaptureBundle,
    *,
    min_bin: int = 3,
    max_bin: int | None = None,
    respiration_band_hz: tuple[float, float] = RESPIRATION_BAND_HZ,
    top_k: int = 12,
) -> dict[str, Any]:
    """Score range bins by respiration-band concentration rather than variance alone."""
    matrix_data = build_range_profile_matrix(bundle)
    matrix = np.asarray(matrix_data["profiles"], dtype=float)
    frame_period_s = float(matrix_data["frame_period_s"])
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")

    num_bins = len(matrix_data["range_axis_m"])
    lower_bin = max(0, int(min_bin))
    upper_bin = num_bins if max_bin is None else min(int(max_bin), num_bins)
    if lower_bin >= upper_bin:
        raise ValueError("No range bins remain after applying min_bin/max_bin")

    frequencies_hz = np.fft.rfftfreq(matrix.shape[0], d=frame_period_s)
    respiration_min_hz, respiration_max_hz = respiration_band_hz
    respiration_mask = (frequencies_hz >= respiration_min_hz) & (frequencies_hz <= respiration_max_hz)
    non_dc_mask = frequencies_hz > 0.0
    if not np.any(respiration_mask):
        raise ValueError("No FFT bins fall within the respiration scoring band")

    candidate_matrix = matrix[:, lower_bin:upper_bin]
    bin_stds = np.std(candidate_matrix, axis=0)
    median_std = float(np.median(bin_stds)) if bin_stds.size else 0.0
    amplitude_floor = median_std if median_std > 0.0 else 1.0
    window = np.hanning(matrix.shape[0]) if matrix.shape[0] > 1 else np.ones(matrix.shape[0])

    rows: list[dict[str, float | int]] = []
    for relative_index, bin_index in enumerate(range(lower_bin, upper_bin)):
        samples = candidate_matrix[:, relative_index]
        centered = samples - np.mean(samples)
        spectrum = np.fft.rfft(centered * window)
        power = np.abs(spectrum) ** 2
        total_power = float(np.sum(power[non_dc_mask]))
        respiration_power = float(np.sum(power[respiration_mask]))
        band_ratio = respiration_power / total_power if total_power > 0.0 else 0.0
        std_value = float(bin_stds[relative_index])
        amplitude_weight = float(np.sqrt(max(std_value, 0.0) / amplitude_floor)) if amplitude_floor > 0.0 else 0.0
        score = band_ratio * amplitude_weight
        rows.append(
            {
                "bin_index": int(bin_index),
                "range_m": float(matrix_data["range_axis_m"][bin_index]),
                "mean": float(np.mean(samples)),
                "std": std_value,
                "respiration_band_power": respiration_power,
                "total_power": total_power,
                "respiration_band_ratio": float(band_ratio),
                "amplitude_weight": amplitude_weight,
                "score": float(score),
            }
        )

    top_bins = sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:top_k]
    return {
        "num_bins": num_bins,
        "frame_period_s": frame_period_s,
        "range_resolution_m": matrix_data["range_resolution_m"],
        "respiration_band_hz": tuple(respiration_band_hz),
        "bin_scores": rows,
        "top_respiration_bins": top_bins,
    }


def _group_candidate_bins(
    candidate_rows: list[dict[str, float | int]],
    *,
    max_gap: int = 2,
) -> list[list[dict[str, float | int]]]:
    sorted_rows = sorted(candidate_rows, key=lambda row: int(row["bin_index"]))
    if not sorted_rows:
        return []

    clusters: list[list[dict[str, float | int]]] = []
    current_cluster = [sorted_rows[0]]
    for row in sorted_rows[1:]:
        previous_bin = int(current_cluster[-1]["bin_index"])
        current_bin = int(row["bin_index"])
        if current_bin - previous_bin <= max_gap:
            current_cluster.append(row)
            continue
        clusters.append(current_cluster)
        current_cluster = [row]

    clusters.append(current_cluster)
    return clusters


def select_respiration_center_bin(
    bundle: CaptureBundle,
    *,
    min_bin: int = 3,
    max_bin: int | None = None,
    top_k: int = 12,
    cluster_max_gap: int = 2,
    selection_scoring: str = "respiration-band",
    respiration_band_hz: tuple[float, float] = RESPIRATION_BAND_HZ,
) -> dict[str, Any]:
    bin_scores: list[dict[str, float | int]] = []
    selection_method = selection_scoring
    selection_reason = ""
    score_key = "std"

    if selection_scoring in {"respiration-band", "respiration_band", "band", "respiration-band-trend", "trend"}:
        try:
            band_summary = score_respiration_range_bins(
                bundle,
                min_bin=min_bin,
                max_bin=max_bin,
                respiration_band_hz=respiration_band_hz,
                top_k=top_k,
            )
            candidate_rows = list(band_summary["top_respiration_bins"])
            bin_scores = list(band_summary["bin_scores"])
            score_key = "score"
            selection_method = "auto-respiration-band"
            selection_reason = (
                f"Selected from bins with concentrated {respiration_band_hz[0]:.2f}-"
                f"{respiration_band_hz[1]:.2f} Hz energy, weighted by signal variability."
            )
            if not candidate_rows or float(candidate_rows[0].get("score", 0.0)) <= 0.0:
                raise ValueError("respiration-band scores were not positive")
        except ValueError as exc:
            candidate_rows = summarize_range_bins(bundle, min_bin=min_bin, max_bin=max_bin, top_k=top_k)[
                "top_variable_bins"
            ]
            selection_method = "auto-cluster"
            selection_reason = f"Respiration-band scoring was unavailable ({exc}); fell back to variability clustering."
            score_key = "std"
    else:
        candidate_rows = summarize_range_bins(bundle, min_bin=min_bin, max_bin=max_bin, top_k=top_k)["top_variable_bins"]
        selection_method = "auto-cluster"
        selection_reason = "Selected from the most variable contiguous range-bin cluster."

    if not candidate_rows:
        raise ValueError("No candidate range bins available for signal extraction")

    clusters = _group_candidate_bins(candidate_rows, max_gap=cluster_max_gap)
    best_cluster = max(clusters, key=lambda cluster: sum(float(row[score_key]) for row in cluster))

    cluster_weight = sum(float(row[score_key]) for row in best_cluster)
    if cluster_weight <= 0.0:
        cluster_weight = sum(float(row["std"]) for row in best_cluster)
        score_key = "std"
    if cluster_weight <= 0.0:
        cluster_weight = float(len(best_cluster))
        score_key = "bin_index"
    weighted_center_bin = int(
        round(
            sum(
                int(row["bin_index"]) * (1.0 if score_key == "bin_index" else float(row[score_key]))
                for row in best_cluster
            )
            / cluster_weight
        )
    )
    center_row = min(
        best_cluster,
        key=lambda row: (
            abs(int(row["bin_index"]) - weighted_center_bin),
            -(1.0 if score_key == "bin_index" else float(row[score_key])),
        ),
    )

    return {
        "center_bin": int(center_row["bin_index"]),
        "center_range_m": float(center_row["range_m"]),
        "candidate_bins": [int(row["bin_index"]) for row in candidate_rows],
        "cluster_bins": [int(row["bin_index"]) for row in best_cluster],
        "cluster_score": cluster_weight,
        "weighted_center_bin": weighted_center_bin,
        "bin_scores": bin_scores,
        "selection_method": selection_method,
        "selection_reason": selection_reason,
        "selection_score_key": score_key,
    }


def estimate_range_motion_from_profiles(
    matrix_data: dict[str, Any],
    *,
    min_range_m: float = 0.5,
    max_range_m: float = 3.2,
    motion_threshold_m: float = 0.50,
    smoothing_seconds: float = 1.0,
) -> dict[str, Any]:
    """Estimate whether the dominant moving range-profile ridge migrates.

    This is a gross-motion diagnostic, not a respiration signal extractor. It
    subtracts the per-bin median range profile so static clutter does not win,
    then tracks the strongest positive residual ridge. Large range migration
    means a fixed range gate is not a valid assumption for the whole capture.
    """
    matrix = np.asarray(matrix_data["profiles"], dtype=float)
    range_axis = np.asarray(matrix_data["range_axis_m"], dtype=float)
    frame_period_s = float(matrix_data["frame_period_s"])
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3 or frame_period_s <= 0.0:
        return {
            "range_motion_detected": False,
            "range_motion_reason": "Range-motion diagnostic unavailable.",
        }

    keep_mask = (range_axis >= float(min_range_m)) & (range_axis <= float(max_range_m))
    candidate_bins = np.flatnonzero(keep_mask)
    if candidate_bins.size < 3:
        return {
            "range_motion_detected": False,
            "range_motion_reason": "No range bins available for range-motion diagnostic.",
        }

    candidate_matrix = matrix[:, candidate_bins]
    per_bin_median = np.median(candidate_matrix, axis=0)
    per_bin_mad = np.median(np.abs(candidate_matrix - per_bin_median), axis=0)
    residual_score = (candidate_matrix - per_bin_median) / np.maximum(per_bin_mad, 1e-6)
    residual_score = np.maximum(residual_score, 0.0)
    if residual_score.shape[1] >= 3:
        smoothed_score = residual_score.copy()
        smoothed_score[:, 1:-1] = (
            (0.20 * residual_score[:, :-2])
            + (0.60 * residual_score[:, 1:-1])
            + (0.20 * residual_score[:, 2:])
        )
        residual_score = smoothed_score

    tracked_bins = candidate_bins[np.argmax(residual_score, axis=1)]
    smooth_frames = max(1, int(round(float(smoothing_seconds) / frame_period_s)))
    if smooth_frames > 1:
        if smooth_frames % 2 == 0:
            smooth_frames += 1
        half_width = smooth_frames // 2
        tracked_bins = np.asarray(
            [
                int(np.median(tracked_bins[max(0, index - half_width) : min(tracked_bins.size, index + half_width + 1)]))
                for index in range(tracked_bins.size)
            ],
            dtype=int,
        )

    tracked_ranges = range_axis[tracked_bins]
    n = tracked_ranges.size
    early_range = float(np.median(tracked_ranges[: max(1, n // 3)]))
    mid_range = float(np.median(tracked_ranges[n // 3 : max(n // 3 + 1, 2 * n // 3)]))
    late_range = float(np.median(tracked_ranges[2 * n // 3 :]))
    p10_range = float(np.percentile(tracked_ranges, 10))
    p90_range = float(np.percentile(tracked_ranges, 90))
    total_span = float(p90_range - p10_range)
    early_late_change = float(late_range - early_range)
    max_change = float(abs(early_late_change))
    motion_detected = bool(max_change >= float(motion_threshold_m))

    preview_step = max(1, tracked_ranges.size // 120)
    preview_indices = np.arange(0, tracked_ranges.size, preview_step, dtype=int)
    times_s = list(matrix_data.get("times_s", []))
    preview_times = [float(times_s[index]) for index in preview_indices] if times_s else [float(index * frame_period_s) for index in preview_indices]
    preview_ranges = [float(tracked_ranges[index]) for index in preview_indices]
    reason = (
        f"Tracked residual range changed by {abs(early_late_change):.2f} m "
        f"({early_range:.2f} m early to {late_range:.2f} m late); "
        f"10th-90th percentile span is {total_span:.2f} m."
    )
    if not motion_detected:
        reason = "No large target range migration was detected. " + reason

    return {
        "range_motion_detected": motion_detected,
        "tracked_range_early_m": early_range,
        "tracked_range_mid_m": mid_range,
        "tracked_range_late_m": late_range,
        "tracked_range_change_m": early_late_change,
        "tracked_range_abs_change_m": abs(early_late_change),
        "tracked_range_p10_m": p10_range,
        "tracked_range_p90_m": p90_range,
        "tracked_range_span_m": total_span,
        "tracked_range_motion_score_m": max_change,
        "tracked_range_preview_times_s": preview_times,
        "tracked_range_preview_m": preview_ranges,
        "range_motion_reason": reason,
    }


def _median_smooth_int(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.size == 0 or window_size <= 1:
        return values.astype(int)
    if window_size % 2 == 0:
        window_size += 1
    half_width = window_size // 2
    return np.asarray(
        [
            int(np.median(values[max(0, index - half_width) : min(values.size, index + half_width + 1)]))
            for index in range(values.size)
        ],
        dtype=int,
    )


def _track_residual_range_path(
    matrix_data: dict[str, Any],
    *,
    min_range_m: float = 0.8,
    max_range_m: float = 3.0,
    tracking_window_seconds: float = 2.0,
    tracking_hop_seconds: float = 0.5,
    max_jump_bins: int = 4,
    jump_penalty: float = 0.4,
    smoothing_seconds: float = 1.0,
) -> dict[str, Any]:
    matrix = np.asarray(matrix_data["profiles"], dtype=float)
    range_axis = np.asarray(matrix_data["range_axis_m"], dtype=float)
    frame_period_s = float(matrix_data["frame_period_s"])
    if matrix.ndim != 2 or matrix.shape[0] < 8 or matrix.shape[1] < 3 or frame_period_s <= 0.0:
        raise ValueError("Range profile matrix is too small for dynamic range tracking")

    keep_mask = (range_axis >= float(min_range_m)) & (range_axis <= float(max_range_m))
    candidate_bins = np.flatnonzero(keep_mask)
    if candidate_bins.size < 3:
        raise ValueError("No range bins available for dynamic range tracking")

    candidate_matrix = matrix[:, candidate_bins]
    per_bin_median = np.median(candidate_matrix, axis=0)
    per_bin_mad = np.median(np.abs(candidate_matrix - per_bin_median), axis=0)
    residual_score = (candidate_matrix - per_bin_median) / np.maximum(per_bin_mad, 1e-6)
    residual_score = np.maximum(residual_score, 0.0)
    if residual_score.shape[1] >= 3:
        smoothed_score = residual_score.copy()
        smoothed_score[:, 1:-1] = (
            (0.20 * residual_score[:, :-2])
            + (0.60 * residual_score[:, 1:-1])
            + (0.20 * residual_score[:, 2:])
        )
        residual_score = smoothed_score

    frame_count = matrix.shape[0]
    window_size = max(3, int(round(float(tracking_window_seconds) / frame_period_s)))
    hop_size = max(1, int(round(float(tracking_hop_seconds) / frame_period_s)))
    starts = list(range(0, max(frame_count - window_size + 1, 1), hop_size))
    if not starts:
        starts = [0]
    final_start = max(0, frame_count - window_size)
    if starts[-1] != final_start:
        starts.append(final_start)

    window_scores: list[np.ndarray] = []
    window_centers: list[float] = []
    for start_index in starts:
        segment = residual_score[start_index : start_index + window_size]
        score = np.percentile(segment, 80, axis=0) + (0.20 * np.mean(segment, axis=0))
        score = (score - np.median(score)) / (np.median(np.abs(score - np.median(score))) + 1e-6)
        window_scores.append(score)
        window_centers.append(float(start_index + (window_size / 2.0)))

    score_matrix = np.asarray(window_scores, dtype=float)
    num_windows, num_bins = score_matrix.shape
    dynamic_score = np.full((num_windows, num_bins), -1e18, dtype=float)
    backtrack = np.zeros((num_windows, num_bins), dtype=int)
    dynamic_score[0] = score_matrix[0]
    max_jump_bins = max(1, int(max_jump_bins))
    for window_index in range(1, num_windows):
        for bin_index in range(num_bins):
            lower = max(0, bin_index - max_jump_bins)
            upper = min(num_bins, bin_index + max_jump_bins + 1)
            previous_indices = np.arange(lower, upper)
            previous_scores = dynamic_score[window_index - 1, lower:upper] - (
                float(jump_penalty) * np.abs(previous_indices - bin_index)
            )
            best_previous = int(np.argmax(previous_scores))
            dynamic_score[window_index, bin_index] = score_matrix[window_index, bin_index] + previous_scores[best_previous]
            backtrack[window_index, bin_index] = lower + best_previous

    bin_index = int(np.argmax(dynamic_score[-1]))
    path_indices = [bin_index]
    for window_index in range(num_windows - 1, 0, -1):
        bin_index = int(backtrack[window_index, bin_index])
        path_indices.append(bin_index)
    path_indices = list(reversed(path_indices))
    window_path_bins = candidate_bins[np.asarray(path_indices, dtype=int)]

    frame_indices = np.arange(frame_count, dtype=float)
    frame_path_bins = np.rint(np.interp(frame_indices, np.asarray(window_centers), window_path_bins)).astype(int)
    smooth_frames = max(1, int(round(float(smoothing_seconds) / frame_period_s)))
    frame_path_bins = _median_smooth_int(frame_path_bins, smooth_frames)
    frame_path_bins = np.clip(frame_path_bins, int(candidate_bins[0]), int(candidate_bins[-1]))
    frame_path_ranges = range_axis[frame_path_bins]

    return {
        "tracked_center_bins": frame_path_bins,
        "tracked_center_ranges_m": frame_path_ranges,
        "candidate_bins": candidate_bins,
        "per_bin_median": per_bin_median,
        "per_bin_mad": per_bin_mad,
    }


def _build_dynamic_range_tracked_signal(
    matrix_data: dict[str, Any],
    range_motion: dict[str, Any],
    *,
    bin_half_width: int = 1,
    min_range_m: float = 0.8,
    max_range_m: float = 3.0,
    detrend_seconds: float = 5.0,
) -> dict[str, Any]:
    matrix = np.asarray(matrix_data["profiles"], dtype=float)
    range_axis = np.asarray(matrix_data["range_axis_m"], dtype=float)
    frame_period_s = float(matrix_data["frame_period_s"])
    path = _track_residual_range_path(matrix_data, min_range_m=min_range_m, max_range_m=max_range_m)
    tracked_bins = np.asarray(path["tracked_center_bins"], dtype=int)
    tracked_ranges = np.asarray(path["tracked_center_ranges_m"], dtype=float)

    # For moving subjects, the range trajectory often preserves the breathing
    # oscillation better than the amplitude at a fixed bin. Remove the gross
    # distance trend and use the residual range wobble as the respiration signal.
    trend = _moving_average(tracked_ranges, max(3, int(round(float(detrend_seconds) / frame_period_s))))
    centered_signal = tracked_ranges - trend
    centered_std = float(np.std(centered_signal))
    zscore_signal = centered_signal / centered_std if centered_std > 0.0 else np.zeros_like(centered_signal)

    representative_bin = int(np.median(tracked_bins))
    lower = max(0, representative_bin - int(bin_half_width))
    upper = min(matrix.shape[1], representative_bin + int(bin_half_width) + 1)
    selected_bins = list(range(lower, upper))
    selected_ranges = [float(range_axis[index]) for index in selected_bins]
    n = tracked_ranges.size
    early_range = float(np.median(tracked_ranges[: max(1, n // 3)]))
    mid_range = float(np.median(tracked_ranges[n // 3 : max(n // 3 + 1, 2 * n // 3)]))
    late_range = float(np.median(tracked_ranges[2 * n // 3 :]))
    p10_range = float(np.percentile(tracked_ranges, 10))
    p90_range = float(np.percentile(tracked_ranges, 90))
    range_change = float(late_range - early_range)

    return {
        "center_bin": representative_bin,
        "center_range_m": float(np.median(tracked_ranges)),
        "selected_bins": selected_bins,
        "selected_ranges_m": selected_ranges,
        "times_s": matrix_data["times_s"],
        "frame_numbers": matrix_data["frame_numbers"],
        "raw_signal": tracked_ranges.tolist(),
        "centered_signal": centered_signal.tolist(),
        "zscore_signal": zscore_signal.tolist(),
        "frame_period_s": frame_period_s,
        "range_resolution_m": matrix_data["range_resolution_m"],
        "selection_method": "dynamic-range-tracking",
        "selection_candidate_bins": [int(index) for index in path["candidate_bins"]],
        "selection_cluster_bins": [],
        "selection_cluster_score": None,
        "selection_weighted_center_bin": representative_bin,
        "selection_bin_scores": [],
        "selection_reason": (
            "Large subject range motion was detected, so respiration was extracted from the detrended "
            "tracked range ridge instead of one fixed range bin."
        ),
        "selection_score_key": "tracked_range",
        "selection_trend_candidate_scores": [],
        "selection_trend_selected": None,
        "range_motion_detected": bool(range_motion.get("range_motion_detected", True)),
        "range_motion_reason": range_motion.get("range_motion_reason", ""),
        "tracked_range_early_m": early_range,
        "tracked_range_mid_m": mid_range,
        "tracked_range_late_m": late_range,
        "tracked_range_change_m": range_change,
        "tracked_range_abs_change_m": abs(range_change),
        "tracked_range_p10_m": p10_range,
        "tracked_range_p90_m": p90_range,
        "tracked_range_span_m": float(p90_range - p10_range),
        "tracked_range_motion_score_m": abs(range_change),
        "tracked_range_preview_times_s": list(range_motion.get("tracked_range_preview_times_s", [])),
        "tracked_range_preview_m": list(range_motion.get("tracked_range_preview_m", [])),
        "tracked_center_bins": tracked_bins.astype(int).tolist(),
        "tracked_center_ranges_m": tracked_ranges.tolist(),
        "selected_range_matches_tracked_target": True,
        "dynamic_range_tracking_used": True,
        "dynamic_range_detrend_seconds": float(detrend_seconds),
    }


def extract_respiration_signal(
    bundle: CaptureBundle,
    *,
    center_bin: int | None = None,
    center_range_m: float | None = None,
    bin_half_width: int = 1,
    min_bin: int = 3,
    max_bin: int | None = None,
    selection_top_k: int = 12,
    selection_cluster_max_gap: int = 2,
    selection_scoring: str = "respiration-band-trend",
    respiration_band_hz: tuple[float, float] = RESPIRATION_BAND_HZ,
    trend_selection_min_bpm: float = 4.0,
    trend_selection_max_bpm: float = 30.0,
) -> dict[str, Any]:
    """Extract one respiration-ready time series from a parsed capture.

    The extractor first decides whether a fixed range gate is still valid for
    the whole recording. If large target migration is detected it tries a
    dynamic tracked-range signal; otherwise it chooses a stable gate from the
    most respiration-like bins.
    """

    matrix_data = build_range_profile_matrix(bundle)
    range_axis = matrix_data["range_axis_m"]
    profiles = matrix_data["profiles"]
    num_bins = len(range_axis)
    range_motion = estimate_range_motion_from_profiles(matrix_data)

    selection_method = "manual-bin" if center_bin is not None else "manual-range" if center_range_m is not None else "auto-cluster"
    selection_info: dict[str, Any] | None = None

    if center_bin is None:
        if center_range_m is not None:
            center_bin = min(num_bins - 1, max(0, round(center_range_m / matrix_data["range_resolution_m"])))
        else:
            # Large target migration invalidates a single fixed gate, so try a
            # tracked-range signal before falling back to static bin selection.
            if range_motion.get("range_motion_detected"):
                try:
                    return _build_dynamic_range_tracked_signal(
                        matrix_data,
                        range_motion,
                        bin_half_width=bin_half_width,
                    )
                except ValueError:
                    pass
            selection_info = select_respiration_center_bin(
                bundle,
                min_bin=min_bin,
                max_bin=max_bin,
                top_k=selection_top_k,
                cluster_max_gap=selection_cluster_max_gap,
                selection_scoring=selection_scoring,
                respiration_band_hz=respiration_band_hz,
            )
            center_bin = int(selection_info["center_bin"])
            selection_method = selection_info.get("selection_method", selection_method)

            if selection_scoring in {"respiration-band-trend", "trend"}:
                # Nearby bins can have similar spectral scores. Re-score those
                # candidates by asking which extracted signal produces the most
                # plausible stable or transitioning breathing pattern.
                bin_score_by_index = {
                    int(row["bin_index"]): float(row.get("score", 0.0))
                    for row in selection_info.get("bin_scores", [])
                }
                max_candidate_bin_score = max(
                    [bin_score_by_index.get(bin_index, 0.0) for bin_index in selection_info.get("cluster_bins", [])]
                    or [1.0]
                )
                candidate_centers = sorted(
                    {
                        bin_index
                        for cluster_bin in selection_info.get("cluster_bins", [])
                        for bin_index in (cluster_bin - 1, cluster_bin, cluster_bin + 1)
                        if 0 <= bin_index < num_bins
                    }
                )
                trend_rows: list[dict[str, Any]] = []
                for candidate_center in candidate_centers:
                    candidate_lower = max(0, candidate_center - bin_half_width)
                    candidate_upper = min(num_bins, candidate_center + bin_half_width + 1)
                    candidate_bins = list(range(candidate_lower, candidate_upper))
                    candidate_raw = [mean([profile[index] for index in candidate_bins]) for profile in profiles]
                    candidate_mean = mean(candidate_raw) if candidate_raw else 0.0
                    candidate_centered = [value - candidate_mean for value in candidate_raw]
                    candidate_std = pstdev(candidate_centered) if len(candidate_centered) > 1 else 0.0
                    candidate_zscore = [
                        value / candidate_std if candidate_std else 0.0 for value in candidate_centered
                    ]
                    candidate_signal = {
                        "center_bin": candidate_center,
                        "center_range_m": range_axis[candidate_center],
                        "selected_bins": candidate_bins,
                        "selected_ranges_m": [range_axis[index] for index in candidate_bins],
                        "times_s": matrix_data["times_s"],
                        "frame_numbers": matrix_data["frame_numbers"],
                        "raw_signal": candidate_raw,
                        "centered_signal": candidate_centered,
                        "zscore_signal": candidate_zscore,
                        "frame_period_s": matrix_data["frame_period_s"],
                        "range_resolution_m": matrix_data["range_resolution_m"],
                    }
                    try:
                        candidate_global = estimate_breathing_rate_multiwindow(
                            candidate_signal,
                            min_bpm=trend_selection_min_bpm,
                            max_bpm=trend_selection_max_bpm,
                        )
                        candidate_trend = estimate_respiration_trend(
                            candidate_signal,
                            min_bpm=trend_selection_min_bpm,
                            max_bpm=trend_selection_max_bpm,
                            global_bpm=candidate_global.get("selected_bpm"),
                            global_confidence=candidate_global.get("final_confidence"),
                            raw_spectral_bpm=candidate_global.get("raw_spectral_bpm"),
                        )
                    except ValueError:
                        continue
                    early_bpm = candidate_trend.get("early_bpm")
                    late_bpm = candidate_trend.get("late_bpm")
                    candidate_global_bpm = candidate_global.get("selected_bpm")
                    candidate_global_confidence = float(candidate_global.get("final_confidence") or 0.0)
                    change = float(candidate_trend.get("bpm_change_magnitude") or 0.0)
                    in_selection_band = (
                        early_bpm is not None
                        and late_bpm is not None
                        and (trend_selection_min_bpm + 1.0)
                        <= float(early_bpm)
                        <= (trend_selection_max_bpm - 1.0)
                        and (trend_selection_min_bpm + 1.0)
                        <= float(late_bpm)
                        <= (trend_selection_max_bpm - 1.0)
                    )
                    global_in_selection_band = (
                        candidate_global_bpm is not None
                        and (trend_selection_min_bpm + 1.0)
                        <= float(candidate_global_bpm)
                        <= (trend_selection_max_bpm - 1.0)
                    )
                    transition_score = (
                        float(candidate_trend.get("trend_confidence") or 0.0)
                        if candidate_trend.get("transition_detected") and in_selection_band
                        else 0.0
                    )
                    physiological_score = 0.0
                    if early_bpm is not None and late_bpm is not None:
                        physiological_score = 1.0 if 6.0 <= float(early_bpm) <= 24.0 and 6.0 <= float(late_bpm) <= 24.0 else 0.0
                    band_score = bin_score_by_index.get(candidate_center, 0.0)
                    normalized_band_score = band_score / max_candidate_bin_score if max_candidate_bin_score > 0.0 else 0.0
                    trend_rows.append(
                        {
                            "center_bin": candidate_center,
                            "center_range_m": float(range_axis[candidate_center]),
                            "transition_detected": bool(candidate_trend.get("transition_detected")),
                            "trend_confidence": float(candidate_trend.get("trend_confidence") or 0.0),
                            "early_bpm": early_bpm,
                            "late_bpm": late_bpm,
                            "bpm_change_magnitude": change,
                            "in_selection_band": in_selection_band,
                            "global_bpm": candidate_global_bpm,
                            "raw_spectral_bpm": candidate_global.get("raw_spectral_bpm"),
                            "global_confidence": candidate_global_confidence,
                            "global_in_selection_band": global_in_selection_band,
                            "global_harmonic_suspected": bool(candidate_global.get("harmonic_suspected")),
                            "global_is_ambiguous": bool(candidate_global.get("is_ambiguous")),
                            "respiration_band_score": band_score,
                            "normalized_respiration_band_score": normalized_band_score,
                            "score": transition_score + (0.25 * normalized_band_score) + (0.10 * physiological_score),
                            "nonstationary_reason": candidate_trend.get("nonstationary_reason"),
                        }
                    )

                transition_rows = [
                    row
                    for row in trend_rows
                    if row["transition_detected"]
                    and row["in_selection_band"]
                    and row["trend_confidence"] >= 0.55
                    and row["bpm_change_magnitude"] >= 4.0
                    and row["normalized_respiration_band_score"] >= 0.25
                ]
                if transition_rows:
                    # Prefer a consistent transition direction across nearby
                    # candidates so one noisy bin does not flip the reported
                    # early/late story for the whole capture.
                    increasing_score = sum(
                        float(row["score"])
                        for row in transition_rows
                        if row.get("late_bpm") is not None
                        and row.get("early_bpm") is not None
                        and float(row["late_bpm"]) >= float(row["early_bpm"])
                    )
                    decreasing_score = sum(
                        float(row["score"])
                        for row in transition_rows
                        if row.get("late_bpm") is not None
                        and row.get("early_bpm") is not None
                        and float(row["late_bpm"]) < float(row["early_bpm"])
                    )
                    if increasing_score > 0.0 or decreasing_score > 0.0:
                        prefer_increasing = increasing_score >= decreasing_score
                        consensus_rows = [
                            row
                            for row in transition_rows
                            if row.get("late_bpm") is not None
                            and row.get("early_bpm") is not None
                            and ((float(row["late_bpm"]) >= float(row["early_bpm"])) == prefer_increasing)
                        ]
                    else:
                        consensus_rows = transition_rows
                    best_trend_row = max(
                        consensus_rows,
                        key=lambda row: (
                            float(row["score"]),
                            -abs(int(row["center_bin"]) - int(selection_info["center_bin"])),
                        ),
                    )
                    center_bin = int(best_trend_row["center_bin"])
                    selection_method = "auto-respiration-band-trend"
                    selection_info["selection_method"] = selection_method
                    selection_info["selection_reason"] = (
                        "Selected a respiration-band candidate whose early/late segment estimates "
                        "show the clearest sustained rate transition."
                    )
                    selection_info["trend_candidate_scores"] = trend_rows
                    selection_info["trend_selected"] = best_trend_row
                    selection_info["trend_direction_consensus"] = {
                        "increasing_score": increasing_score,
                        "decreasing_score": decreasing_score,
                    }
                elif trend_rows:
                    selection_info["trend_candidate_scores"] = trend_rows
                    stable_rows = [
                        row
                        for row in trend_rows
                        if not row["transition_detected"]
                        and row.get("global_in_selection_band")
                        and row["normalized_respiration_band_score"] >= 0.35
                        and float(row.get("global_confidence") or 0.0) >= 0.45
                    ]
                    if stable_rows:
                        def _stable_selection_score(row: dict[str, Any]) -> tuple[float, float]:
                            harmonic_penalty = 0.08 if row.get("global_is_ambiguous") else 0.0
                            score = (
                                (0.45 * float(row.get("normalized_respiration_band_score") or 0.0))
                                + (0.35 * float(row.get("global_confidence") or 0.0))
                                + 0.20
                                - harmonic_penalty
                            )
                            return score, -abs(int(row["center_bin"]) - int(selection_info["center_bin"]))

                        best_stable_row = max(stable_rows, key=_stable_selection_score)
                        center_bin = int(best_stable_row["center_bin"])
                        selection_method = "auto-respiration-band-trend-stable"
                        selection_info["selection_method"] = selection_method
                        selection_info["selection_reason"] = (
                            "Selected a respiration-band candidate with a plausible stable full-record "
                            "respiration estimate after transition candidates were rejected."
                        )
                        selection_info["trend_selected"] = best_stable_row

    lower = max(0, center_bin - bin_half_width)
    upper = min(num_bins, center_bin + bin_half_width + 1)
    selected_bins = list(range(lower, upper))
    selected_ranges_m = [range_axis[index] for index in selected_bins]
    selected_range_m = float(range_axis[center_bin])
    tracked_p10 = range_motion.get("tracked_range_p10_m")
    tracked_p90 = range_motion.get("tracked_range_p90_m")
    selected_range_matches_tracked_target = True
    if tracked_p10 is not None and tracked_p90 is not None:
        selected_range_matches_tracked_target = bool(
            (float(tracked_p10) - 0.35) <= selected_range_m <= (float(tracked_p90) + 0.35)
        )

    raw_signal = [mean([profile[index] for index in selected_bins]) for profile in profiles]
    raw_mean = mean(raw_signal) if raw_signal else 0.0
    centered_signal = [value - raw_mean for value in raw_signal]
    centered_std = pstdev(centered_signal) if len(centered_signal) > 1 else 0.0
    zscore_signal = [value / centered_std if centered_std else 0.0 for value in centered_signal]

    return {
        "center_bin": center_bin,
        "center_range_m": selected_range_m,
        "selected_bins": selected_bins,
        "selected_ranges_m": selected_ranges_m,
        "times_s": matrix_data["times_s"],
        "frame_numbers": matrix_data["frame_numbers"],
        "raw_signal": raw_signal,
        "centered_signal": centered_signal,
        "zscore_signal": zscore_signal,
        "frame_period_s": matrix_data["frame_period_s"],
        "range_resolution_m": matrix_data["range_resolution_m"],
        "selection_method": selection_method,
        "selection_candidate_bins": [] if selection_info is None else selection_info["candidate_bins"],
        "selection_cluster_bins": [] if selection_info is None else selection_info["cluster_bins"],
        "selection_cluster_score": None if selection_info is None else selection_info["cluster_score"],
        "selection_weighted_center_bin": None if selection_info is None else selection_info["weighted_center_bin"],
        "selection_bin_scores": [] if selection_info is None else selection_info.get("bin_scores", []),
        "selection_reason": "" if selection_info is None else selection_info.get("selection_reason", ""),
        "selection_score_key": None if selection_info is None else selection_info.get("selection_score_key"),
        "selection_trend_candidate_scores": []
        if selection_info is None
        else selection_info.get("trend_candidate_scores", []),
        "selection_trend_selected": None if selection_info is None else selection_info.get("trend_selected"),
        "range_motion_detected": bool(range_motion.get("range_motion_detected", False)),
        "range_motion_reason": range_motion.get("range_motion_reason", ""),
        "tracked_range_early_m": range_motion.get("tracked_range_early_m"),
        "tracked_range_mid_m": range_motion.get("tracked_range_mid_m"),
        "tracked_range_late_m": range_motion.get("tracked_range_late_m"),
        "tracked_range_change_m": range_motion.get("tracked_range_change_m"),
        "tracked_range_abs_change_m": range_motion.get("tracked_range_abs_change_m"),
        "tracked_range_p10_m": range_motion.get("tracked_range_p10_m"),
        "tracked_range_p90_m": range_motion.get("tracked_range_p90_m"),
        "tracked_range_span_m": range_motion.get("tracked_range_span_m"),
        "tracked_range_motion_score_m": range_motion.get("tracked_range_motion_score_m"),
        "tracked_range_preview_times_s": range_motion.get("tracked_range_preview_times_s", []),
        "tracked_range_preview_m": range_motion.get("tracked_range_preview_m", []),
        "selected_range_matches_tracked_target": selected_range_matches_tracked_target,
    }


def _select_peak_index(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    *,
    near_max_ratio: float = 0.97,
) -> int:
    peak_index = int(np.argmax(power))
    max_power = float(power[peak_index])
    if max_power <= 0.0:
        return peak_index

    near_max_indices = np.flatnonzero(power >= (max_power * near_max_ratio))
    if near_max_indices.size <= 1:
        return peak_index

    near_max_frequencies = frequencies_hz[near_max_indices]
    return int(near_max_indices[int(np.argmin(near_max_frequencies))])


def _adjust_peak_index_for_subharmonic(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    peak_index: int,
    *,
    min_frequency_hz: float,
    subharmonic_power_ratio: float = 0.35,
    reference_frequency_hz: float | None = None,
) -> tuple[int, bool]:
    peak_frequency_hz = float(frequencies_hz[peak_index])
    if peak_frequency_hz < (2.0 * min_frequency_hz):
        return peak_index, False

    if frequencies_hz.size < 2:
        return peak_index, False

    frequency_step_hz = float(np.median(np.diff(frequencies_hz)))
    half_frequency_hz = 0.5 * peak_frequency_hz
    half_index = int(np.argmin(np.abs(frequencies_hz - half_frequency_hz)))

    if abs(float(frequencies_hz[half_index]) - half_frequency_hz) > (0.75 * frequency_step_hz):
        return peak_index, False

    if reference_frequency_hz is not None:
        raw_gap_hz = abs(peak_frequency_hz - reference_frequency_hz)
        half_gap_hz = abs(float(frequencies_hz[half_index]) - reference_frequency_hz)
        if raw_gap_hz <= (1.0 * frequency_step_hz) and half_gap_hz > (1.0 * frequency_step_hz):
            return peak_index, False

    peak_power = float(power[peak_index])
    half_power = float(power[half_index])
    if peak_power <= 0.0 or half_power < (peak_power * subharmonic_power_ratio):
        return peak_index, False

    return half_index, True


def _finite_centered_array(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return values
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        fill_value = float(np.mean(values[finite_mask])) if np.any(finite_mask) else 0.0
        values = np.where(finite_mask, values, fill_value)
    return values - np.mean(values)


def _detrend_linear(samples: np.ndarray) -> np.ndarray:
    values = _finite_centered_array(samples)
    if values.size < 3:
        return values
    sample_index = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(sample_index, values, 1)
    return values - ((slope * sample_index) + intercept)


def _as_signal_samples(signal: dict[str, Any], *, prefer_zscore: bool = False) -> tuple[np.ndarray, float]:
    key = "zscore_signal" if prefer_zscore and "zscore_signal" in signal else "centered_signal"
    samples = np.asarray(signal[key], dtype=float)
    frame_period_s = float(signal["frame_period_s"])
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if samples.size < 2:
        raise ValueError("Respiration signal is too short")
    return samples, frame_period_s


def _signal_duration_seconds(signal: dict[str, Any], samples: np.ndarray, frame_period_s: float) -> float:
    times_s = signal.get("times_s") or []
    if times_s:
        return float(times_s[-1] + frame_period_s)
    return float(samples.size * frame_period_s)


def _fft_bandpass_signal(
    samples: np.ndarray,
    frame_period_s: float,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> np.ndarray:
    values = _detrend_linear(samples)
    if values.size < 4:
        return values

    nyquist_hz = 0.5 / frame_period_s
    upper_hz = min(max_frequency_hz, nyquist_hz * 0.98)
    lower_hz = max(0.0, min_frequency_hz)
    if upper_hz <= lower_hz:
        return values

    frequencies_hz = np.fft.rfftfreq(values.size, d=frame_period_s)
    spectrum = np.fft.rfft(values)
    keep_mask = (frequencies_hz >= lower_hz) & (frequencies_hz <= upper_hz)
    spectrum[~keep_mask] = 0.0
    return np.fft.irfft(spectrum, n=values.size)


def _moving_average(samples: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or samples.size < 3:
        return samples
    window_size = min(int(window_size), samples.size)
    if window_size % 2 == 0:
        window_size = max(1, window_size - 1)
    if window_size <= 1:
        return samples
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    padded = np.pad(samples, (window_size // 2, window_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _bpm_close(candidate_bpm: float | None, target_bpm: float | None, *, tolerance_bpm: float | None = None) -> bool:
    if candidate_bpm is None or target_bpm is None or target_bpm <= 0.0:
        return False
    tolerance = max(1.5, 0.12 * target_bpm) if tolerance_bpm is None else tolerance_bpm
    return abs(float(candidate_bpm) - float(target_bpm)) <= tolerance


def _local_maxima_indices(samples: np.ndarray) -> np.ndarray:
    if samples.size < 3:
        return np.asarray([], dtype=int)
    return np.flatnonzero((samples[1:-1] >= samples[:-2]) & (samples[1:-1] > samples[2:])) + 1


def _estimate_prominences(samples: np.ndarray, indices: np.ndarray, span_samples: int) -> np.ndarray:
    prominences: list[float] = []
    span = max(1, int(span_samples))
    for index in indices:
        left = max(0, int(index) - span)
        right = min(samples.size, int(index) + span + 1)
        left_min = float(np.min(samples[left : int(index) + 1]))
        right_min = float(np.min(samples[int(index) : right]))
        baseline = max(left_min, right_min)
        prominences.append(max(0.0, float(samples[index]) - baseline))
    return np.asarray(prominences, dtype=float)


def _prune_extrema_by_distance(
    indices: np.ndarray,
    prominences: np.ndarray,
    min_distance_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return indices, prominences
    order = np.argsort(prominences)[::-1]
    selected: list[int] = []
    selected_prominences: list[float] = []
    for order_index in order:
        candidate = int(indices[order_index])
        if all(abs(candidate - existing) >= min_distance_samples for existing in selected):
            selected.append(candidate)
            selected_prominences.append(float(prominences[order_index]))
    sort_order = np.argsort(selected)
    return np.asarray(selected, dtype=int)[sort_order], np.asarray(selected_prominences, dtype=float)[sort_order]


def _build_extrema_cycle_candidates(
    samples: np.ndarray,
    *,
    kind: str,
    min_distance_samples: int,
) -> list[dict[str, Any]]:
    candidate_indices = _local_maxima_indices(samples)
    if candidate_indices.size == 0:
        return []
    candidate_prominences = _estimate_prominences(samples, candidate_indices, min_distance_samples)
    max_prominence = float(np.max(candidate_prominences)) if candidate_prominences.size else 0.0
    if max_prominence <= 0.0:
        return []

    amplitude_floor = max(0.05 * float(np.ptp(samples)), 0.20 * float(np.std(samples)))
    candidates: list[dict[str, Any]] = []
    for threshold_ratio, label in ((0.35, "major"), (0.60, "dominant")):
        threshold = max(amplitude_floor, threshold_ratio * max_prominence)
        keep_mask = candidate_prominences >= threshold
        indices, prominences = _prune_extrema_by_distance(
            candidate_indices[keep_mask],
            candidate_prominences[keep_mask],
            min_distance_samples,
        )
        if indices.size >= 2:
            candidates.append(
                {
                    "kind": kind,
                    "label": label,
                    "indices": indices,
                    "prominences": prominences,
                    "alternating_support": False,
                }
            )

    if candidates:
        base_indices = candidates[0]["indices"]
        base_prominences = candidates[0]["prominences"]
        if base_indices.size >= 5:
            even_prominence = float(np.median(base_prominences[0::2]))
            odd_prominence = float(np.median(base_prominences[1::2]))
            larger = max(even_prominence, odd_prominence)
            smaller = min(even_prominence, odd_prominence)
            if larger > 0.0 and (smaller / larger) < 0.75:
                parity = 0 if even_prominence >= odd_prominence else 1
                alternating_indices = base_indices[parity::2]
                alternating_prominences = base_prominences[parity::2]
                if alternating_indices.size >= 2:
                    candidates.append(
                        {
                            "kind": kind,
                            "label": "alternating-dominant",
                            "indices": alternating_indices,
                            "prominences": alternating_prominences,
                            "alternating_support": True,
                        }
                    )

    return candidates


def _evaluate_cycle_candidate(
    candidate: dict[str, Any],
    *,
    frame_period_s: float,
    duration_s: float,
    min_bpm: float,
    max_bpm: float,
) -> dict[str, Any]:
    indices = np.asarray(candidate["indices"], dtype=int)
    prominences = np.asarray(candidate["prominences"], dtype=float)
    intervals_s = np.diff(indices) * frame_period_s
    min_period_s = 60.0 / max_bpm
    max_period_s = 60.0 / min_bpm
    valid_intervals = intervals_s[(intervals_s >= 0.85 * min_period_s) & (intervals_s <= 1.15 * max_period_s)]
    if valid_intervals.size == 0:
        return {**candidate, "confidence": 0.0, "bpm": None}

    period_s = float(np.median(valid_intervals))
    bpm = 60.0 / period_s if period_s > 0.0 else None
    if bpm is None or bpm < min_bpm or bpm > max_bpm:
        return {**candidate, "confidence": 0.0, "bpm": None}

    q25, q75 = np.percentile(valid_intervals, [25, 75]) if valid_intervals.size > 1 else (period_s, period_s)
    interval_iqr_s = float(q75 - q25)
    spread_ratio = interval_iqr_s / period_s if period_s > 0.0 else 1.0
    count_score = min(1.0, float(valid_intervals.size) / 4.0)
    spread_score = max(0.0, 1.0 - min(1.0, spread_ratio / 0.35))
    if prominences.size > 1 and float(np.mean(prominences)) > 0.0:
        prominence_cv = float(np.std(prominences) / np.mean(prominences))
    else:
        prominence_cv = 0.0
    prominence_score = max(0.0, 1.0 - min(1.0, prominence_cv / 0.80))
    coverage_s = float((indices[-1] - indices[0]) * frame_period_s) if indices.size > 1 else 0.0
    coverage_score = min(1.0, coverage_s / max(duration_s * 0.50, period_s))
    confidence = (0.35 * count_score) + (0.30 * spread_score) + (0.20 * prominence_score) + (0.15 * coverage_score)
    if candidate.get("alternating_support"):
        confidence = min(1.0, confidence + 0.10)

    quality_flags: list[str] = []
    if valid_intervals.size < 3:
        quality_flags.append("few time-domain cycles")
    if spread_ratio > 0.35:
        quality_flags.append("irregular time-domain intervals")
    if prominence_score < 0.35:
        quality_flags.append("uneven breath-cycle prominence")
    if candidate.get("alternating_support"):
        quality_flags.append("alternating peak prominence suggests harmonic")
        if valid_intervals.size < 3:
            confidence = min(confidence, 0.55)

    return {
        **candidate,
        "bpm": float(bpm),
        "period_s": period_s,
        "num_breaths_detected": int(indices.size),
        "intervals_s": valid_intervals.tolist(),
        "interval_median_s": period_s,
        "interval_iqr_s": interval_iqr_s,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "quality_flags": quality_flags,
    }


def estimate_time_domain_breathing_rate(
    signal: dict[str, Any],
    *,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
) -> dict[str, Any]:
    """Estimate full breath-cycle spacing from major peaks/troughs.

    This intentionally looks for repeated full-cycle extrema, not just the
    strongest frequency.  Alternating large/small extrema are treated as
    evidence that a bright 2x spectral component may be a harmonic.
    """
    samples, frame_period_s = _as_signal_samples(signal, prefer_zscore=True)
    duration_s = _signal_duration_seconds(signal, samples, frame_period_s)
    if samples.size < 8:
        return {
            "bpm": None,
            "period_s": None,
            "num_breaths_detected": 0,
            "intervals_s": [],
            "interval_median_s": None,
            "interval_iqr_s": None,
            "confidence": 0.0,
            "quality_flags": ["signal too short for time-domain cycle validation"],
            "decision_reason": "Signal is too short for time-domain cycle validation.",
        }

    min_frequency_hz = min_bpm / 60.0
    max_frequency_hz = max_bpm / 60.0
    filtered = _fft_bandpass_signal(
        samples,
        frame_period_s,
        min_frequency_hz=max(0.0, 0.50 * min_frequency_hz),
        max_frequency_hz=1.35 * max_frequency_hz,
    )
    smooth_window_s = min(1.0, max(0.15, (60.0 / max_bpm) / 4.0))
    smooth_window_samples = max(1, int(round(smooth_window_s / frame_period_s)))
    smoothed = _moving_average(filtered, smooth_window_samples)
    smoothed = _finite_centered_array(smoothed)
    smoothed_std = float(np.std(smoothed))
    if smoothed_std > 0.0:
        smoothed = smoothed / smoothed_std

    min_period_s = 60.0 / max_bpm
    min_distance_samples = max(1, int(round(0.65 * min_period_s / frame_period_s)))
    candidates: list[dict[str, Any]] = []
    for kind, series in (("peak", smoothed), ("trough", -smoothed)):
        for candidate in _build_extrema_cycle_candidates(
            series,
            kind=kind,
            min_distance_samples=min_distance_samples,
        ):
            candidates.append(
                _evaluate_cycle_candidate(
                    candidate,
                    frame_period_s=frame_period_s,
                    duration_s=duration_s,
                    min_bpm=min_bpm,
                    max_bpm=max_bpm,
                )
            )

    usable_candidates = [candidate for candidate in candidates if candidate.get("bpm") is not None]
    if not usable_candidates:
        return {
            "bpm": None,
            "period_s": None,
            "num_breaths_detected": 0,
            "intervals_s": [],
            "interval_median_s": None,
            "interval_iqr_s": None,
            "confidence": 0.0,
            "quality_flags": ["no stable time-domain breath cycles"],
            "decision_reason": "No stable sequence of major peaks or troughs was found in the requested BPM band.",
        }

    best = max(usable_candidates, key=lambda row: float(row.get("confidence", 0.0)))
    flags = list(best.get("quality_flags", []))
    reason = (
        f"Detected {best['num_breaths_detected']} major {best['kind']} extrema; "
        f"median full-cycle spacing is {best['period_s']:.2f} s "
        f"({best['bpm']:.2f} BPM)."
    )
    if best.get("alternating_support"):
        reason += " Alternating extrema amplitudes support treating every other extrema as the full breath cycle."

    return {
        "bpm": best["bpm"],
        "period_s": best["period_s"],
        "num_breaths_detected": best["num_breaths_detected"],
        "intervals_s": best["intervals_s"],
        "interval_median_s": best["interval_median_s"],
        "interval_iqr_s": best["interval_iqr_s"],
        "confidence": best["confidence"],
        "quality_flags": flags,
        "decision_reason": reason,
        "cycle_kind": best["kind"],
        "cycle_label": best["label"],
    }


def _estimate_autocorrelation_candidates(
    samples: np.ndarray,
    frame_period_s: float,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
    top_k: int = 5,
) -> list[dict[str, float | int]]:
    if samples.size < 16 or frame_period_s <= 0.0:
        return []

    centered_samples = _detrend_linear(samples)
    autocorrelation = np.correlate(centered_samples, centered_samples, mode="full")[centered_samples.size - 1 :]
    if autocorrelation.size < 2 or float(autocorrelation[0]) <= 0.0:
        return []

    autocorrelation = autocorrelation / float(autocorrelation[0])
    autocorrelation[0] = 0.0
    sample_rate_hz = 1.0 / frame_period_s
    min_lag = max(1, int(np.floor(sample_rate_hz / max_frequency_hz)))
    max_lag = min(autocorrelation.size - 1, int(np.ceil(sample_rate_hz / min_frequency_hz)))
    if min_lag > max_lag:
        return []

    lag_window = autocorrelation[min_lag : max_lag + 1]
    local_offsets = _local_maxima_indices(lag_window)
    if local_offsets.size == 0:
        local_offsets = np.asarray([int(np.argmax(lag_window))], dtype=int)

    candidates: list[dict[str, float | int]] = []
    for offset in local_offsets:
        lag = int(min_lag + offset)
        correlation = float(autocorrelation[lag])
        if correlation <= 0.0:
            continue
        frequency_hz = float(sample_rate_hz / lag)
        candidates.append(
            {
                "lag_samples": lag,
                "period_s": float(lag * frame_period_s),
                "frequency_hz": frequency_hz,
                "bpm": float(frequency_hz * 60.0),
                "correlation": correlation,
            }
        )

    return sorted(candidates, key=lambda row: float(row["correlation"]), reverse=True)[:top_k]


def _autocorrelation_candidate_near(
    candidates: list[dict[str, float | int]],
    target_bpm: float | None,
) -> dict[str, float | int] | None:
    if target_bpm is None:
        return None
    near_candidates = [candidate for candidate in candidates if _bpm_close(candidate.get("bpm"), target_bpm)]
    if not near_candidates:
        return None
    return max(near_candidates, key=lambda row: float(row.get("correlation") or 0.0))


def _estimate_autocorrelation_peak_frequency_hz(
    samples: np.ndarray,
    frame_period_s: float,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> float | None:
    candidates = _estimate_autocorrelation_candidates(
        samples,
        frame_period_s,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        top_k=1,
    )
    if not candidates:
        return None
    return float(candidates[0]["frequency_hz"])


def _estimate_spectral_rate_from_samples(
    samples: np.ndarray,
    frame_period_s: float,
    *,
    min_bpm: float,
    max_bpm: float,
    top_k: int = 5,
    source: str = "fft",
) -> dict[str, Any]:
    if frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if samples.size < 16:
        raise ValueError("Respiration signal is too short for frequency analysis")

    values = _detrend_linear(samples)
    window = np.hanning(values.size) if values.size > 1 else np.ones(values.size)
    frequencies_hz = np.fft.rfftfreq(values.size, d=frame_period_s)
    power = np.abs(np.fft.rfft(values * window)) ** 2
    min_frequency_hz = min_bpm / 60.0
    max_frequency_hz = max_bpm / 60.0
    keep_mask = (frequencies_hz >= min_frequency_hz) & (frequencies_hz <= max_frequency_hz)
    if not np.any(keep_mask):
        raise ValueError("No FFT bins fall within the requested BPM band")

    band_frequencies = frequencies_hz[keep_mask]
    band_power = power[keep_mask]
    raw_peak_index = _select_peak_index(band_frequencies, band_power)
    sorted_indices = np.argsort(band_power)[-top_k:][::-1]
    top_peaks = [
        {
            "frequency_hz": float(band_frequencies[index]),
            "bpm": float(band_frequencies[index] * 60.0),
            "power": float(band_power[index]),
        }
        for index in sorted_indices
    ]

    confidence_ratio = None
    if len(sorted_indices) > 1 and float(band_power[sorted_indices[1]]) > 0.0:
        confidence_ratio = float(band_power[sorted_indices[0]] / band_power[sorted_indices[1]])

    frequency_resolution_hz = float(np.median(np.diff(frequencies_hz))) if frequencies_hz.size > 1 else 0.0
    noise_floor = float(np.median(band_power[band_power > 0.0])) if np.any(band_power > 0.0) else 0.0
    peak_power = float(band_power[raw_peak_index])
    peak_to_floor_ratio = peak_power / noise_floor if noise_floor > 0.0 else None
    return {
        "source": source,
        "num_samples": int(values.size),
        "duration_s": float(values.size * frame_period_s),
        "frequency_resolution_hz": frequency_resolution_hz,
        "frequency_resolution_bpm": float(frequency_resolution_hz * 60.0),
        "raw_peak_frequency_hz": float(band_frequencies[raw_peak_index]),
        "raw_peak_bpm": float(band_frequencies[raw_peak_index] * 60.0),
        "raw_peak_power": peak_power,
        "confidence_ratio": confidence_ratio,
        "peak_to_floor_ratio": peak_to_floor_ratio,
        "top_peaks": top_peaks,
        "_band_frequencies_hz": band_frequencies,
        "_band_power": band_power,
    }


def _power_near_frequency(spectral_estimate: dict[str, Any], target_frequency_hz: float) -> dict[str, float] | None:
    frequencies_hz = spectral_estimate.get("_band_frequencies_hz")
    power = spectral_estimate.get("_band_power")
    if frequencies_hz is None or power is None:
        return None
    frequencies = np.asarray(frequencies_hz, dtype=float)
    powers = np.asarray(power, dtype=float)
    if frequencies.size == 0:
        return None
    index = int(np.argmin(np.abs(frequencies - target_frequency_hz)))
    return {
        "frequency_hz": float(frequencies[index]),
        "bpm": float(frequencies[index] * 60.0),
        "power": float(powers[index]),
    }


def _public_spectral_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in estimate.items() if not key.startswith("_")}


def _estimate_windowed_spectral_rate(
    samples: np.ndarray,
    frame_period_s: float,
    *,
    window_seconds: float,
    hop_seconds: float,
    min_bpm: float,
    max_bpm: float,
    source: str,
    reliable_for_min_bpm: bool,
    decision_reason: str,
) -> dict[str, Any]:
    sample_rate_hz = 1.0 / frame_period_s
    window_size = min(samples.size, max(16, int(round(window_seconds * sample_rate_hz))))
    hop_size = max(1, int(round(hop_seconds * sample_rate_hz)))
    start_indices = np.arange(0, max(samples.size - window_size + 1, 1), hop_size)
    if start_indices.size == 0:
        start_indices = np.asarray([0], dtype=int)
    final_start = samples.size - window_size
    if start_indices[-1] != final_start:
        start_indices = np.append(start_indices, final_start)

    estimates: list[dict[str, Any]] = []
    for start_index in start_indices:
        segment = samples[int(start_index) : int(start_index) + window_size]
        try:
            estimate = _estimate_spectral_rate_from_samples(
                segment,
                frame_period_s,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                source=source,
            )
        except ValueError:
            continue
        estimate = _public_spectral_estimate(estimate)
        estimate["start_s"] = float(start_index * frame_period_s)
        estimate["center_s"] = float((start_index + (window_size / 2.0)) * frame_period_s)
        estimates.append(estimate)

    bpm_values = [float(estimate["raw_peak_bpm"]) for estimate in estimates]
    if bpm_values:
        median_bpm = float(np.median(bpm_values))
        p10_bpm = float(np.percentile(bpm_values, 10))
        p90_bpm = float(np.percentile(bpm_values, 90))
        confidence_values = [
            float(estimate["confidence_ratio"])
            for estimate in estimates
            if estimate.get("confidence_ratio") is not None
        ]
        confidence_ratio = float(np.median(confidence_values)) if confidence_values else None
    else:
        median_bpm = None
        p10_bpm = None
        p90_bpm = None
        confidence_ratio = None

    actual_window_seconds = float(window_size * frame_period_s)
    actual_hop_seconds = float(hop_size * frame_period_s)
    return {
        "source": source,
        "window_seconds": actual_window_seconds,
        "hop_seconds": actual_hop_seconds,
        "frequency_resolution_hz": float(1.0 / actual_window_seconds) if actual_window_seconds > 0.0 else None,
        "frequency_resolution_bpm": float(60.0 / actual_window_seconds) if actual_window_seconds > 0.0 else None,
        "num_windows": len(estimates),
        "median_bpm": median_bpm,
        "p10_bpm": p10_bpm,
        "p90_bpm": p90_bpm,
        "bpm_values": bpm_values,
        "confidence_ratio": confidence_ratio,
        "reliable_for_min_bpm": bool(reliable_for_min_bpm),
        "decision_reason": decision_reason,
    }


def _window_support_count(window_estimates: list[dict[str, Any]], target_bpm: float | None) -> int:
    if target_bpm is None:
        return 0
    count = 0
    for estimate in window_estimates:
        if estimate.get("source") == "short_responsive" and not estimate.get("reliable_for_min_bpm"):
            continue
        if _bpm_close(estimate.get("median_bpm"), target_bpm):
            count += 1
    return count


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _is_harmonic_pair(candidate_bpm: float | None, reference_bpm: float | None) -> bool:
    if candidate_bpm is None or reference_bpm is None or candidate_bpm <= 0.0 or reference_bpm <= 0.0:
        return False
    ratio = max(float(candidate_bpm), float(reference_bpm)) / min(float(candidate_bpm), float(reference_bpm))
    return 1.80 <= ratio <= 2.25


def _score_signal_quality(samples: np.ndarray, duration_s: float, min_bpm: float) -> float:
    values = np.asarray(samples, dtype=float)
    if values.size < 16:
        return 0.0
    finite_fraction = float(np.count_nonzero(np.isfinite(values)) / values.size)
    centered = _finite_centered_array(values)
    std_value = float(np.std(centered))
    peak_to_peak = float(np.ptp(centered))
    if std_value <= 1e-12 or peak_to_peak <= 1e-12:
        return 0.0

    slow_period_s = 60.0 / min_bpm if min_bpm > 0.0 else duration_s
    duration_score = _clamp01(duration_s / max(2.0 * slow_period_s, 1e-9))
    dynamic_score = _clamp01((peak_to_peak / std_value) / 3.0)
    return _clamp01((0.35 * finite_fraction) + (0.35 * duration_score) + (0.30 * dynamic_score))


def _score_spectral_confidence(global_estimate: dict[str, Any]) -> float:
    confidence_ratio = global_estimate.get("confidence_ratio")
    peak_to_floor_ratio = global_estimate.get("peak_to_floor_ratio")
    separation_score = 0.35 if confidence_ratio is None else _clamp01((float(confidence_ratio) - 1.0) / 2.0)
    if peak_to_floor_ratio is None or float(peak_to_floor_ratio) <= 1.0:
        floor_score = 0.35
    else:
        floor_score = _clamp01(float(np.log10(float(peak_to_floor_ratio))) / 1.5)
    return _clamp01(0.20 + (0.50 * separation_score) + (0.30 * floor_score))


def _score_temporal_consistency(
    *,
    selected_bpm: float,
    raw_spectral_bpm: float,
    time_domain_bpm: float | None,
    time_domain_confidence: float,
    autocorrelation_bpm: float | None,
    window_estimates: list[dict[str, Any]],
    harmonic_suspected: bool,
    used_subharmonic: bool,
) -> float:
    support_weight = 0.0
    conflict_weight = 0.0

    def add_evidence(
        bpm: float | None,
        weight: float,
        *,
        confidence: float = 1.0,
        harmonic_can_be_explained: bool = False,
    ) -> None:
        nonlocal support_weight, conflict_weight
        if bpm is None:
            return
        effective_weight = weight * _clamp01(confidence)
        if effective_weight <= 0.0:
            return
        if _bpm_close(bpm, selected_bpm):
            support_weight += effective_weight
        elif harmonic_can_be_explained and harmonic_suspected and _is_harmonic_pair(bpm, selected_bpm):
            support_weight += 0.60 * effective_weight
            conflict_weight += 0.10 * effective_weight
        else:
            conflict_weight += effective_weight

    add_evidence(raw_spectral_bpm, 1.0, harmonic_can_be_explained=used_subharmonic)
    add_evidence(time_domain_bpm, 1.25, confidence=time_domain_confidence, harmonic_can_be_explained=False)
    add_evidence(autocorrelation_bpm, 0.85, confidence=0.80 if autocorrelation_bpm is not None else 0.0)
    for estimate in window_estimates:
        source = estimate.get("source")
        if source == "global_fft":
            continue
        if source == "short_responsive" and not estimate.get("reliable_for_min_bpm"):
            weight = 0.20
        else:
            weight = 0.45
        add_evidence(estimate.get("median_bpm"), weight, harmonic_can_be_explained=True)

    total_weight = support_weight + conflict_weight
    if total_weight <= 0.0:
        return 0.30
    score = support_weight / total_weight
    if support_weight >= 1.0 and conflict_weight <= 0.10:
        score = max(score, 0.80)
    return _clamp01(score)


def _score_harmonic_ambiguity(
    *,
    selected_bpm: float,
    raw_spectral_bpm: float,
    harmonic_suspected: bool,
    used_subharmonic: bool,
    is_ambiguous: bool,
    time_supports_selected: bool,
    autocorrelation_supports_selected: bool,
    subharmonic_power_ratio: float | None,
) -> float:
    if is_ambiguous:
        return 0.25
    if harmonic_suspected and used_subharmonic:
        score = 0.62
        if time_supports_selected:
            score += 0.12
        if autocorrelation_supports_selected:
            score += 0.10
        if subharmonic_power_ratio is not None and subharmonic_power_ratio >= 0.25:
            score += 0.06
        return _clamp01(score)
    if _is_harmonic_pair(raw_spectral_bpm, selected_bpm) and not _bpm_close(raw_spectral_bpm, selected_bpm):
        return 0.70
    return 1.0


def _combine_confidence_scores(
    *,
    signal_quality_score: float,
    spectral_confidence: float,
    temporal_consistency_score: float,
    harmonic_ambiguity_score: float,
    is_ambiguous: bool,
) -> float:
    if signal_quality_score <= 0.0:
        return 0.0
    if is_ambiguous and temporal_consistency_score <= 0.05 and harmonic_ambiguity_score <= 0.10:
        return 0.0
    final_confidence = (
        (0.20 * signal_quality_score)
        + (0.25 * spectral_confidence)
        + (0.35 * temporal_consistency_score)
        + (0.20 * harmonic_ambiguity_score)
    )
    if is_ambiguous and temporal_consistency_score < 0.35 and harmonic_ambiguity_score < 0.35:
        final_confidence = min(final_confidence, 0.35)
    elif is_ambiguous:
        final_confidence = min(final_confidence, 0.55)
    return _clamp01(final_confidence)


def estimate_breathing_rate_multiwindow(
    signal: dict[str, Any],
    *,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    top_k: int = 5,
) -> dict[str, Any]:
    """Estimate one global breathing rate using multiple complementary cues.

    The estimator starts from a full-record FFT, then compares that answer to
    windowed spectra, a time-domain cycle estimate, and autocorrelation
    candidates. This is where harmonic correction happens when the brightest FFT
    peak is likely the second harmonic of a slower true rate.
    """

    samples, frame_period_s = _as_signal_samples(signal, prefer_zscore=False)
    if samples.size < 16:
        raise ValueError("Respiration signal is too short for frequency analysis")

    duration_s = _signal_duration_seconds(signal, samples, frame_period_s)
    min_frequency_hz = min_bpm / 60.0
    max_frequency_hz = max_bpm / 60.0
    global_estimate = _estimate_spectral_rate_from_samples(
        samples,
        frame_period_s,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        top_k=top_k,
        source="global_fft",
    )
    autocorrelation_candidates = _estimate_autocorrelation_candidates(
        samples,
        frame_period_s,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        top_k=top_k,
    )
    autocorrelation_bpm = float(autocorrelation_candidates[0]["bpm"]) if autocorrelation_candidates else None
    time_domain = estimate_time_domain_breathing_rate(signal, min_bpm=min_bpm, max_bpm=max_bpm)

    window_estimates: list[dict[str, Any]] = [_public_spectral_estimate(global_estimate)]
    window_estimates[0].update(
        {
            "window_seconds": duration_s,
            "hop_seconds": duration_s,
            "median_bpm": global_estimate["raw_peak_bpm"],
            "p10_bpm": global_estimate["raw_peak_bpm"],
            "p90_bpm": global_estimate["raw_peak_bpm"],
            "bpm_values": [global_estimate["raw_peak_bpm"]],
            "num_windows": 1,
            "reliable_for_min_bpm": True,
            "decision_reason": "Full-record FFT uses the entire signal.",
        }
    )

    # Use multiple window lengths so slow breathing has enough cycles for one
    # view, while shorter behavior changes can still support or veto the full-
    # record answer.
    long_window_choice = choose_respiration_window_seconds(
        duration_s,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        cycles_per_window=3.0,
        preferred_hop_fraction=0.25,
        max_window_fraction=0.75,
    )
    medium_window_choice = choose_respiration_window_seconds(
        duration_s,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        cycles_per_window=2.0,
        preferred_hop_fraction=0.25,
        max_window_fraction=0.50,
    )
    short_window_seconds = min(duration_s, max(8.0, 60.0 / min_bpm))
    short_cycles_at_min_bpm = short_window_seconds * min_bpm / 60.0
    short_window_choice = {
        "window_seconds": float(short_window_seconds),
        "hop_seconds": float(max(1e-9, short_window_seconds * 0.25)),
        "frequency_resolution_hz": float(1.0 / short_window_seconds),
        "frequency_resolution_bpm": float(60.0 / short_window_seconds),
        "cycles_at_min_bpm": float(short_cycles_at_min_bpm),
        "reliable_for_min_bpm": bool(short_cycles_at_min_bpm >= 3.0),
        "decision_reason": (
            f"Short responsive window contains {short_cycles_at_min_bpm:.2f} cycles at {min_bpm:.1f} BPM; "
            "use it for visualization/responsiveness, not as primary slow-rate evidence."
        ),
    }

    for source, choice in (
        ("long_window", long_window_choice),
        ("medium_window", medium_window_choice),
        ("short_responsive", short_window_choice),
    ):
        try:
            window_estimates.append(
                _estimate_windowed_spectral_rate(
                    samples,
                    frame_period_s,
                    window_seconds=float(choice["window_seconds"]),
                    hop_seconds=float(choice["hop_seconds"]),
                    min_bpm=min_bpm,
                    max_bpm=max_bpm,
                    source=source,
                    reliable_for_min_bpm=bool(choice["reliable_for_min_bpm"]),
                    decision_reason=str(choice["decision_reason"]),
                )
            )
        except ValueError:
            continue

    raw_spectral_bpm = float(global_estimate["raw_peak_bpm"])
    raw_spectral_frequency_hz = float(global_estimate["raw_peak_frequency_hz"])
    raw_peak_power = float(global_estimate["raw_peak_power"])
    subharmonic_candidate_bpm = raw_spectral_bpm / 2.0 if (raw_spectral_bpm / 2.0) >= min_bpm else None
    subharmonic_candidate_frequency_hz = (
        raw_spectral_frequency_hz / 2.0 if subharmonic_candidate_bpm is not None else None
    )
    half_power = (
        _power_near_frequency(global_estimate, subharmonic_candidate_frequency_hz)
        if subharmonic_candidate_frequency_hz is not None
        else None
    )
    subharmonic_power_ratio = (
        float(half_power["power"] / raw_peak_power)
        if half_power is not None and raw_peak_power > 0.0
        else None
    )
    harmonic_ratio = (
        raw_spectral_bpm / subharmonic_candidate_bpm
        if subharmonic_candidate_bpm is not None and subharmonic_candidate_bpm > 0.0
        else None
    )

    time_domain_bpm = time_domain.get("bpm")
    time_domain_confidence = float(time_domain.get("confidence") or 0.0)
    time_support_available = time_domain_bpm is not None and time_domain_confidence >= 0.35
    time_support_lower = (
        subharmonic_candidate_bpm is not None
        and time_support_available
        and _bpm_close(time_domain_bpm, subharmonic_candidate_bpm)
    )
    time_support_higher = time_support_available and _bpm_close(time_domain_bpm, raw_spectral_bpm)
    lower_autocorrelation_candidate = _autocorrelation_candidate_near(
        autocorrelation_candidates,
        subharmonic_candidate_bpm,
    )
    higher_autocorrelation_candidate = _autocorrelation_candidate_near(
        autocorrelation_candidates,
        raw_spectral_bpm,
    )
    lower_autocorrelation_value = (
        None
        if lower_autocorrelation_candidate is None
        else float(lower_autocorrelation_candidate.get("correlation") or 0.0)
    )
    higher_autocorrelation_value = (
        None
        if higher_autocorrelation_candidate is None
        else float(higher_autocorrelation_candidate.get("correlation") or 0.0)
    )
    lower_autocorrelation_is_clear = bool(
        lower_autocorrelation_value is not None
        and lower_autocorrelation_value >= 0.25
        and (
            higher_autocorrelation_value is None
            or lower_autocorrelation_value >= max(1.35 * higher_autocorrelation_value, higher_autocorrelation_value + 0.08)
        )
    )
    higher_autocorrelation_is_clear = bool(
        higher_autocorrelation_value is not None
        and higher_autocorrelation_value >= 0.25
        and (
            lower_autocorrelation_value is None
            or higher_autocorrelation_value >= max(1.35 * lower_autocorrelation_value, lower_autocorrelation_value + 0.08)
        )
    )
    autocorrelation_support_lower = (
        subharmonic_candidate_bpm is not None
        and lower_autocorrelation_candidate is not None
        and lower_autocorrelation_is_clear
    )
    autocorrelation_support_higher = higher_autocorrelation_candidate is not None and higher_autocorrelation_is_clear
    window_support_lower = _window_support_count(window_estimates, subharmonic_candidate_bpm)
    window_support_higher = _window_support_count(window_estimates, raw_spectral_bpm)
    lower_has_meaningful_power = subharmonic_power_ratio is not None and subharmonic_power_ratio >= 0.10
    lower_has_substantial_power = subharmonic_power_ratio is not None and subharmonic_power_ratio >= 0.25
    weak_alternating_subharmonic = bool(
        time_support_lower
        and time_domain.get("cycle_label") == "alternating-dominant"
        and not lower_has_substantial_power
        and not lower_autocorrelation_is_clear
        and window_support_higher > window_support_lower
    )

    selected_bpm = raw_spectral_bpm
    selected_frequency_hz = raw_spectral_frequency_hz
    selected_bpm_source = "global_fft"
    harmonic_suspected = False
    used_subharmonic = False
    used_time_domain_support = False
    used_autocorrelation_support = False
    is_ambiguous = False
    decision_parts: list[str] = [
        f"Raw strongest spectral peak is {raw_spectral_bpm:.2f} BPM ({raw_spectral_frequency_hz:.4f} Hz)."
    ]

    # When the dominant spectral peak is close to 2x a slower candidate, weigh
    # lower-frequency evidence before accepting the brighter peak as the true RR.
    if subharmonic_candidate_bpm is not None and harmonic_ratio is not None and 1.85 <= harmonic_ratio <= 2.15:
        lower_score = 0.0
        higher_score = 0.0
        if lower_has_meaningful_power:
            lower_score += 1.0
            decision_parts.append(
                f"The {subharmonic_candidate_bpm:.2f} BPM subharmonic has "
                f"{subharmonic_power_ratio:.2f}x the raw-peak power."
            )
        if time_support_lower:
            lower_score += 2.0
        if autocorrelation_support_lower:
            lower_score += 1.0
        elif lower_autocorrelation_candidate is not None:
            lower_score += 0.30
        lower_score += 0.75 * window_support_lower

        if time_support_higher:
            higher_score += 2.0
        if autocorrelation_support_higher:
            higher_score += 1.0
        elif higher_autocorrelation_candidate is not None:
            higher_score += 0.30
        higher_score += 0.75 * window_support_higher

        higher_evidence_overrides_lower_time = bool(
            time_support_lower
            and not weak_alternating_subharmonic
            and autocorrelation_support_higher
            and not autocorrelation_support_lower
            and higher_score >= lower_score
        )

        if higher_evidence_overrides_lower_time:
            harmonic_suspected = True
            is_ambiguous = True
            selected_bpm_source = "global_fft_autocorrelation_supported_harmonic_ambiguous"
            used_autocorrelation_support = True
            decision_parts.append(
                "Time-domain spacing suggests the lower subharmonic, but the raw spectral peak and "
                "autocorrelation both support the higher rate. Keeping the raw spectral BPM and marking "
                "the harmonic evidence ambiguous."
            )
        elif time_support_lower and not weak_alternating_subharmonic:
            selected_bpm = float(subharmonic_candidate_bpm)
            selected_frequency_hz = selected_bpm / 60.0
            selected_bpm_source = "time_domain_supported_subharmonic"
            harmonic_suspected = True
            used_subharmonic = True
            used_time_domain_support = True
            if autocorrelation_support_lower:
                used_autocorrelation_support = True
            decision_parts.append(
                f"Time-domain full-cycle spacing supports {selected_bpm:.2f} BPM, so the raw peak is treated as a likely second harmonic."
            )
            if time_support_higher or higher_score >= lower_score:
                is_ambiguous = True
                decision_parts.append("Some evidence also supports the higher rate, so the result is marked ambiguous.")
        elif weak_alternating_subharmonic:
            harmonic_suspected = True
            is_ambiguous = True
            selected_bpm_source = "global_fft_harmonic_ambiguous"
            decision_parts.append(
                "Every-other-extrema spacing suggests the lower subharmonic, but that evidence is weak: "
                "the subharmonic power is low, autocorrelation does not clearly separate it from the raw peak, "
                "and the spectral windows favor the raw peak. Keeping the raw spectral BPM and marking harmonic ambiguity."
            )
        elif time_support_higher:
            selected_bpm_source = "global_fft_time_domain_supported"
            used_time_domain_support = True
            used_autocorrelation_support = autocorrelation_support_higher
            decision_parts.append(
                f"Time-domain breath intervals also support {raw_spectral_bpm:.2f} BPM; no harmonic correction was applied."
            )
            if autocorrelation_support_lower and lower_has_meaningful_power:
                is_ambiguous = True
                decision_parts.append("Autocorrelation/subharmonic evidence conflicts with the time-domain result.")
        elif lower_score >= 2.0 and lower_score > higher_score:
            selected_bpm = float(subharmonic_candidate_bpm)
            selected_frequency_hz = selected_bpm / 60.0
            selected_bpm_source = "multiwindow_supported_subharmonic"
            harmonic_suspected = True
            used_subharmonic = True
            used_autocorrelation_support = autocorrelation_support_lower
            decision_parts.append(
                f"Autocorrelation/window evidence favors the {selected_bpm:.2f} BPM subharmonic over the brighter 2x peak."
            )
        elif lower_score >= 2.0 and higher_score >= 2.0:
            is_ambiguous = True
            selected_bpm_source = "ambiguous_global_fft"
            decision_parts.append("Lower-frequency and higher-frequency evidence conflict; keeping the raw spectral peak but marking ambiguity.")
        else:
            decision_parts.append("The lower 2x-related candidate was not sufficiently supported, so the raw spectral peak was kept.")
    elif time_support_available and not _bpm_close(time_domain_bpm, raw_spectral_bpm):
        is_ambiguous = True
        decision_parts.append(
            f"Time-domain estimate ({float(time_domain_bpm):.2f} BPM) disagrees with the strongest spectral peak."
        )
    elif time_support_higher:
        selected_bpm_source = "global_fft_time_domain_supported"
        used_time_domain_support = True
        used_autocorrelation_support = autocorrelation_support_higher
        decision_parts.append("Time-domain cycle spacing supports the raw spectral estimate.")

    time_supports_selected = time_support_available and _bpm_close(time_domain_bpm, selected_bpm)
    autocorrelation_supports_selected = _bpm_close(autocorrelation_bpm, selected_bpm)
    signal_quality_score = _score_signal_quality(samples, duration_s, min_bpm)
    spectral_confidence = _score_spectral_confidence(global_estimate)
    temporal_consistency_score = _score_temporal_consistency(
        selected_bpm=float(selected_bpm),
        raw_spectral_bpm=raw_spectral_bpm,
        time_domain_bpm=time_domain_bpm,
        time_domain_confidence=time_domain_confidence,
        autocorrelation_bpm=autocorrelation_bpm,
        window_estimates=window_estimates,
        harmonic_suspected=harmonic_suspected,
        used_subharmonic=used_subharmonic,
    )
    harmonic_ambiguity_score = _score_harmonic_ambiguity(
        selected_bpm=float(selected_bpm),
        raw_spectral_bpm=raw_spectral_bpm,
        harmonic_suspected=harmonic_suspected,
        used_subharmonic=used_subharmonic,
        is_ambiguous=is_ambiguous,
        time_supports_selected=time_supports_selected,
        autocorrelation_supports_selected=autocorrelation_supports_selected,
        subharmonic_power_ratio=subharmonic_power_ratio,
    )
    final_confidence = _combine_confidence_scores(
        signal_quality_score=signal_quality_score,
        spectral_confidence=spectral_confidence,
        temporal_consistency_score=temporal_consistency_score,
        harmonic_ambiguity_score=harmonic_ambiguity_score,
        is_ambiguous=is_ambiguous,
    )
    if harmonic_suspected and used_subharmonic and not is_ambiguous:
        final_confidence = min(final_confidence, 0.82)

    return {
        "duration_s": duration_s,
        "sample_rate_hz": float(1.0 / frame_period_s),
        "selected_bpm": float(selected_bpm),
        "selected_frequency_hz": float(selected_frequency_hz),
        "raw_spectral_bpm": raw_spectral_bpm,
        "raw_spectral_frequency_hz": raw_spectral_frequency_hz,
        "raw_peak_bpm": raw_spectral_bpm,
        "raw_peak_frequency_hz": raw_spectral_frequency_hz,
        "subharmonic_candidate_bpm": subharmonic_candidate_bpm,
        "subharmonic_candidate_frequency_hz": subharmonic_candidate_frequency_hz,
        "autocorrelation_bpm": autocorrelation_bpm,
        "autocorrelation_frequency_hz": None if autocorrelation_bpm is None else float(autocorrelation_bpm / 60.0),
        "autocorrelation_candidates": autocorrelation_candidates,
        "time_domain_bpm": time_domain_bpm,
        "time_domain_frequency_hz": None if time_domain_bpm is None else float(time_domain_bpm / 60.0),
        "time_domain_period_s": time_domain.get("period_s"),
        "time_domain_num_breaths": time_domain.get("num_breaths_detected", 0),
        "time_domain_confidence": time_domain_confidence,
        "time_domain": time_domain,
        "window_estimates": window_estimates,
        "harmonic_suspected": harmonic_suspected,
        "harmonic_ratio": harmonic_ratio,
        "subharmonic_power_ratio": subharmonic_power_ratio,
        "used_subharmonic": used_subharmonic,
        "used_autocorrelation_support": used_autocorrelation_support,
        "used_time_domain_support": used_time_domain_support,
        "is_ambiguous": is_ambiguous,
        "selected_bpm_source": selected_bpm_source,
        "signal_quality_score": signal_quality_score,
        "spectral_confidence": spectral_confidence,
        "temporal_consistency_score": temporal_consistency_score,
        "harmonic_ambiguity_score": harmonic_ambiguity_score,
        "final_confidence": final_confidence,
        "confidence": final_confidence,
        "quality_score": final_confidence,
        "decision_reason": " ".join(decision_parts),
        "top_peaks": global_estimate["top_peaks"],
        "confidence_ratio": global_estimate["confidence_ratio"],
        "peak_to_floor_ratio": global_estimate["peak_to_floor_ratio"],
        "chosen_window": long_window_choice,
    }


def estimate_global_breathing_rate(
    signal: dict[str, Any],
    *,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    top_k: int = 5,
) -> dict[str, Any]:
    estimate = estimate_breathing_rate_multiwindow(signal, min_bpm=min_bpm, max_bpm=max_bpm, top_k=top_k)
    return {
        **estimate,
        "peak_frequency_hz": estimate["selected_frequency_hz"],
        "peak_bpm": estimate["selected_bpm"],
        "raw_peak_frequency_hz": estimate["raw_spectral_frequency_hz"],
        "raw_peak_bpm": estimate["raw_spectral_bpm"],
        "used_subharmonic": estimate["used_subharmonic"],
        "used_autocorrelation_support": estimate["used_autocorrelation_support"],
        "autocorrelation_peak_bpm": estimate["autocorrelation_bpm"],
        "confidence_ratio": estimate["confidence_ratio"],
        "top_peaks": estimate["top_peaks"],
    }


def estimate_bpm_series_from_spectrogram(
    spectrogram: dict[str, Any],
    *,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
) -> dict[str, Any]:
    frequencies_hz = np.asarray(spectrogram["frequencies_hz"], dtype=float)
    power_db = np.asarray(spectrogram["power_db"], dtype=float)
    if power_db.ndim != 2:
        raise ValueError("spectrogram power_db must be a 2D array")

    min_frequency_hz = min_bpm / 60.0
    max_frequency_hz = max_bpm / 60.0
    keep_mask = (frequencies_hz >= min_frequency_hz) & (frequencies_hz <= max_frequency_hz)
    if not np.any(keep_mask):
        raise ValueError("No spectrogram bins fall within the requested BPM band")

    band_frequencies = frequencies_hz[keep_mask]
    band_power_db = power_db[keep_mask, :]
    band_power_linear = np.power(10.0, band_power_db / 10.0)
    single_slice = band_power_linear.shape[1] < 2
    peak_indices: list[int] = []
    for column_index in range(band_power_linear.shape[1]):
        column_power = band_power_linear[:, column_index]
        raw_peak_index = _select_peak_index(band_frequencies, column_power)
        if single_slice:
            peak_indices.append(raw_peak_index)
            continue
        peak_index, _ = _adjust_peak_index_for_subharmonic(
            band_frequencies,
            column_power,
            raw_peak_index,
            min_frequency_hz=min_frequency_hz,
        )
        peak_indices.append(peak_index)
    peak_indices = np.asarray(peak_indices, dtype=int)
    bpm_values = 60.0 * band_frequencies[peak_indices]

    return {
        "times_s": list(spectrogram["times_s"]),
        "bpm_values": bpm_values.tolist(),
        "median_bpm": float(np.median(bpm_values)),
        "p10_bpm": float(np.percentile(bpm_values, 10)),
        "p90_bpm": float(np.percentile(bpm_values, 90)),
        "min_bpm": float(np.min(bpm_values)),
        "max_bpm": float(np.max(bpm_values)),
    }


def _signal_segment(
    signal: dict[str, Any],
    *,
    start_s: float,
    end_s: float,
) -> dict[str, Any] | None:
    frame_period_s = float(signal["frame_period_s"])
    if frame_period_s <= 0.0 or end_s <= start_s:
        return None

    times_s = np.asarray(signal["times_s"], dtype=float)
    centered = np.asarray(signal["centered_signal"], dtype=float)
    if times_s.size != centered.size:
        sample_times_s = np.arange(centered.size, dtype=float) * frame_period_s
    else:
        sample_times_s = times_s
    keep_mask = (sample_times_s >= start_s) & (sample_times_s < end_s)
    if np.count_nonzero(keep_mask) < 16:
        return None

    segment_centered = centered[keep_mask]
    segment_centered = segment_centered - np.mean(segment_centered)
    segment_std = float(np.std(segment_centered))
    segment_zscore = segment_centered / segment_std if segment_std > 0.0 else np.zeros_like(segment_centered)
    segment_times_s = sample_times_s[keep_mask] - float(sample_times_s[keep_mask][0])
    return {
        "times_s": segment_times_s.tolist(),
        "frame_period_s": frame_period_s,
        "raw_signal": segment_centered.tolist(),
        "centered_signal": segment_centered.tolist(),
        "zscore_signal": segment_zscore.tolist(),
        "center_bin": signal.get("center_bin", 0),
        "center_range_m": signal.get("center_range_m", 0.0),
        "selected_bins": list(signal.get("selected_bins", [])),
        "selected_ranges_m": list(signal.get("selected_ranges_m", [])),
    }


def _estimate_segment_rate(
    signal: dict[str, Any],
    *,
    label: str,
    start_s: float,
    end_s: float,
    min_bpm: float,
    max_bpm: float,
) -> dict[str, Any]:
    segment = _signal_segment(signal, start_s=start_s, end_s=end_s)
    if segment is None:
        return {
            "label": label,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "bpm": None,
            "confidence": 0.0,
            "decision_reason": "Segment was too short for a reliable estimate.",
        }
    try:
        estimate = estimate_breathing_rate_multiwindow(segment, min_bpm=min_bpm, max_bpm=max_bpm)
    except ValueError as exc:
        return {
            "label": label,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "bpm": None,
            "confidence": 0.0,
            "decision_reason": str(exc),
        }
    spectral_resolution_bpm = None
    if estimate.get("window_estimates"):
        spectral_resolution_bpm = estimate["window_estimates"][0].get("frequency_resolution_bpm")
    selected_bpm = float(estimate["selected_bpm"])
    representative_candidates: list[tuple[float, float]] = [(selected_bpm, 1.0)]
    time_domain_bpm = estimate.get("time_domain_bpm")
    if (
        time_domain_bpm is not None
        and float(estimate.get("time_domain_confidence") or 0.0) >= 0.55
        and _bpm_close(time_domain_bpm, selected_bpm)
    ):
        representative_candidates.append((float(time_domain_bpm), 1.25))
    autocorrelation_bpm = estimate.get("autocorrelation_bpm")
    if autocorrelation_bpm is not None and _bpm_close(autocorrelation_bpm, selected_bpm):
        representative_candidates.append((float(autocorrelation_bpm), 1.0))

    total_weight = sum(weight for _, weight in representative_candidates)
    representative_bpm = (
        sum(bpm * weight for bpm, weight in representative_candidates) / total_weight
        if total_weight > 0.0
        else selected_bpm
    )
    if spectral_resolution_bpm is not None and spectral_resolution_bpm <= 1.0:
        representative_bpm = selected_bpm
    return {
        "label": label,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "bpm": float(representative_bpm),
        "frequency_hz": float(representative_bpm / 60.0),
        "selected_bpm": estimate["selected_bpm"],
        "selected_frequency_hz": estimate["selected_frequency_hz"],
        "spectral_resolution_bpm": spectral_resolution_bpm,
        "raw_spectral_bpm": estimate["raw_spectral_bpm"],
        "time_domain_bpm": estimate["time_domain_bpm"],
        "autocorrelation_bpm": estimate["autocorrelation_bpm"],
        "confidence": estimate["final_confidence"],
        "harmonic_suspected": estimate["harmonic_suspected"],
        "is_ambiguous": estimate["is_ambiguous"],
        "decision_reason": estimate["decision_reason"],
    }


def _segment_rate_supported(segment_estimate: dict[str, Any]) -> bool:
    bpm = segment_estimate.get("bpm")
    if bpm is None:
        return False
    confidence = float(segment_estimate.get("confidence") or 0.0)
    time_domain_bpm = segment_estimate.get("time_domain_bpm")
    autocorrelation_bpm = segment_estimate.get("autocorrelation_bpm")
    corroborated = _bpm_close(time_domain_bpm, bpm) or _bpm_close(autocorrelation_bpm, bpm)
    return bool((confidence >= 0.35 and corroborated) or (confidence >= 0.70 and not segment_estimate.get("is_ambiguous")))


def _segment_cycle_count(segment_estimate: dict[str, Any]) -> float:
    bpm = segment_estimate.get("bpm")
    if bpm is None:
        return 0.0
    duration_s = float(segment_estimate.get("end_s", 0.0)) - float(segment_estimate.get("start_s", 0.0))
    return max(0.0, duration_s * float(bpm) / 60.0)


def _transition_pair_looks_like_harmonic_flip(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    global_bpm: float | None,
    raw_spectral_bpm: float | None,
) -> bool:
    """Return True when an apparent segment change is likely a 1x/2x representation flip.

    Short segment estimates can jump between a respiration fundamental and its
    second harmonic even when the breathing rate is stable. If the full-record
    estimate favors the lower member of that pair, treat the segment jump as
    harmonic ambiguity rather than a sustained transition.
    """
    before_bpm = before.get("bpm")
    after_bpm = after.get("bpm")
    if before_bpm is None or after_bpm is None or not _is_harmonic_pair(before_bpm, after_bpm):
        return False

    lower_bpm = min(float(before_bpm), float(after_bpm))
    higher_bpm = max(float(before_bpm), float(after_bpm))
    full_record_favors_lower = _bpm_close(global_bpm, lower_bpm) or _bpm_close(raw_spectral_bpm, lower_bpm)
    full_record_favors_higher = _bpm_close(global_bpm, higher_bpm) or _bpm_close(raw_spectral_bpm, higher_bpm)
    lower_segment = before if float(before_bpm) <= float(after_bpm) else after
    higher_segment = after if lower_segment is before else before
    lower_confidence = float(lower_segment.get("confidence") or 0.0)
    higher_confidence = float(higher_segment.get("confidence") or 0.0)
    lower_raw_supports_lower = _bpm_close(lower_segment.get("raw_spectral_bpm"), lower_bpm)
    higher_raw_supports_higher = _bpm_close(higher_segment.get("raw_spectral_bpm"), higher_bpm)
    lower_was_harmonic_corrected = bool(lower_segment.get("harmonic_suspected")) and not lower_raw_supports_lower
    segment_evidence_supports_transition = bool(
        lower_confidence >= 0.65
        and higher_confidence >= 0.65
        and _segment_rate_supported(lower_segment)
        and _segment_rate_supported(higher_segment)
        and lower_raw_supports_lower
        and higher_raw_supports_higher
        and full_record_favors_higher
        and not lower_was_harmonic_corrected
        and not higher_segment.get("is_ambiguous")
    )
    if segment_evidence_supports_transition:
        return False

    if full_record_favors_lower and not full_record_favors_higher:
        return True
    if full_record_favors_lower and _bpm_close(global_bpm, lower_bpm):
        return True
    return False


def _candidate_transition_splits(duration_s: float, min_bpm: float) -> list[float]:
    if duration_s <= 0.0:
        return []

    # Require enough time on each side for at least a few slow breaths, but do
    # not make the minimum so long that a 60 s transition recording cannot be
    # evaluated. This is a transition search grid, not the final window length.
    min_segment_s = min(max(18.0, 2.0 * 60.0 / max(min_bpm, 6.0)), duration_s * 0.40)
    if duration_s < (2.0 * min_segment_s):
        return []

    search_start = min_segment_s
    search_end = duration_s - min_segment_s
    if search_end < search_start:
        return []

    step_s = max(2.0, min(6.0, duration_s / 12.0))
    split_values = list(np.arange(search_start, search_end + (0.5 * step_s), step_s, dtype=float))
    split_values.append(duration_s / 2.0)
    rounded = sorted({round(float(value), 3) for value in split_values if search_start <= value <= search_end})
    if len(rounded) > 7:
        indices = np.linspace(0, len(rounded) - 1, 7).round().astype(int)
        rounded = [rounded[int(index)] for index in indices]
        if round(duration_s / 2.0, 3) not in rounded:
            rounded.append(round(duration_s / 2.0, 3))
            rounded = sorted(set(rounded))
    return rounded


def _find_adaptive_transition(
    signal: dict[str, Any],
    *,
    duration_s: float,
    frame_period_s: float,
    min_bpm: float,
    max_bpm: float,
    global_bpm: float | None = None,
    raw_spectral_bpm: float | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for split_s in _candidate_transition_splits(duration_s, min_bpm):
        before = _estimate_segment_rate(
            signal,
            label=f"before_{split_s:.1f}s",
            start_s=0.0,
            end_s=split_s,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        after = _estimate_segment_rate(
            signal,
            label=f"after_{split_s:.1f}s",
            start_s=split_s,
            end_s=duration_s + frame_period_s,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        before_bpm = before.get("bpm")
        after_bpm = after.get("bpm")
        if before_bpm is None or after_bpm is None:
            candidates.append(
                {
                    "split_time_s": float(split_s),
                    "transition_detected": False,
                    "score": 0.0,
                    "before": before,
                    "after": after,
                }
            )
            continue

        bpm_change = abs(float(after_bpm) - float(before_bpm))
        reference_bpm = min(float(before_bpm), float(after_bpm))
        transition_threshold_bpm = max(4.0, 0.35 * max(reference_bpm, 1.0))
        before_supported = _segment_rate_supported(before)
        after_supported = _segment_rate_supported(after)
        before_cycle_count = _segment_cycle_count(before)
        after_cycle_count = _segment_cycle_count(after)
        enough_cycles_for_transition = bool(min(before_cycle_count, after_cycle_count) >= 3.0)
        harmonic_flip_explained = _transition_pair_looks_like_harmonic_flip(
            before,
            after,
            global_bpm=global_bpm,
            raw_spectral_bpm=raw_spectral_bpm,
        )
        avg_confidence = 0.5 * (float(before.get("confidence") or 0.0) + float(after.get("confidence") or 0.0))
        change_score = _clamp01(bpm_change / max(2.0 * transition_threshold_bpm, 1e-9))
        balance_score = 1.0 - _clamp01(abs(split_s - (duration_s / 2.0)) / max(duration_s / 2.0, 1e-9))
        transition_detected = bool(
            before_supported
            and after_supported
            and avg_confidence >= 0.40
            and bpm_change >= transition_threshold_bpm
            and enough_cycles_for_transition
            and not harmonic_flip_explained
        )
        score = (
            (0.45 * avg_confidence)
            + (0.30 * change_score)
            + (0.15 * balance_score)
            + (0.10 if transition_detected else 0.0)
        )
        candidates.append(
            {
                "split_time_s": float(split_s),
                "transition_detected": transition_detected,
                "score": float(score if transition_detected else 0.0),
                "before_bpm": float(before_bpm),
                "after_bpm": float(after_bpm),
                "bpm_change_magnitude": float(bpm_change),
                "transition_threshold_bpm": float(transition_threshold_bpm),
                "before_confidence": float(before.get("confidence") or 0.0),
                "after_confidence": float(after.get("confidence") or 0.0),
                "before_supported": before_supported,
                "after_supported": after_supported,
                "before_cycle_count": float(before_cycle_count),
                "after_cycle_count": float(after_cycle_count),
                "enough_cycles_for_transition": enough_cycles_for_transition,
                "harmonic_flip_explained": harmonic_flip_explained,
                "before": before,
                "after": after,
            }
        )

    transition_candidates = [candidate for candidate in candidates if candidate["transition_detected"]]
    best = None
    if transition_candidates:
        strongest = max(transition_candidates, key=lambda row: float(row["score"]))
        strongest_score = float(strongest["score"])
        plateau_candidates = [
            candidate
            for candidate in transition_candidates
            if float(candidate["score"]) >= max(strongest_score * 0.90, strongest_score - 0.05)
        ]
        plateau_split_s = float(np.median([candidate["split_time_s"] for candidate in plateau_candidates]))
        best = min(
            plateau_candidates,
            key=lambda row: (abs(float(row["split_time_s"]) - plateau_split_s), -float(row["score"])),
        )
        best["strongest_split_time_s"] = float(strongest["split_time_s"])
        best["plateau_split_time_s"] = plateau_split_s
    return {
        "transition_detected": best is not None,
        "best_transition": best,
        "transition_search_candidates": candidates,
    }


def _split_series_early_late(
    times_s: list[float],
    bpm_values: list[float],
    duration_s: float,
) -> tuple[float | None, float | None, float | None, bool]:
    if len(times_s) < 3 or len(bpm_values) < 3:
        return None, None, None, False
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(bpm_values, dtype=float)
    early_values = values[times <= duration_s / 3.0]
    mid_values = values[(times > duration_s / 3.0) & (times < 2.0 * duration_s / 3.0)]
    late_values = values[times >= 2.0 * duration_s / 3.0]
    if early_values.size == 0:
        early_values = values[: max(1, values.size // 3)]
    if late_values.size == 0:
        late_values = values[-max(1, values.size // 3) :]

    early_bpm = float(np.median(early_values)) if early_values.size else None
    mid_bpm = float(np.median(mid_values)) if mid_values.size else None
    late_bpm = float(np.median(late_values)) if late_values.size else None
    largest_step_detected = False
    if values.size >= 3:
        diffs = np.abs(np.diff(values))
        largest_step_detected = bool(np.max(diffs) >= max(4.0, 0.30 * max(float(np.median(values)), 1.0)))
    return early_bpm, mid_bpm, late_bpm, largest_step_detected


def estimate_respiration_trend(
    signal: dict[str, Any],
    *,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
    bpm_times_s: list[float] | None = None,
    bpm_values: list[float] | None = None,
    global_bpm: float | None = None,
    global_confidence: float | None = None,
    raw_spectral_bpm: float | None = None,
) -> dict[str, Any]:
    samples, frame_period_s = _as_signal_samples(signal, prefer_zscore=False)
    duration_s = _signal_duration_seconds(signal, samples, frame_period_s)
    if duration_s <= 0.0:
        return {
            "recording_type": "failed",
            "respiration_pattern": "failed",
            "is_nonstationary": False,
            "transition_detected": False,
            "early_bpm": None,
            "mid_bpm": None,
            "late_bpm": None,
            "bpm_change_magnitude": None,
            "transition_time_s": None,
            "global_bpm_representative": False,
            "global_bpm_confidence": 0.0,
            "trend_confidence": 0.0,
            "transition_confidence": 0.0,
            "nonstationary_reason": "No usable duration was available.",
            "segment_estimates": [],
        }

    if global_bpm is None or global_confidence is None or raw_spectral_bpm is None:
        try:
            full_record_estimate = estimate_breathing_rate_multiwindow(signal, min_bpm=min_bpm, max_bpm=max_bpm)
        except ValueError:
            full_record_estimate = {}
        if global_bpm is None:
            global_bpm = full_record_estimate.get("selected_bpm")
        if global_confidence is None:
            global_confidence = full_record_estimate.get("final_confidence")
        if raw_spectral_bpm is None:
            raw_spectral_bpm = full_record_estimate.get("raw_spectral_bpm")

    segment_bounds = [
        ("early", 0.0, duration_s / 3.0),
        ("mid", duration_s / 3.0, 2.0 * duration_s / 3.0),
        ("late", 2.0 * duration_s / 3.0, duration_s + frame_period_s),
    ]
    segment_estimates = [
        _estimate_segment_rate(
            signal,
            label=label,
            start_s=start_s,
            end_s=end_s,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        for label, start_s, end_s in segment_bounds
    ]
    adaptive_transition = _find_adaptive_transition(
        signal,
        duration_s=duration_s,
        frame_period_s=frame_period_s,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        global_bpm=global_bpm,
        raw_spectral_bpm=raw_spectral_bpm,
    )
    segment_by_label = {estimate["label"]: estimate for estimate in segment_estimates}
    early_bpm = segment_by_label["early"].get("bpm")
    mid_bpm = segment_by_label["mid"].get("bpm")
    late_bpm = segment_by_label["late"].get("bpm")
    early_confidence = float(segment_by_label["early"].get("confidence") or 0.0)
    late_confidence = float(segment_by_label["late"].get("confidence") or 0.0)
    early_supported = _segment_rate_supported(segment_by_label["early"])
    late_supported = _segment_rate_supported(segment_by_label["late"])

    series_early_bpm = series_mid_bpm = series_late_bpm = None
    series_step_detected = False
    if bpm_times_s is not None and bpm_values is not None:
        series_early_bpm, series_mid_bpm, series_late_bpm, series_step_detected = _split_series_early_late(
            bpm_times_s,
            bpm_values,
            duration_s,
        )

    change_sources: list[float] = []
    if early_bpm is not None and late_bpm is not None:
        change_sources.append(abs(float(late_bpm) - float(early_bpm)))
    if series_early_bpm is not None and series_late_bpm is not None:
        change_sources.append(abs(float(series_late_bpm) - float(series_early_bpm)))
    bpm_change_magnitude = max(change_sources) if change_sources else None

    reference_bpm = None
    if early_bpm is not None and late_bpm is not None:
        reference_bpm = min(float(early_bpm), float(late_bpm))
    elif series_early_bpm is not None and series_late_bpm is not None:
        reference_bpm = min(float(series_early_bpm), float(series_late_bpm))
    transition_threshold_bpm = max(4.0, 0.35 * max(reference_bpm or min_bpm, 1.0))

    segment_harmonic_flip_explained = _transition_pair_looks_like_harmonic_flip(
        segment_by_label["early"],
        segment_by_label["late"],
        global_bpm=global_bpm,
        raw_spectral_bpm=raw_spectral_bpm,
    )
    segment_cycle_supported = bool(
        min(
            _segment_cycle_count(segment_by_label["early"]),
            _segment_cycle_count(segment_by_label["late"]),
        )
        >= 3.0
    )
    segment_transition = (
        early_bpm is not None
        and late_bpm is not None
        and early_confidence >= 0.35
        and late_confidence >= 0.35
        and early_supported
        and late_supported
        and abs(float(late_bpm) - float(early_bpm)) >= transition_threshold_bpm
        and segment_cycle_supported
        and not segment_harmonic_flip_explained
    )
    series_transition = (
        series_early_bpm is not None
        and series_late_bpm is not None
        and abs(float(series_late_bpm) - float(series_early_bpm)) >= transition_threshold_bpm
    )
    adaptive_transition_detected = bool(adaptive_transition.get("transition_detected"))
    if adaptive_transition_detected:
        best_transition = adaptive_transition["best_transition"]
        early_bpm = best_transition["before_bpm"]
        late_bpm = best_transition["after_bpm"]
        bpm_change_magnitude = best_transition["bpm_change_magnitude"]
        transition_threshold_bpm = best_transition["transition_threshold_bpm"]
    transition_detected = bool(adaptive_transition_detected or segment_transition)
    summary_window_transition = False
    series_median_bpm = float(np.median(np.asarray(bpm_values, dtype=float))) if bpm_values else None
    allow_series_only_transition = False
    if allow_series_only_transition and not transition_detected and global_bpm is not None and series_median_bpm is not None:
        global_series_gap = abs(float(global_bpm) - series_median_bpm)
        series_median_is_harmonic = bool(
            _is_harmonic_pair(series_median_bpm, global_bpm)
            and (
                raw_spectral_bpm is None
                or _bpm_close(raw_spectral_bpm, series_median_bpm)
                or _bpm_close(raw_spectral_bpm, global_bpm)
            )
        )
        confident_segments_near_global = [
            estimate
            for estimate in segment_estimates
            if estimate.get("bpm") is not None
            and float(estimate.get("confidence") or 0.0) >= 0.55
            and not estimate.get("is_ambiguous")
            and _bpm_close(estimate.get("bpm"), global_bpm)
        ]
        summary_window_transition = bool(
            global_series_gap >= transition_threshold_bpm
            and len(confident_segments_near_global) < 2
            and not series_median_is_harmonic
        )
        if summary_window_transition:
            transition_detected = True
            if early_bpm is None or not _bpm_close(early_bpm, series_median_bpm):
                early_bpm = series_early_bpm if series_early_bpm is not None else series_median_bpm
            if late_bpm is None or not _bpm_close(late_bpm, global_bpm):
                late_bpm = float(global_bpm)
            bpm_change_magnitude = global_series_gap

    is_nonstationary = transition_detected

    transition_time_s = None
    if transition_detected:
        if adaptive_transition_detected:
            transition_time_s = float(adaptive_transition["best_transition"]["split_time_s"])
        elif bpm_times_s is not None and bpm_values is not None and len(bpm_times_s) >= 2:
            values = np.asarray(bpm_values, dtype=float)
            times = np.asarray(bpm_times_s, dtype=float)
            step_index = int(np.argmax(np.abs(np.diff(values))))
            if abs(float(values[step_index + 1] - values[step_index])) >= transition_threshold_bpm:
                transition_time_s = float(0.5 * (times[step_index] + times[step_index + 1]))
        if transition_time_s is None:
            if mid_bpm is not None and early_bpm is not None and late_bpm is not None:
                mid_to_early = abs(float(mid_bpm) - float(early_bpm))
                mid_to_late = abs(float(mid_bpm) - float(late_bpm))
                if mid_to_early < mid_to_late:
                    transition_time_s = float(2.0 * duration_s / 3.0)
                elif mid_to_late < mid_to_early:
                    transition_time_s = float(duration_s / 3.0)
            if transition_time_s is None:
                transition_time_s = float(duration_s / 2.0)
        if summary_window_transition:
            transition_time_s = float(duration_s / 2.0)

    global_representative = not is_nonstationary
    trend_confidence = 0.0
    if transition_detected:
        if adaptive_transition_detected:
            best_transition = adaptive_transition["best_transition"]
            segment_confidence = 0.5 * (
                float(best_transition.get("before_confidence") or 0.0)
                + float(best_transition.get("after_confidence") or 0.0)
            )
        else:
            segment_confidence = 0.5 * (early_confidence + late_confidence)
        change_score = _clamp01((bpm_change_magnitude or 0.0) / max(2.0 * transition_threshold_bpm, 1e-9))
        series_bonus = 0.10 if series_transition else 0.0
        if summary_window_transition:
            series_bonus = max(series_bonus, 0.10)
            segment_confidence = max(segment_confidence, 0.45)
        if adaptive_transition_detected:
            series_bonus = max(series_bonus, 0.10)
        trend_confidence = _clamp01(0.35 + (0.35 * segment_confidence) + (0.20 * change_score) + series_bonus)
    elif early_bpm is not None and late_bpm is not None:
        segment_confidence = 0.5 * (early_confidence + late_confidence)
        stability_score = 1.0 - _clamp01(abs(float(late_bpm) - float(early_bpm)) / transition_threshold_bpm)
        trend_confidence = _clamp01((0.60 * segment_confidence) + (0.40 * stability_score))

    if global_confidence is None:
        global_confidence = trend_confidence
    global_bpm_confidence = float(global_confidence)
    if not global_representative:
        global_bpm_confidence = min(global_bpm_confidence, 0.35)

    if is_nonstationary:
        recording_type = "nonstationary"
        respiration_pattern = "nonstationary_transition"
        if adaptive_transition_detected:
            nonstationary_reason = (
                f"Best transition split at {transition_time_s:.1f} s separates "
                f"{early_bpm:.2f} BPM before from {late_bpm:.2f} BPM after "
                f"({bpm_change_magnitude:.2f} BPM change), "
                f"exceeding the {transition_threshold_bpm:.2f} BPM transition threshold."
            )
        elif summary_window_transition and series_median_bpm is not None and global_bpm is not None:
            nonstationary_reason = (
                f"Windowed BPM estimates are centered near {series_median_bpm:.2f} BPM while the full-record "
                f"global summary is {float(global_bpm):.2f} BPM; treating the single global BPM as non-representative."
            )
        else:
            nonstationary_reason = (
                f"Early and late segment estimates differ by {bpm_change_magnitude:.2f} BPM, "
                f"exceeding the {transition_threshold_bpm:.2f} BPM transition threshold."
                if bpm_change_magnitude is not None
                else "Segment estimates show a sustained rate transition."
            )
    else:
        recording_type = "stable"
        respiration_pattern = "stable"
        if segment_harmonic_flip_explained:
            nonstationary_reason = (
                "Segment estimates jumped between a rate and its 2x harmonic, but the full-record estimate "
                "supports a stable rate, so this is treated as harmonic ambiguity rather than a transition."
            )
        else:
            nonstationary_reason = "No sustained early-to-late rate shift was detected."

    return {
        "recording_type": recording_type,
        "respiration_pattern": respiration_pattern,
        "is_nonstationary": is_nonstationary,
        "transition_detected": transition_detected,
        "early_bpm": None if early_bpm is None else float(early_bpm),
        "mid_bpm": None if mid_bpm is None else float(mid_bpm),
        "late_bpm": None if late_bpm is None else float(late_bpm),
        "series_early_bpm": series_early_bpm,
        "series_mid_bpm": series_mid_bpm,
        "series_late_bpm": series_late_bpm,
        "bpm_change_magnitude": None if bpm_change_magnitude is None else float(bpm_change_magnitude),
        "transition_time_s": transition_time_s,
        "global_bpm_representative": global_representative,
        "global_bpm_confidence": global_bpm_confidence,
        "trend_confidence": trend_confidence,
        "transition_confidence": trend_confidence if transition_detected else 0.0,
        "nonstationary_reason": nonstationary_reason,
        "segment_estimates": segment_estimates,
        "transition_segment_estimates": []
        if not adaptive_transition_detected
        else [
            adaptive_transition["best_transition"]["before"],
            adaptive_transition["best_transition"]["after"],
        ],
        "transition_search_candidates": adaptive_transition.get("transition_search_candidates", []),
        "adaptive_transition_detected": adaptive_transition_detected,
        "transition_threshold_bpm": transition_threshold_bpm,
        "summary_window_transition": summary_window_transition,
        "segment_harmonic_flip_explained": segment_harmonic_flip_explained,
        "segment_cycle_supported": segment_cycle_supported,
    }


def _derive_reported_transition_segments(features: dict[str, Any]) -> dict[str, Any]:
    if not (features.get("transition_detected") or features.get("is_nonstationary")):
        return {
            "reported_early_bpm": None,
            "reported_mid_bpm": None,
            "reported_late_bpm": None,
            "reported_bpm_change_magnitude": None,
            "reported_transition_pattern": "stable",
            "reported_early_source": None,
            "reported_mid_source": None,
            "reported_late_source": None,
        }

    early_bpm = features.get("early_bpm")
    mid_bpm = features.get("mid_bpm")
    late_bpm = features.get("late_bpm")
    series_early_bpm = features.get("series_early_bpm")
    series_mid_bpm = features.get("series_mid_bpm")
    series_late_bpm = features.get("series_late_bpm")
    fixed_segments = {
        str(segment.get("label")): segment
        for segment in list(features.get("segment_estimates") or [])
        if segment.get("label") is not None
    }
    transition_time_s = features.get("transition_time_s")
    mid_segment = fixed_segments.get("mid")
    transition_inside_mid = False
    if transition_time_s is not None and mid_segment is not None:
        transition_inside_mid = bool(
            float(mid_segment.get("start_s", 0.0))
            < float(transition_time_s)
            < float(mid_segment.get("end_s", 0.0))
        )

    def segment_report_bpm(segment: dict[str, Any] | None) -> float | None:
        if not segment:
            return None
        value = segment.get("selected_bpm")
        if value is None:
            value = segment.get("bpm")
        return None if value is None else float(value)

    def segment_reportable(segment: dict[str, Any] | None) -> bool:
        return bool(
            segment
            and segment_report_bpm(segment) is not None
            and float(segment.get("confidence") or 0.0) >= 0.55
            and not segment.get("is_ambiguous")
        )

    fixed_early = fixed_segments.get("early")
    fixed_mid = fixed_segments.get("mid")
    fixed_late = fixed_segments.get("late")
    fixed_early_bpm = segment_report_bpm(fixed_early)
    fixed_mid_bpm = segment_report_bpm(fixed_mid)
    fixed_late_bpm = segment_report_bpm(fixed_late)

    middle_plateau = False
    series_plateau = False
    plateau_early = series_early_bpm if series_early_bpm is not None else early_bpm
    plateau_mid = series_mid_bpm if series_mid_bpm is not None else mid_bpm
    plateau_late = series_late_bpm if series_late_bpm is not None else late_bpm
    if plateau_early is not None and plateau_mid is not None and plateau_late is not None:
        threshold = max(4.0, 0.25 * max(min(float(plateau_early), float(plateau_late)), 1.0))
        increasing = float(plateau_early) < float(plateau_mid) < float(plateau_late)
        decreasing = float(plateau_early) > float(plateau_mid) > float(plateau_late)
        separated = (
            abs(float(plateau_mid) - float(plateau_early)) >= threshold
            and abs(float(plateau_late) - float(plateau_mid)) >= threshold
        )
        series_plateau = bool(
            series_early_bpm is not None
            and series_mid_bpm is not None
            and series_late_bpm is not None
            and (increasing or decreasing)
            and separated
        )
        fixed_plateau_confident = all(
            fixed_segments.get(label) is not None
            and float(fixed_segments[label].get("confidence") or 0.0) >= 0.75
            and not fixed_segments[label].get("is_ambiguous")
            for label in ("early", "mid", "late")
        )
        middle_plateau = bool(
            (increasing or decreasing)
            and separated
            and (not transition_inside_mid or fixed_plateau_confident or series_plateau)
        )

    if middle_plateau and early_bpm is not None and mid_bpm is not None and late_bpm is not None:
        report_early = fixed_early_bpm if fixed_early_bpm is not None else early_bpm
        report_mid = fixed_mid_bpm if fixed_mid_bpm is not None else mid_bpm
        report_late = fixed_late_bpm if fixed_late_bpm is not None else late_bpm
        report_change = (
            None
            if report_early is None or report_late is None
            else abs(float(report_late) - float(report_early))
        )
        return {
            "reported_early_bpm": report_early,
            "reported_mid_bpm": report_mid,
            "reported_late_bpm": report_late,
            "reported_bpm_change_magnitude": report_change,
            "reported_transition_pattern": "multi_stage",
            "reported_early_source": "early",
            "reported_mid_source": "mid",
            "reported_late_source": "late",
        }

    transition_segments = list(features.get("transition_segment_estimates") or [])
    if len(transition_segments) >= 2:
        before_bpm = transition_segments[0].get("bpm")
        after_bpm = transition_segments[1].get("bpm")
        before_report_bpm = before_bpm
        after_report_bpm = after_bpm
        before_source = transition_segments[0].get("label", "transition_before")
        after_source = transition_segments[1].get("label", "transition_after")

        if before_bpm is not None and after_bpm is not None:
            increasing_transition = float(after_bpm) >= float(before_bpm)
            transition_threshold = float(features.get("transition_threshold_bpm") or 4.0)
            early_mid_matches_direction = bool(
                fixed_early_bpm is not None
                and fixed_mid_bpm is not None
                and abs(fixed_mid_bpm - fixed_early_bpm) >= transition_threshold
                and ((fixed_mid_bpm > fixed_early_bpm) == increasing_transition)
            )
            late_is_unusable_or_contradictory = bool(
                not segment_reportable(fixed_late)
                or fixed_late_bpm is None
                or (
                    increasing_transition
                    and fixed_mid_bpm is not None
                    and fixed_late_bpm is not None
                    and fixed_late_bpm < (fixed_mid_bpm - transition_threshold)
                )
                or (
                    not increasing_transition
                    and fixed_mid_bpm is not None
                    and fixed_late_bpm is not None
                    and fixed_late_bpm > (fixed_mid_bpm + transition_threshold)
                )
            )
            if (
                transition_inside_mid
                and segment_reportable(fixed_early)
                and segment_reportable(fixed_mid)
                and early_mid_matches_direction
                and late_is_unusable_or_contradictory
            ):
                before_report_bpm = fixed_early_bpm
                after_report_bpm = fixed_mid_bpm
                before_source = "early"
                after_source = "mid"

        change = (
            None
            if before_report_bpm is None or after_report_bpm is None
            else abs(float(after_report_bpm) - float(before_report_bpm))
        )
        return {
            "reported_early_bpm": before_report_bpm,
            "reported_mid_bpm": None,
            "reported_late_bpm": after_report_bpm,
            "reported_bpm_change_magnitude": change,
            "reported_transition_pattern": "two_stage",
            "reported_early_source": before_source,
            "reported_mid_source": None,
            "reported_late_source": after_source,
        }

    return {
        "reported_early_bpm": early_bpm,
        "reported_mid_bpm": None,
        "reported_late_bpm": late_bpm,
        "reported_bpm_change_magnitude": features.get("bpm_change_magnitude"),
        "reported_transition_pattern": "two_stage",
        "reported_early_source": "early",
        "reported_mid_source": None,
        "reported_late_source": "late",
    }


def build_respiration_features(
    signal: dict[str, Any],
    *,
    spectrogram: dict[str, Any] | None = None,
    min_bpm: float = 4.0,
    max_bpm: float = 30.0,
) -> dict[str, Any]:
    """Assemble the notebook's one-stop summary object for a recording."""

    global_rate = estimate_breathing_rate_multiwindow(signal, min_bpm=min_bpm, max_bpm=max_bpm)
    zscore = np.asarray(signal["zscore_signal"], dtype=float)
    centered = np.asarray(signal["centered_signal"], dtype=float)
    chosen_window = dict(global_rate.get("chosen_window", {}))

    features = {
        "selected_center_bin": int(signal["center_bin"]),
        "selected_center_range_m": float(signal["center_range_m"]),
        "selected_bins": list(signal["selected_bins"]),
        "selected_ranges_m": list(signal["selected_ranges_m"]),
        "num_samples": len(signal["times_s"]),
        "duration_s": global_rate["duration_s"],
        "sample_rate_hz": global_rate["sample_rate_hz"],
        "signal_peak_to_peak_zscore": float(np.ptp(zscore)) if zscore.size else 0.0,
        "signal_rms_centered": float(np.sqrt(np.mean(centered * centered))) if centered.size else 0.0,
        "global_bpm": global_rate["selected_bpm"],
        "global_frequency_hz": global_rate["selected_frequency_hz"],
        "global_raw_peak_bpm": global_rate["raw_spectral_bpm"],
        "global_raw_peak_frequency_hz": global_rate["raw_spectral_frequency_hz"],
        "global_used_subharmonic": global_rate["used_subharmonic"],
        "global_used_autocorrelation_support": global_rate["used_autocorrelation_support"],
        "global_autocorrelation_bpm": global_rate["autocorrelation_bpm"],
        "global_confidence_ratio": global_rate["confidence_ratio"],
        "global_peak_candidates": global_rate["top_peaks"],
        "selected_bpm": global_rate["selected_bpm"],
        "selected_frequency_hz": global_rate["selected_frequency_hz"],
        "raw_spectral_bpm": global_rate["raw_spectral_bpm"],
        "raw_spectral_frequency_hz": global_rate["raw_spectral_frequency_hz"],
        "raw_peak_bpm": global_rate["raw_peak_bpm"],
        "raw_peak_frequency_hz": global_rate["raw_peak_frequency_hz"],
        "subharmonic_candidate_bpm": global_rate["subharmonic_candidate_bpm"],
        "autocorrelation_bpm": global_rate["autocorrelation_bpm"],
        "time_domain_bpm": global_rate["time_domain_bpm"],
        "time_domain_period_s": global_rate["time_domain_period_s"],
        "time_domain_num_breaths": global_rate["time_domain_num_breaths"],
        "time_domain_confidence": global_rate["time_domain_confidence"],
        "time_domain_interval_iqr_s": global_rate["time_domain"].get("interval_iqr_s"),
        "harmonic_suspected": global_rate["harmonic_suspected"],
        "harmonic_ratio": global_rate["harmonic_ratio"],
        "subharmonic_power_ratio": global_rate["subharmonic_power_ratio"],
        "used_subharmonic": global_rate["used_subharmonic"],
        "used_autocorrelation_support": global_rate["used_autocorrelation_support"],
        "used_time_domain_support": global_rate["used_time_domain_support"],
        "selected_bpm_source": global_rate["selected_bpm_source"],
        "is_ambiguous": global_rate["is_ambiguous"],
        "decision_reason": global_rate["decision_reason"],
        "signal_quality_score": global_rate["signal_quality_score"],
        "spectral_confidence": global_rate["spectral_confidence"],
        "temporal_consistency_score": global_rate["temporal_consistency_score"],
        "harmonic_ambiguity_score": global_rate["harmonic_ambiguity_score"],
        "final_confidence": global_rate["final_confidence"],
        "confidence": global_rate["final_confidence"],
        "chosen_window_seconds": chosen_window.get("window_seconds"),
        "chosen_hop_seconds": chosen_window.get("hop_seconds"),
        "frequency_resolution_bpm": chosen_window.get("frequency_resolution_bpm"),
        "frequency_resolution_hz": chosen_window.get("frequency_resolution_hz"),
        "reliable_window_for_min_bpm": chosen_window.get("reliable_for_min_bpm"),
        "window_decision_reason": chosen_window.get("decision_reason"),
        "window_estimates": global_rate["window_estimates"],
        "selection_method": signal.get("selection_method"),
        "selection_reason": signal.get("selection_reason", ""),
        "selection_candidate_bins": list(signal.get("selection_candidate_bins", [])),
        "selection_cluster_bins": list(signal.get("selection_cluster_bins", [])),
        "selection_bin_scores": list(signal.get("selection_bin_scores", [])),
        "selection_trend_candidate_scores": list(signal.get("selection_trend_candidate_scores", [])),
        "selection_trend_selected": signal.get("selection_trend_selected"),
        "range_motion_detected": bool(signal.get("range_motion_detected", False)),
        "range_motion_reason": signal.get("range_motion_reason", ""),
        "tracked_range_early_m": signal.get("tracked_range_early_m"),
        "tracked_range_mid_m": signal.get("tracked_range_mid_m"),
        "tracked_range_late_m": signal.get("tracked_range_late_m"),
        "tracked_range_change_m": signal.get("tracked_range_change_m"),
        "tracked_range_abs_change_m": signal.get("tracked_range_abs_change_m"),
        "tracked_range_p10_m": signal.get("tracked_range_p10_m"),
        "tracked_range_p90_m": signal.get("tracked_range_p90_m"),
        "tracked_range_span_m": signal.get("tracked_range_span_m"),
        "tracked_range_motion_score_m": signal.get("tracked_range_motion_score_m"),
        "tracked_range_preview_times_s": list(signal.get("tracked_range_preview_times_s", [])),
        "tracked_range_preview_m": list(signal.get("tracked_range_preview_m", [])),
        "selected_range_matches_tracked_target": bool(signal.get("selected_range_matches_tracked_target", True)),
        "dynamic_range_tracking_used": bool(signal.get("dynamic_range_tracking_used", False)),
        "dynamic_range_detrend_seconds": signal.get("dynamic_range_detrend_seconds"),
    }

    # Spectrogram-derived BPM summaries are optional; the global estimator can
    # still run when only a single extracted signal is available.
    if spectrogram is not None:
        bpm_series = estimate_bpm_series_from_spectrogram(spectrogram, min_bpm=min_bpm, max_bpm=max_bpm)
        features.update(
            {
                "bpm_over_time": bpm_series["bpm_values"],
                "bpm_median": bpm_series["median_bpm"],
                "bpm_p10": bpm_series["p10_bpm"],
                "bpm_p90": bpm_series["p90_bpm"],
                "bpm_min": bpm_series["min_bpm"],
                "bpm_max": bpm_series["max_bpm"],
                "bpm_times_s": bpm_series["times_s"],
            }
        )

    # Merge successive interpretation layers: first estimate time structure,
    # then label quality and ambiguity, then normalize the reporting fields
    # consumed by the notebook and handoff docs.
    trend = estimate_respiration_trend(
        signal,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        bpm_times_s=features.get("bpm_times_s"),
        bpm_values=features.get("bpm_over_time"),
        global_bpm=features.get("global_bpm"),
        global_confidence=features.get("final_confidence"),
        raw_spectral_bpm=features.get("raw_spectral_bpm"),
    )
    features.update(trend)
    if features.get("is_nonstationary"):
        features["global_bpm_is_ambiguous"] = features.get("is_ambiguous", False)
        features["is_ambiguous"] = False

    features.update(assess_respiration_quality(features))
    features.update(_derive_reported_transition_segments(features))

    return features


def assess_respiration_quality(features: dict[str, Any]) -> dict[str, Any]:
    """Turn raw estimator outputs into operational quality labels and flags."""

    confidence_ratio = float(features.get("global_confidence_ratio") or 0.0)
    raw_peak_bpm = float(features.get("global_raw_peak_bpm") or features.get("global_bpm") or 0.0)
    selected_bpm = float(features.get("global_bpm") or 0.0)
    harmonic_gap_bpm = abs(raw_peak_bpm - selected_bpm)
    time_domain_bpm = features.get("time_domain_bpm")
    autocorrelation_bpm = features.get("autocorrelation_bpm", features.get("global_autocorrelation_bpm"))
    time_domain_confidence = float(features.get("time_domain_confidence") or 0.0)
    signal_quality_score = float(features.get("signal_quality_score") or 0.0)
    spectral_confidence = float(features.get("spectral_confidence") or 0.0)
    temporal_consistency_score = float(features.get("temporal_consistency_score") or 0.0)
    harmonic_ambiguity_score = float(features.get("harmonic_ambiguity_score") or 1.0)
    is_ambiguous = bool(features.get("is_ambiguous"))
    is_nonstationary = bool(features.get("is_nonstationary"))
    transition_detected = bool(features.get("transition_detected"))
    trend_confidence = float(features.get("trend_confidence") or 0.0)
    transition_confidence = float(features.get("transition_confidence") or 0.0)
    global_bpm_representative = bool(features.get("global_bpm_representative", True))
    global_bpm_confidence = float(features.get("global_bpm_confidence", features.get("final_confidence") or 0.0))
    range_motion_detected = bool(features.get("range_motion_detected", False))
    selected_range_matches_tracked_target = bool(features.get("selected_range_matches_tracked_target", True))
    dynamic_range_tracking_used = bool(features.get("dynamic_range_tracking_used", False))
    uncompensated_range_motion = bool(range_motion_detected and not dynamic_range_tracking_used)

    flags: list[str] = []
    strong_temporal_conflict = False

    if range_motion_detected:
        flags.append("large subject range motion")
        if dynamic_range_tracking_used:
            flags.append("dynamic range tracking used")
        else:
            global_bpm_representative = False
            global_bpm_confidence = min(global_bpm_confidence, 0.25)
            temporal_consistency_score = min(temporal_consistency_score, 0.35)
            spectral_confidence = min(spectral_confidence, 0.45)
        if not selected_range_matches_tracked_target and not dynamic_range_tracking_used:
            flags.append("fixed range gate missed moving target")

    if is_nonstationary:
        flags.append("nonstationary breathing")
        if transition_detected:
            flags.append("rate transition detected")
        is_ambiguous = False
        global_bpm_representative = False
        global_bpm_confidence = min(global_bpm_confidence, 0.35)
        temporal_consistency_score = max(temporal_consistency_score, trend_confidence)

    if features.get("global_used_subharmonic") or features.get("used_subharmonic"):
        flags.append("harmonic-corrected")

    if is_ambiguous and not is_nonstationary:
        flags.append("ambiguous harmonic evidence")

    if confidence_ratio and confidence_ratio < 1.25:
        flags.append("low spectral separation")

    if time_domain_confidence and time_domain_confidence < 0.45:
        flags.append("weak time-domain cycle validation")

    if features.get("reliable_window_for_min_bpm") is False:
        flags.append("analysis window short for slow BPM")

    bpm_spread = None
    if "bpm_p10" in features and "bpm_p90" in features:
        bpm_spread = float(features["bpm_p90"] - features["bpm_p10"])
        if bpm_spread > 8.0:
            flags.append("wide BPM spread from transition" if is_nonstationary else "wide BPM spread")
            if not is_nonstationary:
                temporal_consistency_score = min(temporal_consistency_score, 0.50)
        elif bpm_spread > 5.0:
            flags.append("moderate BPM spread from transition" if is_nonstationary else "moderate BPM spread")
            if not is_nonstationary:
                temporal_consistency_score = min(temporal_consistency_score, 0.65)

    if "bpm_median" in features:
        median_bpm = float(features["bpm_median"])
        median_matches_selected = _bpm_close(median_bpm, selected_bpm)
        median_is_explained_harmonic = (
            bool(features.get("harmonic_suspected"))
            and _is_harmonic_pair(median_bpm, selected_bpm)
            and (
                _bpm_close(time_domain_bpm, selected_bpm)
                or _bpm_close(autocorrelation_bpm, selected_bpm)
            )
        )
        if is_nonstationary:
            pass
        elif median_matches_selected:
            if _bpm_close(time_domain_bpm, selected_bpm) or _bpm_close(autocorrelation_bpm, selected_bpm):
                temporal_consistency_score = max(temporal_consistency_score, 0.82)
        elif median_is_explained_harmonic:
            flags.append("spectrogram median follows likely harmonic")
            temporal_consistency_score = min(max(temporal_consistency_score, 0.62), 0.78)
            harmonic_ambiguity_score = min(harmonic_ambiguity_score, 0.78)
        else:
            flags.append("global/median mismatch")
            strong_temporal_conflict = True
            is_ambiguous = True
            temporal_consistency_score = min(temporal_consistency_score, 0.35)
            harmonic_ambiguity_score = min(harmonic_ambiguity_score, 0.35)

    if harmonic_gap_bpm > 6.0 and not features.get("global_used_subharmonic"):
        flags.append("strong higher harmonic present")
        harmonic_ambiguity_score = min(harmonic_ambiguity_score, 0.70)

    if is_nonstationary:
        final_confidence = _clamp01(
            (0.25 * signal_quality_score)
            + (0.15 * spectral_confidence)
            + (0.50 * max(trend_confidence, transition_confidence))
            + (0.10 * harmonic_ambiguity_score)
        )
    else:
        final_confidence = _combine_confidence_scores(
            signal_quality_score=signal_quality_score,
            spectral_confidence=spectral_confidence,
            temporal_consistency_score=temporal_consistency_score,
            harmonic_ambiguity_score=harmonic_ambiguity_score,
            is_ambiguous=is_ambiguous,
        )
    if (features.get("global_used_subharmonic") or features.get("used_subharmonic")) and not is_ambiguous and not is_nonstationary:
        final_confidence = min(final_confidence, 0.82)
    if strong_temporal_conflict and not is_nonstationary:
        final_confidence = min(final_confidence, 0.40)
    if uncompensated_range_motion:
        # A single fixed range gate is not a valid respiration source when the
        # subject moves a large distance through the capture.
        final_confidence = min(final_confidence, 0.30 if selected_range_matches_tracked_target else 0.20)
        is_ambiguous = True
        transition_detected = False
        is_nonstationary = False
        trend_confidence = 0.0
        transition_confidence = 0.0
        flags = [
            flag
            for flag in flags
            if flag
            not in {
                "nonstationary breathing",
                "rate transition detected",
                "wide BPM spread from transition",
                "moderate BPM spread from transition",
            }
        ]
        flags.append("respiration estimate motion-contaminated")
    elif dynamic_range_tracking_used:
        final_confidence = min(final_confidence, 0.82)

    if signal_quality_score <= 0.0 or final_confidence <= 0.05:
        recording_type = "failed"
        respiration_pattern = "failed"
    elif uncompensated_range_motion:
        recording_type = "failed"
        respiration_pattern = "range_motion_contaminated"
    elif is_nonstationary:
        recording_type = "nonstationary"
        respiration_pattern = "nonstationary_transition" if transition_detected else "nonstationary"
    else:
        recording_type = "stable"
        if features.get("harmonic_suspected") or is_ambiguous:
            respiration_pattern = "stable_harmonic_ambiguity"
        else:
            respiration_pattern = "stable_reliable"

    if final_confidence >= 0.75:
        quality_label = "High"
    elif final_confidence >= 0.45:
        quality_label = "Medium"
    else:
        quality_label = "Low"

    return {
        "quality_label": quality_label,
        "quality_score": final_confidence,
        "final_confidence": final_confidence,
        "confidence": final_confidence,
        "signal_quality_score": signal_quality_score,
        "spectral_confidence": spectral_confidence,
        "temporal_consistency_score": temporal_consistency_score,
        "harmonic_ambiguity_score": harmonic_ambiguity_score,
        "is_ambiguous": is_ambiguous,
        "is_nonstationary": is_nonstationary,
        "transition_detected": transition_detected,
        "recording_type": recording_type,
        "respiration_pattern": respiration_pattern,
        "global_bpm_representative": global_bpm_representative,
        "global_bpm_confidence": global_bpm_confidence,
        "trend_confidence": trend_confidence,
        "transition_confidence": transition_confidence,
        "transition_time_s": None if uncompensated_range_motion else features.get("transition_time_s"),
        "bpm_change_magnitude": None if uncompensated_range_motion else features.get("bpm_change_magnitude"),
        "quality_flags": flags,
        "bpm_spread": bpm_spread,
    }


def format_respiration_feature_report(
    features: dict[str, Any],
    *,
    bpm_preview_count: int = 10,
) -> str:
    """Render the features dictionary into the markdown report shown in the notebook."""

    def _fmt_float(value: float | None, digits: int = 3) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

    show_transition_segments = bool(features.get("transition_detected") or features.get("is_nonstationary"))
    report_early_bpm = features.get("reported_early_bpm") if show_transition_segments else None
    report_mid_bpm = features.get("reported_mid_bpm") if show_transition_segments else None
    report_late_bpm = features.get("reported_late_bpm") if show_transition_segments else None
    report_change_magnitude = features.get("reported_bpm_change_magnitude") if show_transition_segments else None

    lines = [
        "## Respiration Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Selected center bin | {features['selected_center_bin']} |",
        f"| Selected center range | {_fmt_float(features['selected_center_range_m'])} m |",
        f"| Selected bins | {', '.join(str(bin_index) for bin_index in features['selected_bins'])} |",
        f"| Range-bin selection | {features.get('selection_method') or '-'} |",
        f"| Large range motion detected | {'Yes' if features.get('range_motion_detected') else 'No'} |",
        f"| Dynamic range tracking used | {'Yes' if features.get('dynamic_range_tracking_used') else 'No'} |",
        f"| Tracked range early/mid/late | {_fmt_float(features.get('tracked_range_early_m'), 2)} / {_fmt_float(features.get('tracked_range_mid_m'), 2)} / {_fmt_float(features.get('tracked_range_late_m'), 2)} m |",
        f"| Tracked range span | {_fmt_float(features.get('tracked_range_span_m'), 2)} m |",
        f"| Fixed gate matches tracked target | {'Yes' if features.get('selected_range_matches_tracked_target', True) else 'No'} |",
        f"| Duration | {_fmt_float(features['duration_s'], 1)} s |",
        f"| Sample rate | {_fmt_float(features['sample_rate_hz'], 3)} Hz |",
        f"| Recording type | {features.get('recording_type', '-')} |",
        f"| Respiration pattern | {features.get('respiration_pattern', '-')} |",
        f"| Reported transition pattern | {features.get('reported_transition_pattern', '-')} |",
        f"| Final selected RR | {_fmt_float(features.get('selected_bpm', features['global_bpm']), 2)} BPM |",
        f"| Final selected frequency | {_fmt_float(features.get('selected_frequency_hz', features['global_frequency_hz']), 4)} Hz |",
        f"| Global BPM representative | {'Yes' if features.get('global_bpm_representative', True) else 'No'} |",
        f"| Global BPM confidence | {_fmt_float(features.get('global_bpm_confidence'), 2)} |",
        f"| Early BPM | {_fmt_float(report_early_bpm, 2)} |",
        f"| Mid BPM | {_fmt_float(report_mid_bpm, 2)} |",
        f"| Late BPM | {_fmt_float(report_late_bpm, 2)} |",
        f"| BPM change magnitude | {_fmt_float(report_change_magnitude, 2)} |",
        f"| Transition detected | {'Yes' if features.get('transition_detected') else 'No'} |",
        f"| Transition time | {_fmt_float(features.get('transition_time_s'), 1)} s |",
        f"| Trend confidence | {_fmt_float(features.get('trend_confidence'), 2)} |",
        f"| Transition confidence | {_fmt_float(features.get('transition_confidence'), 2)} |",
        f"| Raw strongest spectral peak | {_fmt_float(features.get('raw_spectral_bpm', features.get('global_raw_peak_bpm')), 2)} BPM |",
        f"| Raw strongest frequency | {_fmt_float(features.get('raw_spectral_frequency_hz', features.get('global_raw_peak_frequency_hz')), 4)} Hz |",
        f"| Autocorrelation estimate | {_fmt_float(features.get('autocorrelation_bpm', features.get('global_autocorrelation_bpm')), 2)} BPM |",
        f"| Time-domain estimate | {_fmt_float(features.get('time_domain_bpm'), 2)} BPM |",
        f"| Time-domain period | {_fmt_float(features.get('time_domain_period_s'), 2)} s |",
        f"| Time-domain breaths detected | {features.get('time_domain_num_breaths', '-')} |",
        f"| Time-domain confidence | {_fmt_float(features.get('time_domain_confidence'), 2)} |",
        f"| Harmonic suspected | {'Yes' if features.get('harmonic_suspected') else 'No'} |",
        f"| Harmonic correction | {'Applied' if features.get('used_subharmonic', features.get('global_used_subharmonic')) else 'Not applied'} |",
        f"| Subharmonic power ratio | {_fmt_float(features.get('subharmonic_power_ratio'), 2)} |",
        f"| Ambiguous result | {'Yes' if features.get('is_ambiguous') else 'No'} |",
        f"| Selected BPM source | {features.get('selected_bpm_source', '-')} |",
        f"| Quality | {features.get('quality_label', '-')} |",
        f"| Final confidence | {_fmt_float(features.get('final_confidence', features.get('quality_score')), 2)} |",
        f"| Signal quality score | {_fmt_float(features.get('signal_quality_score'), 2)} |",
        f"| Spectral confidence | {_fmt_float(features.get('spectral_confidence'), 2)} |",
        f"| Temporal consistency score | {_fmt_float(features.get('temporal_consistency_score'), 2)} |",
        f"| Harmonic ambiguity score | {_fmt_float(features.get('harmonic_ambiguity_score'), 2)} |",
        f"| Peak confidence ratio | {_fmt_float(features.get('global_confidence_ratio'), 2)} |",
        f"| Chosen window | {_fmt_float(features.get('chosen_window_seconds'), 1)} s |",
        f"| Chosen hop | {_fmt_float(features.get('chosen_hop_seconds'), 1)} s |",
        f"| Window frequency resolution | {_fmt_float(features.get('frequency_resolution_bpm'), 2)} BPM |",
        f"| Window reliable for slow limit | {'Yes' if features.get('reliable_window_for_min_bpm') else 'No'} |",
        f"| Signal peak-to-peak | {_fmt_float(features['signal_peak_to_peak_zscore'], 2)} z-score |",
        f"| Signal RMS | {_fmt_float(features['signal_rms_centered'], 2)} centered units |",
    ]

    decision_reason = features.get("decision_reason")
    if decision_reason:
        lines.extend(["", f"Decision: {decision_reason}"])

    if features.get("is_nonstationary"):
        lines.extend(
            [
                "",
                (
                    "Nonstationary recording: the global BPM is a summary of the whole capture, "
                    "not a single representative breathing rate."
                ),
            ]
        )
        nonstationary_reason = features.get("nonstationary_reason")
        if nonstationary_reason:
            lines.append(f"Trend: {nonstationary_reason}")

    window_reason = features.get("window_decision_reason")
    if window_reason:
        lines.extend(["", f"Window choice: {window_reason}"])

    selection_reason = features.get("selection_reason")
    if selection_reason:
        lines.extend(["", f"Range-bin choice: {selection_reason}"])

    range_motion_reason = features.get("range_motion_reason")
    if range_motion_reason:
        lines.extend(["", f"Range-motion diagnostic: {range_motion_reason}"])

    quality_flags = list(features.get("quality_flags", []))
    if quality_flags:
        lines.extend(
            [
                "",
                "Quality flags: " + ", ".join(quality_flags),
            ]
        )

    if "bpm_median" in features:
        lines.extend(
            [
                "",
                "## Time-Varying BPM Summary",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Median BPM | {_fmt_float(features['bpm_median'], 2)} |",
                f"| 10th percentile BPM | {_fmt_float(features['bpm_p10'], 2)} |",
                f"| 90th percentile BPM | {_fmt_float(features['bpm_p90'], 2)} |",
                f"| BPM spread | {_fmt_float(features.get('bpm_spread'), 2)} |",
                f"| Min BPM | {_fmt_float(features['bpm_min'], 2)} |",
                f"| Max BPM | {_fmt_float(features['bpm_max'], 2)} |",
            ]
        )

    transition_segment_estimates = list(features.get("transition_segment_estimates", []))
    segment_estimates = list(features.get("segment_estimates", []))
    if (transition_segment_estimates or segment_estimates) and show_transition_segments:
        lines.extend(
            [
                "",
                "## Segmented BPM Summary",
                "",
                "| Segment | Time span (s) | BPM | Confidence | Raw spectral BPM | Time-domain BPM |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for segment in [*transition_segment_estimates, *segment_estimates]:
            lines.append(
                f"| {segment.get('label', '-')} | "
                f"{_fmt_float(segment.get('start_s'), 1)}-{_fmt_float(segment.get('end_s'), 1)} | "
                f"{_fmt_float(segment.get('bpm'), 2)} | "
                f"{_fmt_float(segment.get('confidence'), 2)} | "
                f"{_fmt_float(segment.get('raw_spectral_bpm'), 2)} | "
                f"{_fmt_float(segment.get('time_domain_bpm'), 2)} |"
            )
    elif transition_segment_estimates or segment_estimates:
        lines.extend(
            [
                "",
                (
                    "Segmented BPM diagnostics were computed, but no sustained transition was detected; "
                    "the stable full-record BPM is the reported result."
                ),
            ]
        )

    window_estimates = list(features.get("window_estimates", []))
    if window_estimates:
        lines.extend(
            [
                "",
                "## Multi-Window Estimates",
                "",
                "| Source | Window (s) | Median BPM | BPM range | Reliable for slow limit |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for estimate in window_estimates:
            range_text = "-"
            if estimate.get("p10_bpm") is not None and estimate.get("p90_bpm") is not None:
                range_text = f"{_fmt_float(estimate.get('p10_bpm'), 2)}-{_fmt_float(estimate.get('p90_bpm'), 2)}"
            median_bpm = estimate.get("median_bpm", estimate.get("raw_peak_bpm"))
            lines.append(
                f"| {estimate.get('source', '-')} | {_fmt_float(estimate.get('window_seconds'), 1)} | "
                f"{_fmt_float(median_bpm, 2)} | {range_text} | "
                f"{'Yes' if estimate.get('reliable_for_min_bpm') else 'No'} |"
            )

    peak_candidates = list(features.get("global_peak_candidates", []))
    if peak_candidates:
        top_power = max(peak["power"] for peak in peak_candidates) or 1.0
        lines.extend(
            [
                "",
                "## Strongest Global Peaks",
                "",
                "| Rank | BPM | Frequency (Hz) | Relative Power |",
                "| --- | --- | --- | --- |",
            ]
        )
        for rank, peak in enumerate(peak_candidates, start=1):
            relative_power = peak["power"] / top_power
            lines.append(
                f"| {rank} | {_fmt_float(peak['bpm'], 2)} | {_fmt_float(peak['frequency_hz'], 4)} | {_fmt_float(relative_power, 3)} |"
            )

    bpm_values = list(features.get("bpm_over_time", []))
    bpm_times = list(features.get("bpm_times_s", []))
    if bpm_values and bpm_times:
        lines.extend(
            [
                "",
                f"## First {min(bpm_preview_count, len(bpm_values))} BPM-Over-Time Windows",
                "",
                "| Window center (s) | BPM |",
                "| --- | --- |",
            ]
        )
        for time_s, bpm in list(zip(bpm_times, bpm_values))[:bpm_preview_count]:
            lines.append(f"| {_fmt_float(time_s, 1)} | {_fmt_float(bpm, 2)} |")

    return "\n".join(lines)


def export_respiration_signal_csv(signal: dict[str, Any], destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_s", "frame_number", "raw_signal", "centered_signal", "zscore_signal"],
        )
        writer.writeheader()
        for time_s, frame_number, raw_value, centered_value, zscore_value in zip(
            signal["times_s"],
            signal["frame_numbers"],
            signal["raw_signal"],
            signal["centered_signal"],
            signal["zscore_signal"],
        ):
            writer.writerow(
                {
                    "time_s": time_s,
                    "frame_number": frame_number,
                    "raw_signal": raw_value,
                    "centered_signal": centered_value,
                    "zscore_signal": zscore_value,
                }
            )
    return destination_path


def build_respiration_overview(
    bundle: CaptureBundle,
    *,
    center_bin: int | None = None,
    center_range_m: float | None = None,
    bin_half_width: int = 1,
    min_bin: int = 3,
    max_bin: int | None = None,
    top_k: int = 8,
    selection_scoring: str = "respiration-band-trend",
) -> dict[str, Any]:
    """Return a compact extraction summary without running the full feature stack."""

    bin_summary = summarize_range_bins(bundle, min_bin=min_bin, max_bin=max_bin, top_k=top_k)
    signal = extract_respiration_signal(
        bundle,
        center_bin=center_bin,
        center_range_m=center_range_m,
        bin_half_width=bin_half_width,
        min_bin=min_bin,
        max_bin=max_bin,
        selection_scoring=selection_scoring,
    )
    return {
        "frame_period_s": signal["frame_period_s"],
        "range_resolution_m": signal["range_resolution_m"],
        "top_variable_bins": bin_summary["top_variable_bins"],
        "selected_center_bin": signal["center_bin"],
        "selected_center_range_m": signal["center_range_m"],
        "selected_bins": signal["selected_bins"],
        "selected_ranges_m": signal["selected_ranges_m"],
        "selection_method": signal["selection_method"],
        "selection_candidate_bins": signal["selection_candidate_bins"],
        "selection_cluster_bins": signal["selection_cluster_bins"],
        "selection_cluster_score": signal["selection_cluster_score"],
        "selection_weighted_center_bin": signal["selection_weighted_center_bin"],
        "selection_reason": signal.get("selection_reason", ""),
        "selection_bin_scores": signal.get("selection_bin_scores", []),
        "selection_trend_candidate_scores": signal.get("selection_trend_candidate_scores", []),
        "selection_trend_selected": signal.get("selection_trend_selected"),
        "range_motion_detected": signal.get("range_motion_detected", False),
        "range_motion_reason": signal.get("range_motion_reason", ""),
        "tracked_range_early_m": signal.get("tracked_range_early_m"),
        "tracked_range_mid_m": signal.get("tracked_range_mid_m"),
        "tracked_range_late_m": signal.get("tracked_range_late_m"),
        "tracked_range_span_m": signal.get("tracked_range_span_m"),
        "selected_range_matches_tracked_target": signal.get("selected_range_matches_tracked_target", True),
        "dynamic_range_tracking_used": signal.get("dynamic_range_tracking_used", False),
        "signal_head": signal["raw_signal"][:10],
        "centered_signal_head": signal["centered_signal"][:10],
    }
