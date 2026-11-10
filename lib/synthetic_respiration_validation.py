"""Synthetic signals and regression helpers for respiration-estimator tuning.

These helpers generate notebook-compatible signal dictionaries so changes in the
respiration heuristics can be tested without depending on a specific real-world
capture file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .respiration_pipeline import build_respiration_features, estimate_breathing_rate_multiwindow


def generate_synthetic_respiration_signal(
    bpm: float,
    *,
    duration_s: float = 90.0,
    sample_rate_hz: float = 20.0,
    fundamental_amplitude: float = 1.0,
    second_harmonic_amplitude: float = 0.0,
    noise_std: float = 0.10,
    drift_amplitude: float = 0.20,
    seed: int = 7,
) -> dict[str, Any]:
    """Build a notebook-compatible synthetic respiration signal dictionary."""
    times_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    frequency_hz = bpm / 60.0
    rng = np.random.default_rng(seed)
    samples = (
        fundamental_amplitude * np.sin(2.0 * np.pi * frequency_hz * times_s)
        + second_harmonic_amplitude * np.sin(2.0 * np.pi * 2.0 * frequency_hz * times_s + 0.4)
        + drift_amplitude * np.sin(2.0 * np.pi * 0.02 * times_s)
        + noise_std * rng.normal(size=times_s.size)
    )
    centered = samples - np.mean(samples)
    std = float(np.std(centered))
    zscore = centered / std if std > 0.0 else centered
    return {
        "times_s": times_s.tolist(),
        "frame_period_s": float(1.0 / sample_rate_hz),
        "raw_signal": samples.tolist(),
        "centered_signal": centered.tolist(),
        "zscore_signal": zscore.tolist(),
        "center_bin": 0,
        "center_range_m": 0.0,
        "selected_bins": [0],
        "selected_ranges_m": [0.0],
    }


def generate_synthetic_transition_respiration_signal(
    early_bpm: float,
    late_bpm: float,
    *,
    transition_time_s: float = 45.0,
    duration_s: float = 90.0,
    sample_rate_hz: float = 20.0,
    amplitude: float = 1.0,
    second_harmonic_amplitude: float = 0.25,
    noise_std: float = 0.10,
    drift_amplitude: float = 0.20,
    seed: int = 11,
) -> dict[str, Any]:
    """Build a signal with phase-continuous early-to-late respiration rates."""
    times_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    early_frequency_hz = early_bpm / 60.0
    late_frequency_hz = late_bpm / 60.0
    early_time = np.minimum(times_s, transition_time_s)
    late_time = np.maximum(times_s - transition_time_s, 0.0)
    phase = 2.0 * np.pi * ((early_frequency_hz * early_time) + (late_frequency_hz * late_time))
    rng = np.random.default_rng(seed)
    samples = (
        amplitude * np.sin(phase)
        + second_harmonic_amplitude * np.sin(2.0 * phase + 0.4)
        + drift_amplitude * np.sin(2.0 * np.pi * 0.02 * times_s)
        + noise_std * rng.normal(size=times_s.size)
    )
    centered = samples - np.mean(samples)
    std = float(np.std(centered))
    zscore = centered / std if std > 0.0 else centered
    return {
        "times_s": times_s.tolist(),
        "frame_period_s": float(1.0 / sample_rate_hz),
        "raw_signal": samples.tolist(),
        "centered_signal": centered.tolist(),
        "zscore_signal": zscore.tolist(),
        "center_bin": 0,
        "center_range_m": 0.0,
        "selected_bins": [0],
        "selected_ranges_m": [0.0],
    }


def generate_synthetic_multistage_respiration_signal(
    stages: list[tuple[float, float]],
    *,
    sample_rate_hz: float = 20.0,
    amplitude: float = 1.0,
    second_harmonic_amplitude: float = 0.25,
    noise_std: float = 0.10,
    drift_amplitude: float = 0.20,
    seed: int = 13,
) -> dict[str, Any]:
    """Build a phase-continuous signal with multiple rate plateaus.

    Each stage is a (duration_s, bpm) tuple.
    """
    if not stages:
        raise ValueError("stages must contain at least one (duration_s, bpm) tuple")
    duration_s = float(sum(duration for duration, _ in stages))
    times_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    bpm_values = np.zeros_like(times_s, dtype=float)
    start_s = 0.0
    for duration, bpm in stages:
        end_s = start_s + float(duration)
        bpm_values[(times_s >= start_s) & (times_s < end_s)] = float(bpm)
        start_s = end_s
    bpm_values[bpm_values <= 0.0] = float(stages[-1][1])

    dt = 1.0 / sample_rate_hz
    phase = 2.0 * np.pi * np.cumsum((bpm_values / 60.0) * dt)
    rng = np.random.default_rng(seed)
    samples = (
        amplitude * np.sin(phase)
        + second_harmonic_amplitude * np.sin(2.0 * phase + 0.4)
        + drift_amplitude * np.sin(2.0 * np.pi * 0.02 * times_s)
        + noise_std * rng.normal(size=times_s.size)
    )
    centered = samples - np.mean(samples)
    std = float(np.std(centered))
    zscore = centered / std if std > 0.0 else centered
    return {
        "times_s": times_s.tolist(),
        "frame_period_s": float(dt),
        "raw_signal": samples.tolist(),
        "centered_signal": centered.tolist(),
        "zscore_signal": zscore.tolist(),
        "center_bin": 0,
        "center_range_m": 0.0,
        "selected_bins": [0],
        "selected_ranges_m": [0.0],
    }


def run_synthetic_respiration_validation(*, assert_expected: bool = True) -> list[dict[str, Any]]:
    """Exercise stable-rate harmonic edge cases against the global estimator."""

    cases = [
        {
            "name": "9 BPM with stronger 18 BPM harmonic",
            "actual_bpm": 9.0,
            "expected_bpm": 9.0,
            "fundamental_amplitude": 1.0,
            "second_harmonic_amplitude": 1.8,
            "expected_harmonic": True,
        },
        {
            "name": "12 BPM with stronger 24 BPM harmonic",
            "actual_bpm": 12.0,
            "expected_bpm": 12.0,
            "fundamental_amplitude": 1.0,
            "second_harmonic_amplitude": 1.5,
            "expected_harmonic": True,
        },
        {
            "name": "true 18 BPM full cycles",
            "actual_bpm": 18.0,
            "expected_bpm": 18.0,
            "fundamental_amplitude": 1.0,
            "second_harmonic_amplitude": 0.2,
            "expected_harmonic": False,
        },
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        signal = generate_synthetic_respiration_signal(
            case["actual_bpm"],
            fundamental_amplitude=case["fundamental_amplitude"],
            second_harmonic_amplitude=case["second_harmonic_amplitude"],
        )
        estimate = estimate_breathing_rate_multiwindow(signal, min_bpm=4.0, max_bpm=30.0)
        row = {
            "case": case["name"],
            "selected_bpm": estimate["selected_bpm"],
            "raw_spectral_bpm": estimate["raw_spectral_bpm"],
            "time_domain_bpm": estimate["time_domain_bpm"],
            "harmonic_suspected": estimate["harmonic_suspected"],
            "final_confidence": estimate["final_confidence"],
            "decision_reason": estimate["decision_reason"],
        }
        results.append(row)

        if assert_expected:
            if abs(float(estimate["selected_bpm"]) - float(case["expected_bpm"])) > 1.5:
                raise AssertionError(f"{case['name']} selected {estimate['selected_bpm']:.2f} BPM")
            if bool(estimate["harmonic_suspected"]) != bool(case["expected_harmonic"]):
                raise AssertionError(f"{case['name']} harmonic_suspected={estimate['harmonic_suspected']}")

    return results


def run_synthetic_transition_validation(*, assert_expected: bool = True) -> dict[str, Any]:
    """Check that a two-stage synthetic signal is labeled nonstationary."""

    signal = generate_synthetic_transition_respiration_signal(9.0, 18.0)
    features = build_respiration_features(signal, min_bpm=4.0, max_bpm=30.0)
    result = {
        "case": "9 BPM to 18 BPM transition",
        "is_nonstationary": features["is_nonstationary"],
        "transition_detected": features["transition_detected"],
        "early_bpm": features["early_bpm"],
        "late_bpm": features["late_bpm"],
        "bpm_change_magnitude": features["bpm_change_magnitude"],
        "transition_time_s": features["transition_time_s"],
        "global_bpm": features["global_bpm"],
        "global_bpm_representative": features["global_bpm_representative"],
        "global_bpm_confidence": features["global_bpm_confidence"],
        "trend_confidence": features["trend_confidence"],
        "final_confidence": features["final_confidence"],
        "quality_label": features["quality_label"],
        "nonstationary_reason": features["nonstationary_reason"],
    }
    if assert_expected:
        if not result["is_nonstationary"] or not result["transition_detected"]:
            raise AssertionError("Transition signal was not marked nonstationary")
        if result["early_bpm"] is None or abs(float(result["early_bpm"]) - 9.0) > 2.0:
            raise AssertionError(f"early_bpm={result['early_bpm']}")
        if result["late_bpm"] is None or abs(float(result["late_bpm"]) - 18.0) > 2.5:
            raise AssertionError(f"late_bpm={result['late_bpm']}")
        if result["global_bpm_representative"]:
            raise AssertionError("Transition signal marked global BPM representative")
        if float(result["trend_confidence"]) < 0.45:
            raise AssertionError(f"trend_confidence={result['trend_confidence']}")
    return result


def run_synthetic_stability_validation(*, assert_expected: bool = True) -> list[dict[str, Any]]:
    """Verify that clearly stable synthetic recordings stay classified stable."""

    results: list[dict[str, Any]] = []
    for bpm in (7.5, 12.0, 15.0):
        signal = generate_synthetic_respiration_signal(
            bpm,
            duration_s=90.0,
            second_harmonic_amplitude=0.20,
            noise_std=0.10,
        )
        features = build_respiration_features(signal, min_bpm=4.0, max_bpm=30.0)
        result = {
            "case": f"stable {bpm:.1f} BPM",
            "expected_bpm": bpm,
            "recording_type": features["recording_type"],
            "transition_detected": features["transition_detected"],
            "global_bpm": features["global_bpm"],
            "early_bpm": features["early_bpm"],
            "late_bpm": features["late_bpm"],
            "trend_confidence": features["trend_confidence"],
            "final_confidence": features["final_confidence"],
        }
        results.append(result)
        if assert_expected:
            if result["recording_type"] != "stable" or result["transition_detected"]:
                raise AssertionError(f"{result['case']} was marked as {result['recording_type']}")
            if result["global_bpm"] is None or abs(float(result["global_bpm"]) - bpm) > 1.5:
                raise AssertionError(f"{result['case']} global_bpm={result['global_bpm']}")
    return results


if __name__ == "__main__":
    for result in run_synthetic_respiration_validation(assert_expected=True):
        print(
            f"{result['case']}: selected={result['selected_bpm']:.2f} BPM, "
            f"raw={result['raw_spectral_bpm']:.2f} BPM, "
            f"time_domain={result['time_domain_bpm']:.2f} BPM, "
            f"harmonic_suspected={result['harmonic_suspected']}, "
            f"confidence={result['final_confidence']:.2f}"
        )
        print(f"  {result['decision_reason']}")
    transition = run_synthetic_transition_validation(assert_expected=True)
    print(
        f"{transition['case']}: early={transition['early_bpm']:.2f} BPM, "
        f"late={transition['late_bpm']:.2f} BPM, "
        f"transition={transition['transition_detected']}, "
        f"global_representative={transition['global_bpm_representative']}, "
        f"trend_confidence={transition['trend_confidence']:.2f}"
    )
    print(f"  {transition['nonstationary_reason']}")
    for stable in run_synthetic_stability_validation(assert_expected=True):
        print(
            f"{stable['case']}: type={stable['recording_type']}, "
            f"global={stable['global_bpm']:.2f} BPM, transition={stable['transition_detected']}"
        )
