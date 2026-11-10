"""Helpers for locating, loading, and summarizing capture files.

This module is the notebook-facing entrypoint for the raw parsed data. It is
responsible for pairing `.dat` captures with nearby `.cfg` files, building the
`CaptureBundle` container used by the rest of the pipeline, and exporting small
human-readable previews that are convenient in notebook cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from .parser_mmw_demo import (
    TLV_TYPE_AZIMUT_STATIC_HEAT_MAP,
    TLV_TYPE_DETECTED_POINTS,
    TLV_TYPE_DETECTED_POINTS_SIDE_INFO,
    TLV_TYPE_NOISE_PROFILE,
    TLV_TYPE_RANGE_DOPPLER_HEAT_MAP,
    TLV_TYPE_RANGE_PROFILE,
    TLV_TYPE_STATS,
    TLV_TYPE_TEMPERATURE_STATS,
    detections_to_rows,
    parse_capture_file,
    parse_config_file,
    summarize_capture,
    write_detection_csv,
)


TLV_NAMES = {
    TLV_TYPE_DETECTED_POINTS: "detected_points",
    TLV_TYPE_RANGE_PROFILE: "range_profile",
    TLV_TYPE_NOISE_PROFILE: "noise_profile",
    TLV_TYPE_AZIMUT_STATIC_HEAT_MAP: "azimuth_static_heatmap",
    TLV_TYPE_RANGE_DOPPLER_HEAT_MAP: "range_doppler_heatmap",
    TLV_TYPE_STATS: "stats",
    TLV_TYPE_DETECTED_POINTS_SIDE_INFO: "detected_points_side_info",
    TLV_TYPE_TEMPERATURE_STATS: "temperature_stats",
}


@dataclass(slots=True)
class CaptureBundle:
    """All parsed artifacts needed to analyze one capture pair."""

    root: Path
    data_dir: Path
    config_path: Path
    data_path: Path
    config: dict[str, Any]
    frames: list[Any]
    summary: dict[str, Any]


def resolve_data_dir(root: str | Path | None = None) -> Path:
    """Return the default `ref/data` directory for a repo root or current cwd."""

    base = Path.cwd() if root is None else Path(root)
    return base / "ref" / "data"


_CAPTURE_TIMESTAMP_RE = re.compile(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}_\d{3,6})$")


def _capture_timestamp(path: Path) -> datetime:
    """Extract the timestamp embedded in a capture filename when available."""

    match = _CAPTURE_TIMESTAMP_RE.search(path.stem)
    if match is not None:
        return datetime.strptime(match.group(1), "%Y_%m_%dT%H_%M_%S_%f")
    return datetime.fromtimestamp(path.stat().st_mtime)


def list_capture_paths(root: str | Path | None = None) -> list[tuple[Path, Path]]:
    """Pair each `.dat` file with the nearest `.cfg` file by timestamp.

    The pairing intentionally advances through configs only once. That keeps the
    order stable for batch comparisons and avoids reusing one config for every
    later capture in the folder.
    """

    data_dir = resolve_data_dir(root)
    cfg_files = sorted(data_dir.glob("*.cfg"), key=_capture_timestamp)
    dat_files = sorted(data_dir.glob("*.dat"), key=_capture_timestamp)
    if not cfg_files:
        raise FileNotFoundError(f"No .cfg files found in {data_dir}")
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {data_dir}")

    pairs: list[tuple[Path, Path]] = []
    next_cfg_index = 0
    for data_file in dat_files:
        if next_cfg_index >= len(cfg_files):
            break

        data_timestamp = _capture_timestamp(data_file)
        best_cfg_index = next_cfg_index
        best_delta_s = abs((_capture_timestamp(cfg_files[best_cfg_index]) - data_timestamp).total_seconds())

        for candidate_index in range(next_cfg_index + 1, len(cfg_files)):
            candidate_timestamp = _capture_timestamp(cfg_files[candidate_index])
            candidate_delta_s = abs((candidate_timestamp - data_timestamp).total_seconds())
            if candidate_delta_s <= best_delta_s:
                best_cfg_index = candidate_index
                best_delta_s = candidate_delta_s
                continue
            if candidate_timestamp >= data_timestamp:
                break

        pairs.append((cfg_files[best_cfg_index], data_file))
        next_cfg_index = best_cfg_index + 1

    if not pairs:
        raise FileNotFoundError(f"No .cfg/.dat capture pairs found in {data_dir}")
    return pairs


def recent_capture_paths(limit: int = 3, root: str | Path | None = None) -> list[tuple[Path, Path]]:
    """Return the newest `limit` capture pairs discovered in `ref/data`."""

    if limit < 1:
        raise ValueError("limit must be at least 1")
    return list_capture_paths(root)[-limit:]


def latest_capture_paths(root: str | Path | None = None) -> tuple[Path, Path]:
    """Return the single newest capture pair in the workspace."""

    return list_capture_paths(root)[-1]


def load_capture_bundle(
    config_path: str | Path,
    data_path: str | Path,
    *,
    root: str | Path | None = None,
) -> CaptureBundle:
    """Parse one explicit `.cfg`/`.dat` pair into a `CaptureBundle`."""

    config_file = Path(config_path)
    data_file = Path(data_path)
    base = Path.cwd() if root is None else Path(root)
    frames = parse_capture_file(data_file)
    return CaptureBundle(
        root=base,
        data_dir=data_file.parent,
        config_path=config_file,
        data_path=data_file,
        config=parse_config_file(config_file),
        frames=frames,
        summary=summarize_capture(frames),
    )


def load_latest_capture(root: str | Path | None = None) -> CaptureBundle:
    """Load the newest capture pair under the default data directory."""

    config_path, data_path = latest_capture_paths(root)
    return load_capture_bundle(config_path, data_path, root=root)


def load_recent_captures(limit: int = 3, root: str | Path | None = None) -> list[CaptureBundle]:
    """Load the newest `limit` capture pairs as parsed bundles."""

    return [
        load_capture_bundle(config_path, data_path, root=root)
        for config_path, data_path in recent_capture_paths(limit=limit, root=root)
    ]


def build_capture_overview(bundle: CaptureBundle, num_commands: int = 8) -> dict[str, Any]:
    """Build the small metadata summary shown in notebook step 1."""

    first_frame = bundle.frames[0] if bundle.frames else None
    return {
        "config_file": bundle.config_path.name,
        "data_file": bundle.data_path.name,
        "summary": bundle.summary,
        "config_metadata": bundle.config["metadata"],
        "first_commands": bundle.config["commands"][:num_commands],
        "first_frame_header": asdict(first_frame.header) if first_frame else None,
        "first_frame_tlvs": [
            {"name": TLV_NAMES.get(tlv_type, f"unknown_{tlv_type}"), "length": tlv_len}
            for tlv_type, tlv_len in (first_frame.tlvs if first_frame else [])
        ],
    }


def export_capture_readables(
    bundle: CaptureBundle,
    *,
    num_rows: int = 5,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a detections CSV and return small previews for notebook display."""

    rows = detections_to_rows(bundle.frames)
    output_csv = (
        Path(csv_path)
        if csv_path is not None
        else bundle.data_path.with_name(f"{bundle.data_path.stem}_detections.csv")
    )
    write_detection_csv(bundle.frames, output_csv)
    first_frame = bundle.frames[0] if bundle.frames else None
    return {
        "output_csv": str(output_csv),
        "num_rows": len(rows),
        "first_detection_rows": rows[:num_rows],
        "points_per_frame_head": [len(frame.points) for frame in bundle.frames[:20]],
        "first_range_profile_bins": first_frame.range_profile[:10] if first_frame and first_frame.range_profile else [],
        "first_frame_stats": asdict(first_frame.stats) if first_frame and first_frame.stats is not None else None,
    }