#!/usr/bin/env python3
"""Minimal CARSE environment and capture checks; does not control the radar.

Run in the devcontainer:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --require-capture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_environment() -> None:
    import numpy as np
    import matplotlib
    import IPython
    import ipykernel
    from lib import capture_pipeline, parser_mmw_demo, respiration_pipeline, visualization_pipeline
    from lib.synthetic_respiration_validation import generate_synthetic_respiration_signal

    print(f"Python      : {sys.version.split()[0]}")
    print(f"NumPy       : {np.__version__}")
    print(f"Matplotlib  : {matplotlib.__version__}")
    print(f"IPython     : {IPython.__version__}")
    print(f"ipykernel   : {ipykernel.__version__}")
    print("Pipeline    : capture, parser, respiration, visualization imports OK")

    signal = generate_synthetic_respiration_signal(12.0, duration_s=60.0)
    count = len(signal["times_s"])
    assert count > 0 and len(signal["zscore_signal"]) == count
    print(f"Synthetic   : generated {count} samples for a 12 BPM signal")
    assert capture_pipeline.resolve_data_dir(ROOT) == ROOT / "ref" / "data"
    assert parser_mmw_demo is not None and respiration_pipeline is not None and visualization_pipeline is not None


def check_capture(require_capture: bool) -> None:
    from lib.capture_pipeline import latest_capture_paths, load_latest_capture
    from lib.visualization_pipeline import describe_capture_visualizations

    try:
        cfg, dat = latest_capture_paths(root=ROOT)
    except FileNotFoundError as exc:
        if require_capture:
            raise RuntimeError(f"Capture required: {exc}") from exc
        print("Capture     : SKIPPED (no paired .cfg/.dat files in ref/data)")
        return

    print(f"Config      : {cfg.name}")
    print(f"Data        : {dat.name}")
    bundle = load_latest_capture(root=ROOT)
    print(f"Parsed      : {len(bundle.frames)} frame(s)")
    visuals = describe_capture_visualizations(bundle)
    print(f"TLVs        : {visuals['tlv_names_present']}")

    if not bundle.frames:
        raise RuntimeError("The recording parsed as zero frames")
    if not visuals["can_plot_respiration_signal"]:
        raise RuntimeError("Range-profile TLV 2 is absent; enable Range Profile in the radar config")
    print("Capture     : PASS (valid frames and range-profile TLV 2 found)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-capture", action="store_true", help="Fail if no usable radar capture is present")
    args = parser.parse_args()
    try:
        check_environment()
        check_capture(args.require_capture)
    except Exception as exc:
        print(f"FAIL        : {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Smoke test  : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
