# ****************************************************************************
# * (C) Copyright 2020, Texas Instruments Incorporated. - www.ti.com
# ****************************************************************************
# *
# *  Redistribution and use in source and binary forms, with or without
# *  modification, are permitted provided that the following conditions are
# *  met:
# *
# *    Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *
# *    Redistributions in binary form must reproduce the above copyright
# *    notice, this list of conditions and the following disclaimer in the
# *     documentation and/or other materials provided with the distribution.
# *
# *    Neither the name of Texas Instruments Incorporated nor the names of its
# *    contributors may be used to endorse or promote products derived from
# *    this software without specific prior written permission.
# *
# *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# *  PARTICULAR TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# *  A PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  OWNER OR
# *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# *  EXEMPLARY, ORCONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# *  LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY, OR TORT (INCLUDING
# *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
# ****************************************************************************


# ****************************************************************************
# Sample mmW demo UART output parser script - should be invoked using python3
#       ex: python3 mmw_demo_example_script.py <recorded_dat_file_from_Visualizer>.dat
#
# Notes:
#   1. The parser_mmw_demo script will output the text version 
#      of the captured files on stdio. User can redirect that output to a log file, if desired
#   2. This example script also outputs the detected point cloud data in mmw_demo_output.csv 
#      to showcase how to use the output of parser_one_mmw_demo_output_packet
# ****************************************************************************

"""Minimal CLI example for loading a capture and exporting detection rows.

This script is intentionally much smaller than the notebook flow. It is useful
as a parser smoke test or quick command-line summary, but it does not run the
full respiration pipeline.
"""

import sys
from pathlib import Path

try:
    from .capture_pipeline import export_capture_readables, load_capture_bundle, load_latest_capture
except ImportError:
    from capture_pipeline import export_capture_readables, load_capture_bundle, load_latest_capture


def main(argv: list[str] | None = None) -> int:
    """Load an explicit or latest capture pair and print a compact summary."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2:
        bundle = load_capture_bundle(
            Path(args[0]).expanduser().resolve(),
            Path(args[1]).expanduser().resolve(),
        )
    elif len(args) == 1:
        data_path = Path(args[0]).expanduser().resolve()
        cfg_candidates = sorted(data_path.parent.glob("*.cfg"), key=lambda path: path.stat().st_mtime)
        if not cfg_candidates:
            print(f"Error: no matching .cfg file found next to {data_path}")
            return 1
        bundle = load_capture_bundle(cfg_candidates[-1], data_path)
    else:
        try:
            bundle = load_latest_capture(Path(__file__).resolve().parents[1])
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

    readable = export_capture_readables(bundle)
    summary = bundle.summary

    print(f"config: {bundle.config_path}")
    print(f"capture: {bundle.data_path}")
    print(f"frames parsed: {summary['frame_count']}")
    print(f"frames with detections: {summary['non_empty_frame_count']}")
    print(f"total detected points: {summary['total_detected_points']}")
    print(f"max points in frame: {summary['max_points_in_frame']}")
    print(f"wrote: {readable['output_csv']}")

    if bundle.frames:
        first_frame = bundle.frames[0]
        print(
            f"first frame -> frame_number={first_frame.header.frame_number}, "
            f"subframe={first_frame.header.subframe_number}, "
            f"tlvs={first_frame.header.num_tlvs}, points={len(first_frame.points)}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
