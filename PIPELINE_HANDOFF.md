# CARSE mmWave Pipeline Quick Handoff

## Read This First

This repository is an exploratory post-processing pipeline for TI IWR6843ISK + MMWAVEICBOOST Out-of-Box demo captures.

The pipeline processes recorded `.dat` UART TLV output files paired with nearby `.cfg` radar configuration files.

It does **not** process raw ADC data.

The parser and capture-loading portions are the most stable parts of the repository. The respiration extraction and BPM estimation logic are experimental and heuristic-based. Treat the reported BPM, quality score, and trend classification as research outputs that require visual inspection and validation, not as clinically reliable measurements.

## Quick Start

1. Create and activate a Python virtual environment.

   Windows PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Linux/macOS:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Place capture files in:

   ```text
   ref/data/
   ```

4. Each recording should include:

   - one `.dat` file
   - one nearby `.cfg` file

5. Open:

   ```text
   CARSE mmWave.ipynb
   ```

6. Run the notebook from top to bottom.

Expected outputs:

- parsed capture summary
- detected-points CSV
- range-time waterfall
- extracted respiration signal
- respiration spectrogram
- BPM / quality report
- latest-three capture comparison

## Trust Boundaries

### Relatively stable

These parts are mainly mechanical parsing, loading, or plotting:

- UART packet parsing
- `.cfg` / `.dat` capture pairing
- frame summaries
- TLV inventory reporting
- detected-points CSV export
- range-profile waterfall plotting when TLV 2 is present

### Experimental

These parts are research heuristics and should not be trusted blindly:

- automatic respiration range-bin selection
- respiration-band scoring
- dynamic range tracking
- harmonic correction
- transition detection
- final BPM selection
- confidence / quality scoring

Do **not** treat a BPM estimate as valid unless the range-time waterfall, extracted signal, spectrogram, and quality flags all make sense together.

## Required TLVs

Respiration extraction usually requires:

- TLV 2: `range_profile`

Useful TLVs that may also appear:

- TLV 1: `detected_points`
- TLV 6: `stats`
- TLV 7: `detected_points_side_info`
- TLV 9: `temperature_stats`

If TLV 2 is missing, the capture can still be parsed and summarized, but respiration extraction will fail.

## Main Files

| File | Purpose |
| --- | --- |
| `CARSE mmWave.ipynb` | Main user-facing workflow |
| `requirements.txt` | Python dependencies required to run the notebook and pipeline |
| `PIPELINE_HANDOFF.md` | Full detailed handoff document |
| `PIPELINE_DIAGRAM.md` | High-level pipeline diagram, if present |
| `lib/parser_mmw_demo.py` | Parses TI UART TLV packets and `.cfg` files |
| `lib/capture_pipeline.py` | Finds `.cfg` / `.dat` pairs and loads capture bundles |
| `lib/respiration_pipeline.py` | Extracts respiration signal and estimates BPM |
| `lib/visualization_pipeline.py` | Plots waterfall, respiration trace, spectrogram, and dashboard |
| `lib/synthetic_respiration_validation.py` | Generates synthetic respiration signals and validates estimator behavior |
| `lib/mmw_demo_example_script.py` | Minimal command-line parser smoke test |
| `tests/test_synthetic_respiration.py` | Unit tests for synthetic respiration behavior, if present |

## Repository Flow

The notebook is intentionally thin. Most of the actual behavior lives in the Python modules.

The usual flow is:

1. Load the newest `.cfg` / `.dat` capture pair.
2. Parse the UART TLV stream into frame packets.
3. Build a capture summary and TLV inventory.
4. Export readable detection artifacts.
5. Build a range-profile matrix from TLV 2.
6. Select a respiration range bin or tracked range path.
7. Extract a one-dimensional respiration-like signal.
8. Plot the range-time waterfall, signal trace, and spectrogram.
9. Estimate BPM and report quality / confidence warnings.
10. Compare recent captures.

## Respiration Pipeline In Plain English

The respiration pipeline tries to do the following:

1. Build a matrix where each row is a radar frame and each column is a range bin.
2. Use the range profile TLV to look for bins whose signal changes over time.
3. Prefer bins with energy inside the respiration band.
4. Select a center range bin or small group of neighboring bins.
5. Average or track those bins over time to make a one-dimensional signal.
6. Center and z-score the signal.
7. Estimate BPM using spectral and time-domain methods.
8. Check whether the signal looks stable, transitioning, ambiguous, or motion-contaminated.
9. Produce a report with the selected BPM, quality flags, and diagnostic notes.

This is exploratory. The selected bin and BPM should be verified visually.

## Where To Change Things

| Goal | Start here |
| --- | --- |
| Add or modify TLV parsing | `lib/parser_mmw_demo.py` |
| Change config parsing | `lib/parser_mmw_demo.py` |
| Change capture discovery or pairing | `lib/capture_pipeline.py` |
| Change capture summary/export behavior | `lib/capture_pipeline.py` |
| Change range-bin scoring | `score_respiration_range_bins()` |
| Change range-bin selection | `select_respiration_center_bin()` |
| Change motion detection threshold | `estimate_range_motion_from_profiles()` |
| Change dynamic range tracking | `_track_residual_range_path()` / `_build_dynamic_range_tracked_signal()` |
| Change respiration extraction | `extract_respiration_signal()` |
| Change BPM estimation | `estimate_breathing_rate_multiwindow()` |
| Change trend detection | `estimate_respiration_trend()` |
| Change quality labels | `assess_respiration_quality()` |
| Change final report text | `format_respiration_feature_report()` |
| Change plots | `lib/visualization_pipeline.py` |
| Add synthetic validation cases | `lib/synthetic_respiration_validation.py` |

## Common Failure Modes

### No respiration signal

Likely cause: TLV 2, `range_profile`, is missing.

Check:

- the `.cfg` file
- the `guiMonitor` command
- the TLV inventory printed by the notebook
- `describe_capture_visualizations(bundle)`

Some captures may only contain TLVs such as stats and temperature. Those files can be parsed, but they cannot be used for respiration extraction.

### Waterfall exists but BPM looks wrong

Possible causes:

- wrong range bin selected
- subject moved forward or backward during the recording
- breathing signal is weak compared to clutter
- harmonic got selected instead of the fundamental
- recording is too short for slow breathing
- frame rate or config metadata parsed incorrectly

Check:

- selected range bin
- selected range in meters
- extracted signal trace
- respiration spectrogram
- quality warnings
- harmonic warnings
- motion flags

### Spectrogram looks useful but final BPM seems questionable

Check:

- `quality_flags`
- `global_bpm_representative`
- `range_motion_detected`
- `dynamic_range_tracking_used`
- `bpm_spread`
- time-domain estimate
- autocorrelation estimate
- global FFT peak

The final BPM logic is heuristic. If the spectrogram and selected BPM disagree, do not assume the selected BPM is correct.

### Latest capture is wrong

The loader pairs files by timestamp. If files were manually copied into `ref/data`, confirm that the `.cfg` and `.dat` files actually belong together.

### Notebook cells behave differently when run out of order

Some notebook cells depend on variables defined in earlier cells, such as BPM limits. Run the notebook from top to bottom unless you know exactly which state variables are already defined.

### Range-Doppler or azimuth heatmap plots are unavailable

Those plots require the corresponding TLVs to be enabled in the radar configuration. If the capture does not contain those TLVs, the pipeline cannot plot true range-Doppler or azimuth heatmap data.


## Recommended Manual Validation

For every new real capture, inspect these in order:

1. Capture summary  
   Confirm frames were parsed and the expected TLVs are present.

2. Range-time waterfall  
   Confirm the subject appears at a reasonable range and that the selected region makes sense.

3. Extracted respiration signal  
   Confirm the signal has visible breathing-like periodicity.

4. Spectrogram  
   Confirm energy is concentrated in a plausible breathing range.

5. BPM report  
   Confirm the final BPM agrees with the visual evidence.

6. Quality warnings  
   Treat warnings seriously. They usually indicate motion, ambiguity, short duration, weak signal, or harmonic confusion.

## Known Cleanup Needed

The repo works as an exploratory notebook pipeline, but it should be cleaned up if development continues.

Recommended cleanup:

- Add a top-level command-line runner for the full capture → signal → features flow.
- Split `respiration_pipeline.py` into smaller modules.
- Move notebook constants into a shared config file.
- Save one or more known-good example captures with expected outputs.
- Validate BPM estimates against a reference respiration sensor.
- Add more tests using real captured data, not only synthetic signals.
- Separate stable parsing code from experimental physiological-estimation code.
- Add clearer config documentation for which `guiMonitor` settings are required for each plot.

## Suggested Future Structure

If this pipeline keeps growing, consider splitting `respiration_pipeline.py` into smaller files:

```text
lib/
  parser_mmw_demo.py
  capture_pipeline.py
  respiration/
    matrix.py
    range_selection.py
    motion_tracking.py
    extraction.py
    bpm_estimation.py
    quality.py
  visualization_pipeline.py
  synthetic_respiration_validation.py
```

This would make the code easier to maintain because the current respiration module mixes signal extraction, range tracking, BPM estimation, trend analysis, and quality scoring in one place.

## Final Notes For The Next Person

The most important thing to understand is that this pipeline is not a finished medical monitoring system. It is a research tool for exploring whether useful respiratory information can be extracted from processed mmWave radar output.

The parser and capture-loading logic are the safest parts to build on.

The respiration/BPM logic is useful, but it is still experimental. Always validate it with plots and, when possible, with a reference respiration sensor.
