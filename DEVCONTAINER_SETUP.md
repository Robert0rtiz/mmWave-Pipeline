# CARSE mmWave — portable VS Code development container

## What this provides

- Python 3.12, with the **existing repository `requirements.txt`** installed while building the Docker image.
- VS Code Python, Pylance, and Jupyter extensions, with the container's `/opt/venv/bin/python` selected.
- A non-root `vscode` user and a mounted workspace at `/workspace`. Editing source code or adding radar data does **not** require rebuilding the image.
- Optional headless smoke tests of imports, synthetic signal generation, and the latest real capture.

The `requirements.txt` and the existing notebook/Python modules are **not included in this drop-in configuration**. Keep your own copies unchanged.

## Expected repository layout

```text
CARSE-mmWave/
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── .dockerignore
├── requirements.txt
├── CARSE mmWave.ipynb
├── lib/
│   ├── capture_pipeline.py
│   ├── parser_mmw_demo.py
│   ├── respiration_pipeline.py
│   ├── visualization_pipeline.py
│   └── synthetic_respiration_validation.py
├── scripts/
│   └── smoke_test.py
└── ref/
    └── data/
        ├── recording_timestamp.cfg       # placed here after capture
        └── recording_timestamp.dat       # processed OOB UART TLV, NOT raw ADC
```

In the currently available project handoff, modules live in `lib/` and the main notebook is in the repository root. Preserve that layout; `lib` may use Python's namespace-package behavior or its existing `__init__.py`.

## 1. Install and open

1. Install **Docker Desktop**, **VS Code**, and the **Dev Containers** extension (`ms-vscode-remote.remote-containers`) on Windows 11. Ensure Docker Desktop is running and using Linux containers / WSL 2.
2. Copy `.devcontainer/`, `.dockerignore`, and `scripts/smoke_test.py` from this bundle into the **same folder** as your existing `requirements.txt` and notebook.
3. Open **that repository root folder** in VS Code. Open the command palette (`Ctrl+Shift+P`), choose **Dev Containers: Reopen in Container**.
4. Wait for the image build to install dependencies. Subsequent source-data changes should not reinstall them. Changing `requirements.txt` requires **Dev Containers: Rebuild Container**.
5. Open a terminal **inside the container** and run:

   ```bash
   python -m pip check
   python scripts/smoke_test.py
   ```

6. Open `CARSE mmWave.ipynb`. Select the Python kernel corresponding to `/opt/venv/bin/python` if VS Code prompts for a kernel.

## 2. First radar-backed end-to-end test (Windows host captures; container processes)

The current CARSE code uses **processed UART TLV streams** from the IWR6843ISK **Out-of-Box demo**, not the raw ADC files captured by DCA1000EVM/mmWave Studio. The existing report describes the original IWR6843ISK + MMWAVEICBOOST and mmWave Demo Visualizer capture path; match the capture tool to whichever carrier/firmware setup is physically connected. Do not feed DCA1000 `adc_data*.bin` files into this parser.

1. On the **Windows host**, confirm the radar runs the appropriate Out-of-Box demo firmware and that its **CLI/config and data UART** serial interfaces are available in Device Manager. If the demo does not expose the expected ports, resolve that before capturing. Do **not** assume an mmWave Studio raw-ADC firmware image is interchangeable with the Out-of-Box UART demo.
2. Using the TI mmWave Demo Visualizer, select the matching xWR68xx platform and firmware/SDK, then load/send a valid `.cfg`. **Enable Range Profile** (TLV 2) via the visualizer/`guiMonitor` settings. Detected points and statistics are useful optional outputs.
3. Record a processed UART stream as a `.dat` and save the matching `.cfg`. For an initial system test, use about **60 seconds** of controlled, stationary breathing; a shorter recording can test parsing, but low respiration rates need longer windows.
4. Copy the `.dat` and matching `.cfg` into your project's `ref/data/` directory. The container instantly sees them through the workspace bind mount—no `docker cp`, reinstall, or rebuild required. The pipeline matches recordings to configuration files by timestamps, so keep pairs consistent.
5. In the container terminal, run:

   ```bash
   python scripts/smoke_test.py --require-capture
   ```

   A pass means a `.cfg`/`.dat` pair was found, frames were parsed, and range-profile TLV 2 is available. It does **not** establish the accuracy of respiration estimates.
6. Open `CARSE mmWave.ipynb` and **run all cells in order**. Verify the capture summary, range-time waterfall, selected range bins, extracted breathing waveform, spectrogram, BPM estimate, and any quality warnings.

### Practical interpretation

- **Imports fail**: check repo layout (`lib/`) and whether `requirements.txt` from the current repo was used to rebuild.
- **No capture found**: check `ref/data/` and make sure both `.cfg` and `.dat` are present.
- **TLV 2 missing**: enable Range Profile before recording a new `.dat`.
- **No Windows COM port inside container**: expected in this setup. Windows Docker Desktop does not automatically pass host COM/USB devices to Linux containers. Keep acquisition in a native Windows tool for this first test; adding real-time container UART access is a separate integration task.
- **Raw ADC `.bin` from DCA1000**: incompatible with the current `parser_mmw_demo.py` UART parser; requires a separate acquisition/processing pipeline.

## 3. Optional CLI build without VS Code

From the project root (the directory containing `requirements.txt`):

```bash
docker build -f .devcontainer/Dockerfile -t carse-mmwave:dev .
```

With VS Code closed you can also run the container with a local bind mount. In PowerShell:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/workspace" -w /workspace carse-mmwave:dev python scripts/smoke_test.py
```

For reproducibility across other team members, share the repository (including the pinned `requirements.txt` and this configuration), **not** the built image. Keep human-subject recordings out of source control unless authorized and de-identified.
