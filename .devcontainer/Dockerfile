# CARSE mmWave sensing pipeline
# This image installs dependencies only. The source tree and radar recordings
# are bind-mounted by VS Code, rather than copied into the image.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/workspace

# git is useful for normal development; these libraries also keep commonly
# used numerical/plotting wheels compatible with headless Linux containers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash ca-certificates git libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash vscode \
    && python -m venv /opt/venv \
    && chown -R vscode:vscode /opt/venv \
    && mkdir -p /workspace \
    && chown vscode:vscode /workspace

# Build context is repository root (see devcontainer.json).
# Keep this layer independent of source/data so code and captures can change
# without reinstalling all pinned scientific Python dependencies.
COPY requirements.txt /tmp/carse-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/carse-requirements.txt \
    && python -m pip check

USER vscode
WORKDIR /workspace
CMD ["sleep", "infinity"]
