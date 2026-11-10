#!/usr/bin/env bash

# CARSE mmWave DevContainer
# Prevent Windows/Linux file permission differences
# from appearing as modifications in Git.

set -e

# Move to the repository root
cd /workspace

# Ignore executable-bit differences between Windows and Linux
git config --local core.filemode false

echo "Git configuration applied successfully."
echo "File permission differences will be ignored."