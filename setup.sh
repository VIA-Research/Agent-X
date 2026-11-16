#!/usr/bin/env bash
# Installs Agent-X into the active Python environment.
set -euo pipefail
cd "$(dirname "$0")"

pip install -r mlx-engine/requirements.txt
# Install mlx-lm after the requirements, so that it replaces any mlx-lm pulled
# in as a dependency
pip install -e mlx-lm
pip install -e mlx-engine
# Only for the PromptWeaver offline phase
pip install -r promptweaver/requirements.txt
