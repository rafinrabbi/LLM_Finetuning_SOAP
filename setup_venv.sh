#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup_venv.sh
# Creates a virtual environment in .venv and installs packages from requirements.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv already exists at $VENV_DIR"
else
  python3 -m venv "$VENV_DIR"
  echo "Created virtualenv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r "$ROOT_DIR/requirements.txt"

echo "Installation complete. To activate the venv run:"
echo "  source $VENV_DIR/bin/activate"
echo "Or, add the venv as a Jupyter kernel with:" 
echo "  python -m ipykernel install --user --name=bart-finetune --display-name 'Python (bart-finetune)'"
