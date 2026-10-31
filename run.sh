#!/usr/bin/env bash
# Quickstart launcher for the Gemini Lab Robot demo.
# Works out of the box on macOS, Linux, and WSL.

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_BIN="python"
if [[ "$*" == *"--viewer"* ]]; then
    if [ -x "$SCRIPT_DIR/.venv/bin/mjpython" ]; then
        PYTHON_BIN="mjpython"
    fi
fi

# 1. Use project-local virtual environment if present
if [ -x "$SCRIPT_DIR/.venv/bin/$PYTHON_BIN" ]; then
    exec "$SCRIPT_DIR/.venv/bin/$PYTHON_BIN" demo.py "$@"
fi

# 2. Use uv if available in PATH or ~/.local/bin
if command -v uv >/dev/null 2>&1; then
    exec uv run "$PYTHON_BIN" demo.py "$@"
elif [ -x "$HOME/.local/bin/uv" ]; then
    exec "$HOME/.local/bin/uv" run "$PYTHON_BIN" demo.py "$@"
fi

# 3. Use active virtualenv python if activated
if [ -n "$VIRTUAL_ENV" ] && command -v python >/dev/null 2>&1; then
    exec python demo.py "$@"
fi

echo "❌ Error: Python virtual environment not found in .venv."
echo ""
echo "To set up the environment, run:"
echo "    uv sync"
echo "or:"
echo "    python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
exit 1
