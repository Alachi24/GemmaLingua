#!/usr/bin/env bash
set -euo pipefail

echo "=== GemTun Setup ==="

# Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r python/requirements.txt

echo "Creating data directories..."
mkdir -p data/raw data/processed models/adapters models/base logs

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Place your Igbo text file in data/raw/"
echo "  2. Place your En-Ig translation CSV in data/raw/"
echo "  3. Edit config/pipeline.yaml to point to your files"
echo "  4. Set your HF_TOKEN env var"
echo "  5. Run: ./scripts/run_pipeline.sh"
