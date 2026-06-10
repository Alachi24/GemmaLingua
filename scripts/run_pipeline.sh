#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    echo "ERROR: No Python virtual environment found. Run scripts/setup.sh first."
    exit 1
fi

source venv/bin/activate

echo "Building Java project..."
./mvnw clean package -q

echo ""
echo "Running GemTun pipeline..."
java -jar target/gemTun-1.0.0-jar-with-dependencies.jar "$@"
