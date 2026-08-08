#!/usr/bin/env bash
# Removes local build artifacts and caches before zipping the project for
# submission. Never touches dataset/output.csv (the deliverable), .env, or
# anything under version control that isn't a generated artifact.
#
# Usage:
#   bash scripts/clean.sh
#   bash scripts/clean.sh --deep   # also remove the .venv virtualenv

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Cleaning Python build artifacts..."
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
find . -type f -name "*.pyo" -not -path "./.venv/*" -delete
rm -rf .pytest_cache .mypy_cache .ruff_cache

echo "Cleaning local media-analysis cache (code/cache/)..."
rm -rf code/cache  # re-created automatically on the next audio-analysis call

if [[ "${1:-}" == "--deep" ]]; then
  echo "Removing .venv (re-create with: python -m venv .venv)..."
  rm -rf .venv
fi

echo "Done. Reminder: dataset/output.csv and .env are intentionally left untouched."
