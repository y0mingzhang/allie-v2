#!/bin/bash

# fix.sh - Format and fix code with ruff

set -e

echo "🔧 Running ruff format..."
uv run ruff format src/

echo "🔍 Running ruff check with auto-fix..."
uv run ruff check --fix src/

echo "✅ Code formatting and fixes complete!"