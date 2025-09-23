#!/bin/bash

# fix.sh - Format and fix code with ruff

set -e

echo "🔧 Running ruff format..."
uv run ruff format .

echo "🔍 Running ruff check with auto-fix..."
uv run ruff check --fix .

echo "✅ Code formatting and fixes complete!"