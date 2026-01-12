#!/bin/bash
set -e

echo "Running ruff check..."
ruff check .

echo "Running ruff format check..."
ruff format --check .

echo "Running mypy..."
mypy .
