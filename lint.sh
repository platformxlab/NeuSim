#!/bin/bash
# Note: set -e is not used globally to allow non-blocking checks to run without exiting immediately

FAILURE=0

echo "Running ruff check (Critical Errors)..."
# Critical errors verify correctness:
# E9: Syntax errors
# F821: Undefined name
# F822: Undefined export in __all__
# F823: Local variable referenced before assignment
ruff check --select E9,F821,F822,F823 .
if [ $? -ne 0 ]; then
  echo "Critical linting errors found!"
  FAILURE=1
fi

echo "Running ruff check (Warnings - Style/Complexity)..."
# Warnings (non-blocking):
# E: pycodestyle errors
# I: isort imports
# B: flake8-bugbear
# UP: pyupgrade
# F: All Pyflakes (including unused imports/variables) that are not in critical
ruff check --select E,I,B,UP,F --exit-zero .

echo "Running ruff format check (Non-blocking)..."
ruff format --check . || echo "Ruff format check failed (warning only)"

echo "Running mypy (Non-blocking)..."
mypy . || echo "Mypy type checking failed (warning only)"

if [ $FAILURE -ne 0 ]; then
  echo "Linting failed due to critical errors."
  exit 1
fi

echo "Linting passed."
exit 0
