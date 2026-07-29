#!/bin/bash
# Setup third-party dependencies (verl + evalchemy)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up third-party dependencies..."

# verl — GRPO training framework
if [ ! -d "verl/.git" ] && [ ! -f "verl/setup.py" ]; then
    echo "Cloning verl..."
    git clone --depth 1 https://github.com/volcengine/verl.git verl
else
    echo "verl already present, skipping."
fi

# evalchemy — evaluation framework (LiveCodeBench, etc.)
if [ ! -d "evalchemy/.git" ] && [ ! -f "evalchemy/pyproject.toml" ]; then
    echo "Cloning evalchemy..."
    git clone --depth 1 https://github.com/EvalAlchemy/evalchemy.git evalchemy
else
    echo "evalchemy already present, skipping."
fi

# Install verl
echo ""
echo "Installing verl..."
cd verl
pip install -e . -q
cd ..

# Install evalchemy
echo "Installing evalchemy..."
cd evalchemy
pip install -e . -q
cd ..

echo ""
echo "Done! Third-party dependencies ready."
