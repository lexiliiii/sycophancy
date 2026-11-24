#!/bin/bash
#SBATCH --job-name=syco-eval
#SBATCH --output=logs/syco-%j.out
#SBATCH --error=logs/syco-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Add your school's GPU partition/directives only if you also launch vLLM here.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository directory}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/config.yaml}"

cd "$PROJECT_DIR"
mkdir -p logs
source .venv/bin/activate

# API keys should be exported by the shell, a secrets manager, or your cluster's
# protected environment mechanism before sbatch. Do not put them in this file.
syco-run --config "$CONFIG_PATH"

