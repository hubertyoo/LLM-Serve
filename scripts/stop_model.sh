#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name> [--grace-period seconds] [--json]"
  exit 1
fi

python3 -m llm_serve.launcher stop "$@"
