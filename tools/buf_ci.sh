#!/usr/bin/env bash

set -euo pipefail

if ! command -v buf >/dev/null 2>&1; then
  echo "buf CLI not installed; skipping lint/breaking checks." >&2
  exit 0
fi

echo "Running buf format --diff"
buf format --diff

echo "Running buf lint"
buf lint

if [[ -n "${BUF_BREAKING_BASE:-}" ]]; then
  echo "Running buf breaking against ${BUF_BREAKING_BASE}"
  buf breaking --against "${BUF_BREAKING_BASE}"
else
  echo "BUF_BREAKING_BASE not set; skipping breaking change detection." >&2
fi
