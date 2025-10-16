#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: tools/buf_publish.sh <semver>" >&2
  exit 1
fi

VERSION="$1"

if ! command -v buf >/dev/null 2>&1; then
  echo "buf CLI not installed" >&2
  exit 1
fi

echo "Linting Protobuf module..."
buf lint

if [[ -n "${BUF_BREAKING_BASE:-}" ]]; then
  echo "Running breaking change check against $BUF_BREAKING_BASE"
  buf breaking --against "$BUF_BREAKING_BASE"
else
  echo "BUF_BREAKING_BASE not set; skipping breaking check" >&2
fi

echo "Generating language bindings..."
buf generate

MODULE=buf.build/tommyk/crypto-market-data

echo "Pushing version $VERSION to Buf Schema Registry ($MODULE)"
buf push proto --label "$VERSION" --label main

echo "Publish complete."
