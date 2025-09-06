# Buf Schema Registry (BSR) Quickstart

This guide walks you through publishing the protobuf schemas in this repo to the Buf Schema Registry (BSR), following Buf’s quickstart and adapted to this project’s setup.

## Prerequisites
- Buf CLI installed (CI uses `bufbuild/buf-setup-action`).
- BSR account and organization (e.g., `tommyk`).
- Repo already has:
  - `buf.yaml` with module name: `buf.build/tommyk/cryptofeed-schemas`.
  - `buf.gen.yaml` configured for Python/Go/TS/Rust codegen.
  - Protos in `proto/`.

## 1) Authenticate to BSR
Interactive (local):
```
buf registry login
```
Non-interactive (CI): set secret `BUF_TOKEN` and export as env `BUF_TOKEN`.

## 2) Prepare the module
- Ensure `buf.yaml` has the module name:
```
version: v2
modules:
  - path: proto
    name: buf.build/tommyk/cryptofeed-schemas
```
- Lint and build locally:
```
buf lint
buf build
```

## 3) Push to BSR
- First push (creates the repository on BSR if you have permissions):
```
buf push
```
- Tag a version (semver-like) and push the tag:
```
buf push --tag v1.0.0
```

Recommendation: Use a dedicated tag namespace for schema releases (e.g., `schema-v1.0.0`).

## 4) Breaking-Change Checks
Run breaking checks locally (compares current branch against `main`):
```
git fetch --no-tags --prune origin +refs/heads/main:refs/heads/main
buf breaking --against '.git#branch=main'
```
CI already runs breaking checks in `.github/workflows/proto-ci.yml`.

## 5) Code Generation
Generate code for local development:
```
buf generate
```
The repo enforces a “no drift” policy: CI fails if `buf generate` produces uncommitted changes.

## 6) CI: Publish on Tag
Create a GH secret `BUF_TOKEN`. The workflow `.github/workflows/bsr-publish.yml` (to be added) will:
- Run `buf lint`, `buf build`, `buf breaking`.
- Run `buf generate` and fail if codegen drifts.
- Push schemas to BSR with the tag that triggered the workflow.

## 7) Consumers
Other repos can depend on this module:
```
buf dep add buf.build/tommyk/cryptofeed-schemas
buf mod update
```
Use the generated languages according to `buf.gen.yaml` (Python/Go/TS/Rust). The Go `go_package_prefix` is managed via `buf.gen.yaml` overrides.

## Versioning & Policy
- Semver-like tags per module lifecycle (don’t reuse tags).
- Backward-compatible changes only within `cryptofeed.v1`.
- Breaking changes require a new major package (e.g., `cryptofeed.v2`).

## Troubleshooting
- 401 on push: ensure you’re logged in or `BUF_TOKEN` is set.
- Lint/breaking failures: follow CLI output to resolve issues before pushing.
- Drift failures: run `buf generate` locally and commit changes.

