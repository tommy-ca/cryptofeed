Buf Schema Registry (BSR) Workflow

Overview
- Manage protobuf schemas with Buf, run lint/breaking checks, and push tagged versions to BSR.
- Enforce single source of truth via codegen drift checks and lakehouse schema contract tests.
- Consumers pin module versions; no tag reuse; evolve with backward-compatible changes.

Prerequisites
- Buf CLI installed (see Makefile `make install`).
- Access to the BSR module named in `buf.yaml`.
- GitHub secret `BUF_TOKEN` for publish workflow.

Local Developer Workflow (Buf CLI)
- Lint: `buf lint`
- Build: `buf build`
- Generate stubs: `buf generate` (to `gen/*` per `buf.gen.yaml`)
- Breaking check (vs main):
  `git fetch --no-tags --prune origin +refs/heads/main:refs/heads/main && buf breaking --against '.git#branch=main'`
- Format: `buf format -w`
- Push to BSR (manual): `buf push` or `buf push --tag vX.Y.Z`

Local Tests
- Run fast CLI smoke tests: `pytest -q tests/proto_integration/test_buf_cli_workflow.py`

CI Workflows (GitHub Actions)
- Proto CI: lint, generate (fail on drift), breaking check, proto tests.
- Lakehouse contracts CI: runs schema contract tests when proto/lakehouse change.
- BSR publish: on `schema-v*` tags, runs lint, build, breaking, codegen drift gate, then `buf push --tag <version>`.
  - Also supports `workflow_dispatch` with inputs: `version`, `dry_run`, `create`, and `visibility`.
  - Dry-run mode executes all checks and skips push.
  - `create=true` allows creating the module on first publish (default `private` visibility).

BSR Smoke Test
- Workflow: `BSR Smoke Test` (`.github/workflows/bsr-smoke.yml`)
- Runs on `workflow_dispatch` and validates:
  - BUF_TOKEN presence
  - `buf registry whoami buf.build`
  - `buf lint`, `buf build`, and `buf generate`

Tagging Guidelines
- Use semver-like tags per module lifecycle; do not reuse tags.
- Use git tags `schema-vMAJOR.MINOR.PATCH` to trigger publish (e.g., `schema-v1.0.0`).
- Backward-compatible changes only within `cryptofeed.v1`.
- Breaking changes require a new major package (`cryptofeed.v2`).

Release Steps
1) Update protos/specs; run `buf lint` and `buf breaking`.
2) Run `buf generate` and ensure language stubs compile where applicable (no drift).
3) Tag and push:
   - Local BSR push (optional): `make push TAG=v1.0.0`.
   - CI publish (preferred):
     - Option A: create git tag `schema-v1.0.0` and push.
     - Option B: run `Publish Schemas to BSR` workflow manually with inputs `version=v1.0.0`, `dry_run=false`, and (first-time) `create=true`.
4) Update consumers to pin the new version.

Quickstart
- See `docs/BSR_QUICKSTART.md` for step-by-step setup and usage.
