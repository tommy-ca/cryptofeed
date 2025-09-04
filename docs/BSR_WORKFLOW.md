Buf Schema Registry (BSR) Workflow

Overview
- Manage protobuf schemas with Buf, run lint/breaking checks, and push tagged versions to BSR.
- Consumers pin module versions; no tag reuse; evolve with backward-compatible changes.

Prerequisites
- Buf CLI installed (see Makefile `make install`).
- Access to the BSR module named in `buf.yaml`.

Commands
- Lint: `buf lint`
- Build: `buf build`
- Generate: `buf generate` (outputs to `gen/*` per `buf.gen.yaml`)
- Breaking check (against `main`): `buf breaking --against '.git#branch=main'`
- Push with tag: `buf push --tag v1.0.x`

Tagging Guidelines
- Use semver-like tags per module lifecycle; do not reuse tags.
- Backward-compatible changes only within the same major package (`cryptofeed.v1`).
- Breaking changes require a new major package (`cryptofeed.v2`) and new topics.

Release Steps
1) Update protos/specs; run `buf lint` and `buf breaking`.
2) Run `buf generate` and ensure language stubs compile where applicable.
3) Create a new tag and push: `make push TAG=v1.0.x`.
4) Update producers/consumers to pin the new version as needed.

