# Project Structure

## Top-Level Directories
- `cryptofeed/` – Core library modules (feeds, exchanges, backends, proxy
  system, utilities, and typed definitions).
- `docs/` – User and developer documentation, including exchange guides,
  specs (`docs/specs/`), and proxy user guides (`docs/proxy/`).
- `examples/` – Runnable scripts demonstrating live data ingestion,
  NBBO aggregation, and backend integrations.
- `tests/` – Unit, integration, and fixture suites following the "no mocks"
  principle with real transports or patched CCXT clients.
- `config.yaml` / `config_example.yaml` – Sample configuration surfaces for
  feed and proxy settings.
- `.kiro/` – Spec and steering workspace for agentic workflows.

## `cryptofeed/` Package Layout
- `feedhandler.py` – Entry point for orchestrating connections and callbacks.
- `connection/` – Async connection primitives (WebSocket clients, throttling).
- `exchanges/`
  - `__init__.py` – Registry of exchange feed classes.
  - `ccxt/` – Generic CCXT abstraction with submodules:
    - `config.py` / `context.py` – Pydantic models and runtime contexts.
    - `extensions.py` – Hook registration utilities.
    - `transport/` – Proxy-aware REST & WebSocket transports with retry logic.
    - `adapters/` – Trade/order book normalisation utilities and registry.
    - `feed.py` / `builder.py` – CCXT feed bridge into the main Feed hierarchy.
  - `<exchange>.py` – Native implementations for venues with bespoke APIs.
- `proxy.py` – Central proxy injector, pools, and shared logging utilities.
- `backends/` – Output connectors (Redis, Arctic, sockets, file writers).
- `defines.py` / `types.py` – Constants and typed structures shared across the
  codebase.
- `util/` – Helper functions (symbol normalisation, throttling, misc tools).

## Testing Layout
- `tests/unit/` – Focused tests for config validation, transports, adapters,
  proxy selection, etc. Each spec milestone (e.g., CCXT transport) adds
  dedicated coverage.
- `tests/integration/` – Exchange-level scenarios using CCXT or native clients
  with proxy injection and authentication hooks.
- `tests/fixtures/` – Recorded payloads or helper classes for deterministic
  integration flows.

## Conventions & Patterns
- **Async-First:** All network IO uses asyncio; synchronous helpers wrap async
  contexts when required.
- **Spec Alignment:** New features land under active specs (`.kiro/specs/`)
  with TDD-driven tasks and documentation updates.
- **Proxy-First:** Any new transport or exchange module integrates with
  `get_proxy_injector()` to honour per-exchange routing.
- **Typed Configuration:** Pydantic v2 models enforce validation at load time;
  extension hooks allow per-exchange custom fields without touching core
  modules.
- **No Legacy:** Deprecated modules are removed rather than maintained; use
  compatibility shims only as temporary bridges.

## Spec-Driven Structure & Compound Workstreams

- **Spec Layout:** Each spec under `.kiro/specs/` (e.g., normalized schemas,
  protobuf serialization, Kafka producer, E2E flows) contains requirements,
  design, and tasks. These specs map onto distinct workstreams that compose
  via clearly defined contracts (schemas, APIs, topics, headers, tools).
- **Cross-Stream Changes:** When a change crosses streams (for example, adding
  a new schema field that affects serialization and Kafka topics), the
  owning spec for each stream must be updated and coordinated. Code changes
  should then be localised to the modules that each spec explicitly owns.
- **AI Navigation:** AI agents should treat this structure as a map: start
  from steering (`.kiro/steering/`), locate the relevant spec(s), and use
  the code and test layout (`cryptofeed/`, `tests/`, `docs/`) as informed by
  those specs. Multi-area edits should only happen when driven by
  cross-referenced spec updates, not by convenience.

