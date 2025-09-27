# Requirements Document

## Introduction
This spec covers Backpack exchange integration using the same native cryptofeed architecture employed by Binance (and other first-party exchanges). The goal is to deliver a complete Backpack feed that reuses the stock `Feed` subclass, `HTTPAsyncConn`/`WSAsyncConn`, and adapter patterns without relying on `ccxt` or the generic ccxt wrapper. The ccxt-based feed will be removed from this branch; the native path stands alone and mirrors the Binance structure end-to-end.

## Requirements

### Requirement 1: Exchange Configuration
**Objective:** As a deployer, I want Backpack-specific config options exposed cleanly, so that enabling the exchange is straightforward.

#### Acceptance Criteria
1. WHEN Backpack config is loaded THEN it SHALL follow cryptofeed Feed configuration patterns with Backpack-specific options (API key format, ED25519 authentication, sandbox endpoints).
2. IF optional features (private channels) require additional keys THEN validation SHALL enforce presence when enabled.
3. WHEN config is invalid THEN descriptive errors SHALL reference Backpack-specific fields and ED25519 key requirements.

### Requirement 2: Transport Behavior
**Objective:** As an operator, I want Backpack HTTP and WebSocket transports to leverage existing cryptofeed infrastructure and proxy support, so that infrastructure remains consistent.

#### Acceptance Criteria
1. WHEN Backpack REST requests execute THEN they SHALL use the same `HTTPAsyncConn` scaffolding as Binance, pointing to Backpack endpoints (`https://api.backpack.exchange/`) with proxy support.
2. WHEN Backpack WebSocket sessions connect THEN they SHALL reuse the Binance-style `WSAsyncConn` integration (`wss://ws.backpack.exchange/`) with proxy support and identical lifecycle hooks.
3. IF Backpack requires subscription mapping or ED25519 authentication THEN the integration SHALL implement helpers in the native module without modifying generic ccxt classes.

### Requirement 3: Data Normalization
**Objective:** As a downstream consumer, I want Backpack trade/book data normalized like other exchanges, so pipelines remain uniform.

#### Acceptance Criteria
1. WHEN Backpack emits trades/books THEN the integration SHALL convert payloads into cryptofeed `Trade`/`OrderBook` objects using native parsing methods.
2. IF Backpack supplies additional metadata (e.g., order types, microsecond timestamps) THEN optional fields SHALL be passed through without breaking consumers.
3. WHEN data discrepancies occur THEN logging SHALL highlight the mismatch and skip invalid entries.

### Requirement 4: Symbol Management
**Objective:** As a user, I want Backpack symbols normalized to cryptofeed standards, so symbol handling is consistent across exchanges.

#### Acceptance Criteria
1. WHEN Backpack symbols are loaded THEN they SHALL be converted from Backpack format to cryptofeed Symbol objects.
2. WHEN symbol mapping occurs THEN Backpack instrument types SHALL be properly identified (SPOT, FUTURES, etc.).
3. WHEN users request symbols THEN both normalized and exchange-specific formats SHALL be available.

### Requirement 5: ED25519 Authentication
**Objective:** As a user with API credentials, I want private channel access using Backpack's ED25519 signature scheme.

#### Acceptance Criteria
1. WHEN private channels are requested THEN ED25519 key pairs SHALL be used for signature generation.
2. WHEN authentication fails THEN descriptive error messages SHALL explain ED25519 key format requirements.
3. WHEN signatures are generated THEN they SHALL follow Backpack's exact specification (base64 encoding, microsecond timestamps).

### Requirement 6: Testing & Documentation
**Objective:** As a maintainer, I want tests and docs for Backpack's integration, ensuring the exchange stays healthy over time.

#### Acceptance Criteria
1. WHEN unit tests run THEN they SHALL cover configuration, ED25519 authentication, subscription mapping, and data parsing logic unique to Backpack.
2. WHEN integration tests execute THEN they SHALL confirm proxy-aware transports, WebSocket subscriptions, and normalized callbacks using fixtures or sandbox endpoints.
3. WHEN documentation is updated THEN the exchange list SHALL include Backpack with setup steps referencing native cryptofeed patterns and ED25519 key generation.
