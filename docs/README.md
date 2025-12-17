## Cryptofeed Documentation

### Core Documentation
* [High level](high_level.md)
* [Data Types](dtypes.md)
* [Callbacks](callbacks.md)
* [Adding a new exchange](exchange.md)
* [Data Integrity for Orderbooks](book_validation.md)
* [Configuration](config.md)
* [Authenticated Channels](auth_channels.md)
* [Performance Considerations](performance.md)
* [REST endpoints](rest.md)

### 🔄 Enterprise Features
* **[Kafka Backend](kafka/)** - High-performance Apache Kafka integration with Protocol Buffer support
  * **[Quick Start](kafka/README.md)** - Basic usage and configuration
  * **[User Guide](kafka/user-guide.md)** - Advanced configuration and migration
  * **[Technical Specification](kafka/technical-specification.md)** - Detailed technical documentation
  * **[Architecture](kafka/architecture.md)** - System design and components
* **[Transparent Proxy System](specs/proxy_system_overview.md)** - Zero-code proxy support for all exchanges
* **[Proxy Testing Overview](proxy/testing.md)** - Test suites and execution guidance for proxy integration
* **[Technical Specifications](specs/)** - Detailed specs for advanced features and integrations

### 📋 Development & Specifications
* **[Normalized Data Schema](specs/normalized-data-schema/)** - Baseline schemas, governance framework, and monitoring infrastructure
* **[E2E Testing](e2e/)** - End-to-end test infrastructure, results, and planning
  * **[Planning & Coordination](e2e/planning/)** - Test plans, commit strategies, and execution documentation
  * **[Test Results & Analysis](e2e/results/)** - Detailed test execution results and validations

### 🔬 Technical Investigations
* **[Issues & Investigations](investigations/)** - Technical deep-dives into discovered issues and resolutions
