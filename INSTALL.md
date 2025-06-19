# Cryptofeed Installation

The Cryptofeed library is intended for use by Python developers.

**Requirements:**

- Python 3.9+
- Optional: [uv](https://github.com/astral-sh/uv) for faster dependency management

## Installation Methods

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager with superior dependency resolution and installation speed.

**1. Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Basic Installation:**

```bash
uv add cryptofeed
```

**3. With All Optional Dependencies:**

```bash
uv add cryptofeed[all]
```

### Using pip (Traditional)

**Basic Installation:**

```bash
pip install --user --upgrade cryptofeed
```

**With All Optional Dependencies:**

```bash
pip install --user --upgrade cryptofeed[all]
```

## Backend-Specific Installation

Cryptofeed supports many backends including Redis, ZeroMQ, RabbitMQ, MongoDB, PostgreSQL, Google Cloud, and others. Backend dependencies are optional to minimize installation footprint.

### Using uv

- **All backends:**

  ```bash
  uv add cryptofeed[all]
  ```

- **Arctic backend:**

  ```bash
  uv add cryptofeed[arctic]
  ```

- **Google Cloud Pub/Sub backend:**

  ```bash
  uv add cryptofeed[gcp_pubsub]
  ```

- **Kafka backend:**

  ```bash
  uv add cryptofeed[kafka]
  ```

- **MongoDB backend:**

  ```bash
  uv add cryptofeed[mongo]
  ```

- **PostgreSQL backend:**

  ```bash
  uv add cryptofeed[postgres]
  ```

- **QuasarDB backend:**

  ```bash
  uv add cryptofeed[quasardb]
  ```

- **RabbitMQ backend:**

  ```bash
  uv add cryptofeed[rabbit]
  ```

- **Redis backend:**

  ```bash
  uv add cryptofeed[redis]
  ```

- **ZeroMQ backend:**
  ```bash
  uv add cryptofeed[zmq]
  ```

### Using pip

Replace `uv add` with `pip install --user --upgrade` for any of the above commands.

## Development Installation

### Using uv (Recommended)

**1. Clone the repository:**

```bash
git clone https://github.com/bmoscon/cryptofeed.git
cd cryptofeed
```

**2. Create virtual environment and install dependencies:**

```bash
uv venv                     # Create virtual environment  
source .venv/bin/activate   # Activate it (Linux/macOS)
uv pip install -e .        # Install core dependencies

# Optional: Install with specific backend dependencies
uv pip install -e ".[arctic]"   # Arctic backend
uv pip install -e ".[all]"      # All optional dependencies
```

### Development Dependency Groups

The project uses uv dependency groups for organized development:

```bash
uv sync --group test        # Testing dependencies only
uv sync --group lint        # Code quality tools
uv sync --group build       # Build tools
uv sync --group security    # Security scanning tools
uv sync --group performance # Performance benchmarking
uv sync --group quality     # Code complexity analysis
```

### Alternative Development Setup (pip)

```bash
git clone https://github.com/bmoscon/cryptofeed.git
cd cryptofeed
pip install -e .  # Editable installation
```

If you have a problem with the installation/hacking of Cryptofeed, you are welcome to:

- open a new issue: https://github.com/bmoscon/cryptofeed/issues/
- join us on Slack: [cryptofeed-dev.slack.com](https://join.slack.com/t/cryptofeed-dev/shared_invite/enQtNjY4ODIwODA1MzQ3LTIzMzY3Y2YxMGVhNmQ4YzFhYTc3ODU1MjQ5MDdmY2QyZjdhMGU5ZDFhZDlmMmYzOTUzOTdkYTZiOGUwNGIzYTk)
- or on GitHub Discussion: https://github.com/bmoscon/cryptofeed/discussions

Your Pull Requests are also welcome, even for minor changes.

## Building Wheels

### Local Wheel Building

**Build with hatch (recommended):**

```bash
# Install build dependencies
pip install hatch

# Build wheel and source distribution
hatch build

# Build wheel only
hatch build --target wheel

# Build source distribution only  
hatch build --target sdist

# Clean previous builds
hatch clean
```

**Build with build (alternative):**

```bash
# Install build dependencies
pip install build

# Build both wheel and sdist
python -m build

# Build wheel only
python -m build --wheel

# Build sdist only
python -m build --sdist
```

**Build with uv (fastest):**

```bash
# Native UV build (recommended - fastest)
uv build                    # Build both wheel and sdist
uv build --wheel           # Build wheel only
uv build --sdist           # Build source distribution only

# Alternative: UV + hatch in managed environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install hatch
hatch build

# Alternative: UV run approach (may have dependency conflicts)
uv run hatch build
uv run python -m build
```

### Cross-Platform Wheel Building (CI/CD)

The project uses GitHub Actions with UV-accelerated cibuildwheel for automated cross-platform wheel building:

**Supported platforms:**
- Linux x86_64 (Ubuntu latest)
- macOS x86_64 (Intel)
- macOS ARM64 (Apple Silicon)

**Trigger wheel builds:**

1. **Tag-based release (recommended):**
   ```bash
   git tag v2.4.2
   git push origin v2.4.2
   ```

2. **Manual trigger:**
   - Go to GitHub Actions → "Build Wheels" workflow
   - Click "Run workflow"

3. **GitHub release:**
   - Create a new release on GitHub
   - Wheels are automatically built and uploaded

**Performance optimizations:**
- Uses `uvx cibuildwheel` for tool isolation and speed
- UV build frontend (`build[uv]`) for 10-100x faster dependency resolution
- UV pip for rapid build dependency installation

**Python versions supported:**
- Python 3.9, 3.10, 3.11, 3.12

**Build configuration:**
- Uses hatch as build backend with UV frontend for maximum speed
- Cython extensions optimized with `-O3 -ffast-math`
- Object pooling with `@cython.freelist(128)`
- No assertions in release builds for maximum performance
- UV-accelerated dependency resolution and installation

**Artifacts:**
- Wheels: `cryptofeed-{version}-{python}-{abi}-{platform}.whl`
- Source: `cryptofeed-{version}.tar.gz`
- Automatically uploaded to PyPI on releases
