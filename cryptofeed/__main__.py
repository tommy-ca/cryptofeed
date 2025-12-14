"""
Entry point for running cryptofeed as a module.

Allows running cryptofeed with:
    python -m cryptofeed --config /config/config.yaml

This delegates to cryptofeed.run module.
"""
from cryptofeed.run import main

if __name__ == '__main__':
    main()
