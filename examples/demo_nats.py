"""
Cryptofeed NATS Backend Demo

This script demonstrates how to use the NATS backend with Cryptofeed to publish
market data from exchanges to NATS subjects.

Prerequisites:
1. Install Cryptofeed with NATS support:
   pip install cryptofeed[nats]

2. Run a NATS server. You can use Docker for a quick setup:
   docker run -p 4222:4222 -p 8222:8222 --name nats-main -ti nats:latest

   This command starts a NATS server accessible at nats://localhost:4222
   and exposes the monitoring interface at http://localhost:8222.

To run this script:
   python examples/demo_nats.py

You can then use a NATS client (e.g., nats-cli, or another NATS client library)
to subscribe to the subjects and see the data flowing. For example, using nats-cli:

   nats sub "cryptofeed.demo.>"

This will subscribe to all subjects under the 'cryptofeed.demo.' prefix.
You should see trade and ticker data for BTC-USD from Coinbase.
"""
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Coinbase
from cryptofeed.defines import TRADES, TICKER, L2_BOOK
from cryptofeed.backends.nats import TradeNATS, TickerNATS, BookNATS

# Configuration for NATS connection
# Replace with your NATS server address(es) if different
NATS_SERVER_ADDR = 'nats://localhost:4222'
# NATS_SERVER_ADDR = ['nats://server1:4222', 'nats://server2:4222'] # Example for multiple servers

# Custom prefix for NATS subjects
NATS_SUBJECT_PREFIX = 'cryptofeed.demo'


def main():
    # Create a FeedHandler instance
    fh = FeedHandler()

    # Define the NATS callbacks
    # Each callback instance can have its own subject_prefix or share one.
    # Data will be published to subjects like:
    #   <NATS_SUBJECT_PREFIX>-trades-<exchange>-<symbol>
    #   <NATS_SUBJECT_PREFIX>-ticker-<exchange>-<symbol>
    #   <NATS_SUBJECT_PREFIX>-book-<exchange>-<symbol>

    trade_cb = TradeNATS(addr=NATS_SERVER_ADDR, subject_prefix=NATS_SUBJECT_PREFIX)
    ticker_cb = TickerNATS(addr=NATS_SERVER_ADDR, subject_prefix=NATS_SUBJECT_PREFIX)
    # Example for L2 Book data
    book_cb = BookNATS(addr=NATS_SERVER_ADDR, subject_prefix=NATS_SUBJECT_PREFIX, book_depth=10)


    # Add subscriptions to the FeedHandler
    # We'll subscribe to Coinbase trades and tickers for BTC-USD
    fh.add_feed(Coinbase(
        symbols=['BTC-USD'],
        channels=[TRADES, TICKER, L2_BOOK], # Subscribe to trades, tickers, and L2 order book
        callbacks={
            TRADES: trade_cb,
            TICKER: ticker_cb,
            L2_BOOK: book_cb
        }
    ))

    print(f"Starting FeedHandler. Publishing data to NATS server at {NATS_SERVER_ADDR}")
    print(f"Subscribing to Coinbase BTC-USD trades, tickers, and L2 Book.")
    print(f"NATS subjects will be prefixed with: {NATS_SUBJECT_PREFIX}")
    print(f"Example subjects:")
    print(f"  Trades:   {NATS_SUBJECT_PREFIX}-trades-COINBASE-BTC-USD")
    print(f"  Tickers:  {NATS_SUBJECT_PREFIX}-ticker-COINBASE-BTC-USD")
    print(f"  L2 Book:  {NATS_SUBJECT_PREFIX}-book-COINBASE-BTC-USD")
    print("Use a NATS client to subscribe, e.g., 'nats sub \"cryptofeed.demo.>\"'")
    print("Press CTRL+C to stop.")

    # Run the FeedHandler
    # This will block and start processing data
    fh.run()


if __name__ == '__main__':
    main()
