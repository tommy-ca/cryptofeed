"""
Cryptofeed NATS Backend Demo with JetStream

This script demonstrates how to use the NATS backend with Cryptofeed to publish
market data from exchanges to NATS subjects, specifically using NATS JetStream
for persistent streaming.

Prerequisites:
1. Install Cryptofeed with NATS support:
   pip install cryptofeed[nats]

2. Run a NATS server with JetStream enabled. JetStream is enabled by default in
   NATS server versions 2.2.0 and later. You can use Docker for a quick setup:
   docker run -p 4222:4222 -p 8222:8222 --name nats-main -ti nats:latest

   This command starts a NATS server accessible at nats://localhost:4222
   and exposes the monitoring interface at http://localhost:8222.
   If using an older NATS server, you might need to enable JetStream explicitly
   (e.g., by passing the `-js` flag to the nats-server command).

3. Create JetStream Streams (User Responsibility):
   For JetStream to capture and persist messages, you must create streams on the
   NATS server that subscribe to the subjects Cryptofeed will publish to.
   The NATS backend itself does not create these streams.

   Use the NATS CLI to create streams. Examples based on this demo's configuration:
   (Assumes NATS_SUBJECT_PREFIX = 'cryptofeed.demo')

   nats stream add CRYPTO_TRADES --subjects "cryptofeed.demo-trades.>" --ack --storage file --retention limits --defaults
   nats stream add CRYPTO_TICKERS --subjects "cryptofeed.demo-ticker.>" --ack --storage file --retention limits --defaults
   nats stream add CRYPTO_BOOKS --subjects "cryptofeed.demo-book.>" --ack --storage file --retention limits --defaults

   Explanation of common flags:
   - `CRYPTO_TRADES`: Arbitrary name for your stream.
   - `--subjects "cryptofeed.demo-trades.>" `: Defines which subjects this stream captures.
     The subject pattern should match what the backend produces (e.g., <prefix>-<datatype>-<exchange>-<symbol>).
     Adjust if your `NATS_SUBJECT_PREFIX` or data types change. `>` is a wildcard for one or more tokens at the end of a subject.
     For example, `cryptofeed.demo-trades.COINBASE.BTC-USD` would be matched by `cryptofeed.demo-trades.>`.
   - `--ack`: Enables acknowledgment, ensuring at-least-once delivery.
   - `--storage file`: Persists messages to the filesystem. `memory` is another option.
   - `--retention limits`: Retains messages based on limits (e.g., max messages, max bytes, max age).
     Other policies: `interest` (messages kept if consumers are active), `workqueue`.
   - `--defaults`: Uses default JetStream settings for some parameters if not specified.

To run this script:
   python examples/demo_nats.py

Viewing Data with JetStream:
You can use the NATS CLI to subscribe and view messages. To see messages from the beginning
of the stream (if retained), use the subject patterns that match the stream's configuration:

   nats sub "cryptofeed.demo-trades.>" --deliver-all
   nats sub "cryptofeed.demo-ticker.>" --deliver-all
   nats sub "cryptofeed.demo-book.>" --deliver-all

   For more advanced consumption (e.g., durable consumers, specific delivery policies),
   refer to NATS JetStream documentation.
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

# JetStream Configuration
USE_JETSTREAM = True
JETSTREAM_TIMEOUT = 5.0 # Timeout for JetStream publish ACKs (in seconds)


def main():
    # Create a FeedHandler instance
    fh = FeedHandler()

    # Define the NATS callbacks
    # Data will be published to subjects like:
    #   <NATS_SUBJECT_PREFIX>-trades-<exchange>-<symbol>
    #   <NATS_SUBJECT_PREFIX>-ticker-<exchange>-<symbol>
    #   <NATS_SUBJECT_PREFIX>-book-<exchange>-<symbol>

    common_nats_options = {
        'addr': NATS_SERVER_ADDR,
        'subject_prefix': NATS_SUBJECT_PREFIX,
        'jetstream_mode': USE_JETSTREAM,
        'jetstream_timeout': JETSTREAM_TIMEOUT
    }

    trade_cb = TradeNATS(**common_nats_options)
    ticker_cb = TickerNATS(**common_nats_options)
    # Example for L2 Book data
    book_cb = BookNATS(**common_nats_options, book_depth=10)


    # Add subscriptions to the FeedHandler
    # We'll subscribe to Coinbase trades, tickers, and L2 book for BTC-USD
    fh.add_feed(Coinbase(
        symbols=['BTC-USD'],
        channels=[TRADES, TICKER, L2_BOOK],
        callbacks={
            TRADES: trade_cb,
            TICKER: ticker_cb,
            L2_BOOK: book_cb
        }
    ))

    print(f"Starting FeedHandler.")
    if USE_JETSTREAM:
        print(f"Publishing data to NATS JetStream at {NATS_SERVER_ADDR}")
        print(f"Ensure JetStream streams are configured to capture subjects starting with '{NATS_SUBJECT_PREFIX}.*'")
        print(f"Example stream creation for trades: nats stream add TRADES_STREAM --subjects \"{NATS_SUBJECT_PREFIX}.trades.>\" --ack --storage file")
    else:
        print(f"Publishing data to core NATS at {NATS_SERVER_ADDR}")

    print(f"Subscribing to Coinbase BTC-USD: Trades, Tickers, L2 Book.")
    print(f"NATS subjects will be prefixed with: {NATS_SUBJECT_PREFIX}")
    print(f"Example subjects (actual subjects use hyphens as separators after prefix):")
    print(f"  Trades:   {NATS_SUBJECT_PREFIX}-trades-COINBASE-BTC-USD")
    print(f"  Tickers:  {NATS_SUBJECT_PREFIX}-ticker-COINBASE-BTC-USD")
    print(f"  L2 Book:  {NATS_SUBJECT_PREFIX}-book-COINBASE-BTC-USD")

    if USE_JETSTREAM:
        print(f"To view data, subscribe to subjects using the correct pattern, e.g., 'nats sub \"{NATS_SUBJECT_PREFIX}-trades.>\" --deliver-all'")
    else:
        print(f"To view data, subscribe to subjects, e.g., 'nats sub \"{NATS_SUBJECT_PREFIX}-trades.>\"'")
    print("Press CTRL+C to stop.")

    # Run the FeedHandler
    # This will block and start processing data
    fh.run()


if __name__ == '__main__':
    main()
