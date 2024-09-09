"""
Copyright (C) 2018-2024 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""
from cryptofeed import FeedHandler
from cryptofeed.backends.deltalake import FundingDeltaLake, TickerDeltaLake, TradeDeltaLake
from cryptofeed.defines import FUNDING, TICKER, TRADES
from cryptofeed.exchanges import Binance


def main():
    f = FeedHandler()
    
    # Define the Delta Lake base path (can be local or S3)
    delta_base_path = 's3://your-bucket/path/to/delta/tables'

    # S3 storage options (remove if using local storage)
    s3_options = {
        "AWS_ACCESS_KEY_ID": "your_access_key",
        "AWS_SECRET_ACCESS_KEY": "your_secret_key",
        "AWS_REGION": "your_region"
    }

    # Add Binance feed with Delta Lake callbacks
    f.add_feed(Binance(
        channels=[TRADES, FUNDING, TICKER],
        symbols=['BTC-USDT', 'ETH-USDT'],
        callbacks={
            TRADES: TradeDeltaLake(
                base_path=delta_base_path, 
                optimize_interval=50,  # More frequent table optimization
                time_travel=True,  # Enable time travel feature
                storage_options=s3_options  # Add S3 configuration
            ),
            FUNDING: FundingDeltaLake(
                base_path=delta_base_path,
                storage_options=s3_options  # Add S3 configuration
            ),
            TICKER: TickerDeltaLake(
                base_path=delta_base_path,
                partition_cols=['exchange', 'symbol', 'year', 'month', 'day'],  # Custom partitioning
                z_order_cols=['timestamp', 'bid', 'ask'],  # Enable Z-ordering
                storage_options=s3_options  # Add S3 configuration
            )
        }
    ))
    
    f.run()


if __name__ == '__main__':
    main()