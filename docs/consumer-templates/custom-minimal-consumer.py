"""
Minimal Consumer Template for Cryptofeed Consolidated Topics (25 lines)

A bare-bones example showing the essential pattern for consuming cryptofeed
data from consolidated Kafka topics. Perfect as a starting point for custom
implementations.

Key Features:
- Uses kafka-python (most common library)
- Minimal dependencies (only kafka-python + protobuf)
- Simple consumer loop
- Header extraction for filtering
- Basic error handling
- Easy to extend and customize
"""

from kafka import KafkaConsumer
from cryptofeed.schema.v1 import trade_pb2

# 1. Create consumer for consolidated topics
consumer = KafkaConsumer(
    'cryptofeed.trades',  # Subscribe to consolidated topic
    bootstrap_servers=['localhost:9092'],
    group_id='my-consumer',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: m,  # Keep as bytes, we'll deserialize
)

# 2. Consume messages
for message in consumer:
    try:
        # Extract headers
        headers = dict(message.headers or [])
        exchange = headers.get(b'exchange', b'unknown').decode()
        symbol = headers.get(b'symbol', b'unknown').decode()

        # Deserialize protobuf
        trade = trade_pb2.Trade()
        trade.ParseFromString(message.value)

        # Print message
        print(f"{exchange}:{symbol} @ {trade.price} ({trade.side})")

        # TODO: Add your processing logic here
        # - Transform data
        # - Store in database
        # - Send to downstream systems

    except Exception as e:
        print(f"Error processing message: {e}")
        # Send to DLQ (optional)
        continue

# 3. Graceful exit
consumer.close()
