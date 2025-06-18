import argparse
import asyncio

import websockets


parser = argparse.ArgumentParser()
parser.add_argument("--uri", default="wss://api.huobi.pro/ws", help="URI to connect to")
parser.add_argument("--sub", default='{"sub": "market.btcusdt.trade.detail", "id": 4}', help="Subscription string")
parser.add_argument("--count", default=3, type=int, help="Number of messages to receive before exiting")
parser.add_argument("-z", action="store_true", help="Use gzip on messages")
args = parser.parse_args()

uri = args.uri
sub = args.sub
is_gzip = args.z
count = args.count



async def main():
    async with websockets.connect(uri) as websocket:
        await websocket.send(sub)

        for _i in range(count):
            await websocket.recv()
            if not is_gzip:
                pass
            else:
                pass


asyncio.get_event_loop().run_until_complete(main())
