from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from typing import List
from src.utils import POLYGON_APIKEY

client = WebSocketClient(
	api_key=POLYGON_APIKEY,
	feed=Feed.RealTime,
	market=Market.Stocks
	)

client.subscribe("AM.AAPL") # single ticker
# client.subscribe("AM.AAPL", "AM.MSFT") # multiple tickers

def handle_msg(msgs: List[WebSocketMessage]):
    for m in msgs:
        print(m)

if __name__ == '__main__':
    client.run(handle_msg)