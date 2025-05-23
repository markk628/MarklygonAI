import json
from polygon import RESTClient

from src.utils import utils
from src.config.config import (
    DATA_DIR, 
    DATA_START_DATE, 
    DATA_FREQUENCY, 
    POLYGON_APIKEY, 
    TICKERS
)

def save_raw_market_data_as_json(tickers: list[str], 
                                 multiplier: int,
                                 timespan: str,
                                 start: str,
                                 end: str,
                                 adjusted: str='true',
                                 sort: str='asc'):
    client = RESTClient(POLYGON_APIKEY)
    for ticker in tickers:
        aggregates = []
        for aggregate in client.list_aggs(
            ticker,
            multiplier,
            timespan,
            start,
            end,
            adjusted=adjusted,
            sort=sort
        ):
            aggregates.append({
                'timestamp': aggregate.timestamp, 
                'open': aggregate.open, 
                'high': aggregate.high,
                'low': aggregate.low,
                'close': aggregate.close,
                'transactions': aggregate.transactions,
                'volume': aggregate.volume,
                'vwap': aggregate.vwap
            })
        base_dir = DATA_DIR / 'raw' / f'{multiplier}_{timespan}'
        utils.create_directory(base_dir)
        file_path = f'{base_dir}/{ticker}.json'
        with open (file_path, 'w') as file:
            json.dump(aggregates, file, indent=4)

if __name__ == '__main__':
    save_raw_market_data_as_json(TICKERS, 1, DATA_FREQUENCY, DATA_START_DATE, '1746573780000')