from polygon import RESTClient
from tickers import tickers
import json
import apikey

def save_raw_market_data_as_json(tickers: list[str], 
                                 multiplier: int,
                                 timespan: str,
                                 start: str,
                                 end: str,
                                 adjusted: str='true',
                                 sort: str='asc'):
    client = RESTClient(apikey.apikey)
    for ticker in tickers:
        aggregates = []
        for aggregate in client.list_aggs(
            ticker,
            multiplier,
            timespan,
            start,
            end,
            adjusted=adjusted,
            sort=sort,
            limit=120
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
        file_path = f'./data/data/raw/{multiplier}_{timespan}/{ticker}.json'
        with open (file_path, 'w') as file:
            json.dump(aggregates, file, indent=4)
            
def main():
    # get minute market data from 2015-05-06 4:00 AM to 2025-05-06 8:00 PM
    save_raw_market_data_as_json(tickers, 1, 'minute', '1430899200000', '1746573780000')
    # save_raw_market_data_as_json(tickers, 5, 'minute', '1430899200000', '1430899500000')

if __name__ == '__main__':
    main()