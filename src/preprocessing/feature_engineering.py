import pandas as pd
import numpy as np
import talib as ta
import json
import concurrent.futures

from src.config.config import DATA_DIR, CUTOFF_TIMESTAMP, TICKERS
from src.utils.database import DatabaseManager
from src.utils import utils

class FeatureEngineer:
    def __init__(self, timestamp: str=CUTOFF_TIMESTAMP, use_json: bool=False):
        self.timestamp = timestamp
        self.use_json = use_json

    def _get_data(self, ticker: str) -> pd.DataFrame:
        """
        Returns a dataframe of the ticker's raw data
        
        티커의 raw 데이터를 받아서 데이터프레임으로 반환
        """
        print(f'retrieving {ticker}...')
        if self.use_json:
            file_path = DATA_DIR / 'raw/1_minute'
            with open(f'{file_path}/{ticker}.json', 'r') as file:
                data = json.load(file)
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
            return df
        database_manager = DatabaseManager()
        df = database_manager.get_ticker_df(ticker.lower())
        return df

    def _handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data
        
        누락된 데이터 처리
        """
        print('handling gaps...')
        df = df.set_index('timestamp').resample('min').asfreq()
        df = df.between_time('08:00', '23:59').copy() # .copy() avoids chained indexing
        cols_to_ffill = ['open', 'high', 'low', 'close', 'vwap']
        cols_to_fillna = ['volume', 'transactions']
        df[cols_to_ffill] = df[cols_to_ffill].ffill()
        df[cols_to_fillna] = df[cols_to_fillna].fillna(0)
        df = df[~df.index.weekday.isin([5, 6])]
        return df.reset_index()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add techincal indicators
        
        보조지표 추가
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        indicators = {
            'stochrsi_k_14_1min,stochrsi_d_14_1min': (ta.STOCHRSI, {'real': close, 'timeperiod': 14}),
            'stochrsi_k_21_1min,stochrsi_d_21_1min': (ta.STOCHRSI, {'real': close, 'timeperiod': 21}),
            'rsi_7_1min': (ta.RSI, {'real': close, 'timeperiod': 7}),
            'rsi_14_1min': (ta.RSI, {'real': close, 'timeperiod': 14}),
            'rsi_21_1min': (ta.RSI, {'real': close, 'timeperiod': 21}),
            'macd_5_13_4_1min,macd_signal_5_13_4_1min,macd_hist_5_13_4_1min': (ta.MACD, {'real': close, 'fastperiod': 5, 'slowperiod': 13, 'signalperiod': 4}),
            'macd_12_26_9_1min,macd_signal_12_26_9_1min,macd_hist_12_26_9_1min': (ta.MACD, {'real': close, 'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            'roc_5_1min': (ta.ROC, {'real': close, 'timeperiod': 5}),
            'roc_10_1min': (ta.ROC, {'real': close, 'timeperiod': 10}),
            'roc_20_1min': (ta.ROC, {'real': close, 'timeperiod': 20}),
            'ultosc_5_15_30_1min': (ta.ULTOSC, {'high': high, 'low': low, 'close': close, 'timeperiod1': 5, 'timeperiod2': 15, 'timeperiod3': 30}),
            'ultosc_7_21_42_1min': (ta.ULTOSC, {'high': high, 'low': low, 'close': close, 'timeperiod1': 7, 'timeperiod2': 21, 'timeperiod3': 42}),
            'ultosc_10_30_60_1min': (ta.ULTOSC, {'high': high, 'low': low, 'close': close, 'timeperiod1': 10, 'timeperiod2': 30, 'timeperiod3': 60}),
            'plusdi_10_1min': (ta.PLUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 10}),
            'plusdi_20_1min': (ta.PLUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'plusdi_30_1min': (ta.PLUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 30}),
            'minusdi_10_1min': (ta.MINUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 10}),
            'minusdi_20_1min': (ta.MINUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'minusdi_30_1min': (ta.MINUS_DI, {'high': high, 'low': low, 'close': close, 'timeperiod': 30}),
            'adx_10_1min': (ta.ADX, {'high': high, 'low': low, 'close': close, 'timeperiod': 10}),
            'adx_20_1min': (ta.ADX, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'adx_30_1min': (ta.ADX, {'high': high, 'low': low, 'close': close, 'timeperiod': 30}),
            'cci_10_1min': (ta.CCI, {'high': high, 'low': low, 'close': close, 'timeperiod': 10}),
            'cci_20_1min': (ta.CCI, {'high': high, 'low': low, 'close': close, 'timeperiod': 20}),
            'cci_30_1min': (ta.CCI, {'high': high, 'low': low, 'close': close, 'timeperiod': 30}),
            'ema_3_1min': (ta.EMA, {'real': close, 'timeperiod': 3}),
            'ema_5_1min': (ta.EMA, {'real': close, 'timeperiod': 5}),
            'ema_9_1min': (ta.EMA, {'real': close, 'timeperiod': 9}),
            'ema_21_1min': (ta.EMA, {'real': close, 'timeperiod': 21}),
            'ema_50_1min': (ta.EMA, {'real': close, 'timeperiod': 50}),
            'sma_5_1min': (ta.SMA, {'real': close, 'timeperiod': 5}),
            'sma_10_1min': (ta.SMA, {'real': close, 'timeperiod': 10}),
            'sma_20_1min': (ta.SMA, {'real': close, 'timeperiod': 20}),
            'sma_50_1min': (ta.SMA, {'real': close, 'timeperiod': 50}),
            'obv_1min': (ta.OBV, {'close': close, 'volume': volume}),
            'mfi_7_1min': (ta.MFI, {'high': high, 'low': low, 'close': close, 'volume': volume, 'timeperiod': 7}),
            'mfi_14_1min': (ta.MFI, {'high': high, 'low': low, 'close': close, 'volume': volume, 'timeperiod': 14}),
            'mfi_21_1min': (ta.MFI, {'high': high, 'low': low, 'close': close, 'volume': volume, 'timeperiod': 21}),
            'bband_upper_10_1min,bband_middle_10_1min,bband_lower_10_1min': (ta.BBANDS, {'real': close, 'timeperiod': 10}),
            'bband_upper_20_1min,bband_middle_20_1min,bband_lower_20_1min': (ta.BBANDS, {'real': close, 'timeperiod': 20}),
            'bband_upper_50_1min,bband_middle_50_1min,bband_lower_50_1min': (ta.BBANDS, {'real': close, 'timeperiod': 50}),
            'atr_14_1min': (ta.ATR, {'high': high, 'low': low, 'close': close, 'timeperiod': 14}),
            'atr_21_1min': (ta.ATR, {'high': high, 'low': low, 'close': close, 'timeperiod': 21})
        }

        print('adding technical indicators...')
        for out_cols, (func, params) in indicators.items():
            if out_cols == 'obv_1min':
                results = func(*params.values())
            else:
                results = func(**params)
            if isinstance(results, tuple):
                for i, col_name in enumerate(out_cols.split(',')):
                    df[col_name] = results[i]
            else:
                df[out_cols] = results
        # df['obv_1min'] = ta.OBV(close, volume)
        return df

    def _add_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds temporal patterns (minute, hour, day)
        
        시간 패턴(분, 시간, 일) 추가
        """
        print('adding temporal patterns...')
        timestamp = df['timestamp']
        df['minute'] = timestamp.dt.minute
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['hour'] = timestamp.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day'] = timestamp.dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 5)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 5)
        df['month'] = timestamp.dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 5)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 5)
        df['quarter'] - timestamp.dt.quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 5)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 5)
        return df

    def _add_last_significant_change(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Adds time since the last significant price change
        
        마지막 큰 가격 변경 이후 시간 추가
        """
        print('adding last significant change...')
        price_change_pct = df['close'].pct_change().abs()
        significant_change = (price_change_pct > threshold).astype(int)
        group_changes = (significant_change != significant_change.shift()).cumsum()
        df['time_since_last_significant_change'] = df.groupby(group_changes).cumcount() + 1
        return df

    def _add_lagged_features(self, df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
        """
        Adds lagged features for 'open', 'high', 'low', 'close', 'volume', 'vwap' columns
        
        'open', 'high', 'low', 'close', 'volume', 'vwap'열에 대해 지연된 피처를 추가
        """
        print('adding lagged features...')
        for lag in lags:
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def _add_windowed_statistics(self, df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
        """
        Adds rolling window statistics for close and volume
        
        close와 volume에 대한 롤링 창 통계를 추가
        """
        print('adding windowed statistics...')
        for window in windows:
            for col in ['close', 'volume']:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        return df

    def _add_price_differences_and_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds price differences and related features
        
        가격 차이 및 관련 피처 추가
        """
        print('adding price differences and related features...')
        df['close_diff_1'] = df['close'].diff(1)
        df['close_pct_change_1'] = df['close'].pct_change(1)
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_range'] = df['high'] - df['low']
        df['close_open_range'] = df['close'] - df['open']
        return df

    def _add_volatility_measures(self, df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
        """
        Adds volatility measures of rolling windows
        
        롤링 윈도우의 변동성 측정을 추가
        """
        print('adding volatility measures...')
        for window in windows:
            df[f'log_return_rolling_std_{window}'] = df['log_return_1'].rolling(window=window).std()
        return df

    def _add_OHLC_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds OHLC ratios
        
        OHLC 비율들 추가
        """
        print('adding OHLC ratios...')
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a target variable based on price direction
        
        추세 타겟 추가
        """
        print('adding targets...')
        diff = df['close'].diff()
        # up: 1, down: -1, no change: 0
        df['target'] = np.sign(diff)
        return df

    def _drop_rows_before_timestamp(self, df: pd.DataFrame, timestamp: str) -> pd.DataFrame:
        """
        Drops rows before a specified timestamp
        
        지정된 타임스탬프 이전의 행들 삭제
        """
        print('dropping rows...')
        cutoff_timestamp = pd.to_datetime(timestamp).tz_localize('UTC')
        return df[df['timestamp'] >= cutoff_timestamp]

    def _save_feature_engineer_ticker(self, ticker: str, timestamp: str):
        """
        Feature engineers data for a single ticker
        
        하나의 티커에 피처 엔지니어링
        """
        print(f'starting feature engineering for {ticker}')
        df = self._get_data(ticker)
        df = self._handle_gaps(df)
        df = self._add_technical_indicators(df)
        df = self._add_temporal_patterns(df)
        df = self._add_last_significant_change(df, threshold=0.001)
        df = self._add_lagged_features(df, [1, 5, 10])
        df = self._add_windowed_statistics(df, [5, 15, 30])
        df = self._add_price_differences_and_returns(df)
        df = self._add_volatility_measures(df, [15, 30])
        df = self._add_OHLC_ratios(df)
        df = self._add_targets(df)
        df = self._drop_rows_before_timestamp(df, timestamp)
        base_dir = DATA_DIR / 'feature_engineered'
        utils.create_directory(base_dir)
        df.to_csv(f'{base_dir}/{ticker}.csv', index=False)
        print(f'{ticker} dataframe saved succesfully')

    def save_feature_engineered_tickers(self, tickers: list[str]=TICKERS):
        """
        Feature engineers data for multiple tickers and saves the results
        
        여러 티커를 피처 엔지니어링한 후 저장
        """
        
        for ticker in tickers:
            self._save_feature_engineer_ticker(ticker, self.timestamp)

if __name__ == '__main__':
    feature_engineer = FeatureEngineer()
    feature_engineer.save_feature_engineered_tickers()