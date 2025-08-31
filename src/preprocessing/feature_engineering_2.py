import pandas as pd
import numpy as np
import talib as ta
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
from scipy import signal

from src.config.config import DATA_DIR, CUTOFF_TIMESTAMP, TICKERS, MINUTES_PER_TRADING_DAY
from src.utils.database import DatabaseManager
from src.utils.utils import create_directory, save_to_csv

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
        """Handle missing data and filter to regular market hours only"""
        print('handling gaps and filtering to regular market hours...')
        df = df.set_index('timestamp').resample('min').asfreq()
        
        # Convert to ET and filter to regular hours (9:30 AM - 4:00 PM)
        df_et = df.tz_convert('US/Eastern') if df.index.tz else df.tz_localize('UTC').tz_convert('US/Eastern')
        df_filtered = df_et.between_time('09:30', '15:59').copy()
        
        # Fill gaps and clean
        cols_to_ffill = ['open', 'high', 'low', 'close', 'vwap']
        cols_to_fillna = ['volume', 'transactions']
        df_filtered[cols_to_ffill] = df_filtered[cols_to_ffill].ffill()
        df_filtered[cols_to_fillna] = df_filtered[cols_to_fillna].fillna(0)
        df_filtered = df_filtered[~df_filtered.index.weekday.isin([5, 6])]
        
        return df_filtered.tz_convert('UTC').reset_index()

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price features
        
        가격 특성 추가
        """
        print('adding price features...')
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # 1-20차원: Multi-period returns - shortened for intraday
        # Very short term: 1-5 minutes, Short term: 5-20 minutes
        periods = [1, 2, 3, 5, 7, 10, 12, 15]
        for i, period in enumerate(periods):
            df[f'return_{period}m'] = ta.ROC(close, timeperiod=period)
        
        # 21-24차원: Normalized OHLC
        close_prev = close.shift(1)
        df['open_gap'] = ((open_price - close_prev) / close_prev)
        df['high_level'] = ((high - close_prev) / close_prev)
        df['low_level'] = ((low - close_prev) / close_prev)
        df['close_change'] = ta.ROC(close, timeperiod=1)
        
        # 25-32차원: Price momentum
        df['momentum_1m'] = ta.MOM(close, timeperiod=1)
        df['momentum_5m'] = ta.MOM(close, timeperiod=5)
        df['momentum_15m'] = ta.MOM(close, timeperiod=15)
        df['momentum_1m_accel'] = df['momentum_1m'].diff()
        df['momentum_5m_accel'] = df['momentum_5m'].diff()
        df['momentum_15m_accel'] = df['momentum_15m'].diff()
        # Momentum persistence - multi-timeframe approach
        
        # 1-minute momentum persistence (micro-signals)
        roc_1m = ta.ROC(close, timeperiod=1)
        bullish_1m = (roc_1m > 0).astype(int)
        bearish_1m = (roc_1m < 0).astype(int)
        df['momentum_persistence_bullish_1m'] = bullish_1m.rolling(5).sum() / 5 # 5-min window
        df['momentum_persistence_bearish_1m'] = bearish_1m.rolling(5).sum() / 5
        
        # 5-minute momentum persistence (short-term signals)
        roc_5m = ta.ROC(close, timeperiod=5)  
        bullish_5m = (roc_5m > 0).astype(int)
        bearish_5m = (roc_5m < 0).astype(int)
        df['momentum_persistence_bullish_5m'] = bullish_5m.rolling(3).sum() / 3  # 15-min window
        df['momentum_persistence_bearish_5m'] = bearish_5m.rolling(3).sum() / 3
        
        # 15-minute momentum persistence (medium-term context)
        roc_15m = ta.ROC(close, timeperiod=15)
        bullish_15m = (roc_15m > 0).astype(int)  
        bearish_15m = (roc_15m < 0).astype(int)
        df['momentum_persistence_bullish_15m'] = bullish_15m.rolling(2).sum() / 2  # 30-min window
        df['momentum_persistence_bearish_15m'] = bearish_15m.rolling(2).sum() / 2
        
        # ENHANCEMENT: Savitzky-Golay smoothed price features for noise reduction
        # Smoothed price signals (reduce noise while preserving trends)
        df['price_smoothed_5m'] = signal.savgol_filter(close, window_length=11, polyorder=3)
        df['price_smoothed_15m'] = signal.savgol_filter(close, window_length=31, polyorder=3)
        
        # Price deviation from smooth trend (detect short-term noise vs real moves)
        df['price_trend_deviation_5m'] = (close - df['price_smoothed_5m']) / df['price_smoothed_5m']
        df['price_trend_deviation_15m'] = (close - df['price_smoothed_15m']) / df['price_smoothed_15m']
        
        # Gradient-based momentum (cleaner than simple ROC)
        momentum_5m_gradient = np.gradient(df['price_smoothed_5m'])
        df['momentum_gradient_5m'] = signal.savgol_filter(momentum_5m_gradient, 15, 2)
        
        # 33-40차원: Volatility indicators - shortened for intraday
        df['volatility_5m'] = ta.STDDEV(close, timeperiod=5)   
        df['volatility_15m'] = ta.STDDEV(close, timeperiod=15) 
        df['atr_5m'] = ta.ATR(high, low, close, timeperiod=5)
        df['atr_15m'] = ta.ATR(high, low, close, timeperiod=15)
        df['parkinson_vol'] = np.where(low > 0, (np.log(high/low)**2 / (4*np.log(2))).rolling(10).mean(), 0)
        # Compare recent vs slightly longer term volatility
        # recent_vol = ta.STDDEV(close, timeperiod=5)
        # context_vol = ta.STDDEV(close, timeperiod=20)
        # df['volatility_ratio'] = (recent_vol / context_vol).fillna(1)
        df['vol_regime_5m'] = (df['volatility_5m'] > ta.EMA(df['volatility_5m'], timeperiod=5)).astype(int)
        df['vol_regime_15m'] = (df['volatility_15m'] > ta.EMA(df['volatility_15m'], timeperiod=15)).astype(int)
        
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume features
        
        거래량 특성 추가
        """
        print('adding volume features...')
        volume = df['volume']
        close = df['close']
        vwap = df['vwap']
        
        # Normalized volume using shorter timeframe
        # log_volume = np.log(volume + 1)
        # volume_ma = ta.SMA(log_volume, timeperiod=20)  # 20min instead of 50min
        # volume_std = ta.STDDEV(log_volume, timeperiod=20)
        # df['volume_normalized'] = (log_volume - volume_ma) / volume_std
        
        # Volume moving averages - using EMA for intraday responsiveness
        df['volume_ma_5m'] = ta.EMA(volume, timeperiod=5)  
        df['volume_ma_15m'] = ta.EMA(volume, timeperiod=15)
        
        # Volume ratios
        df['volume_ratio_5m'] = volume / df['volume_ma_5m']
        df['volume_ratio_15m'] = volume / df['volume_ma_15m']
        
        # VWAP related
        df['price_vwap_distance'] = ((close - vwap) / vwap)
        df['price_vwap_abs_distance'] = (np.abs(close - vwap) / vwap)
        # df['price_vwap_std_multiple_5m'] = ((close - vwap) / ta.STDDEV(close, timeperiod=5)).fillna(0)
        # df['price_vwap_std_multiple_15m'] = ((close - vwap) / ta.STDDEV(close, timeperiod=15)).fillna(0)
        
        # Trading intensity - calculate actual time-based volume ratios
        df = self._calculate_session_volume_ratios(df)
        
        # Volume on up/down minutes
        up_minutes = ta.ROC(close, timeperiod=1) > 0
        down_minutes = ta.ROC(close, timeperiod=1) < 0
        volume_up = volume * up_minutes.astype(int)
        volume_down = volume * down_minutes.astype(int)
        up_minutes_count_5 = up_minutes.rolling(5).sum()
        down_minutes_count_5 = down_minutes.rolling(5).sum()
        up_minutes_count_15 = up_minutes.rolling(15).sum()
        down_minutes_count_15 = down_minutes.rolling(15).sum()
        
        # Average volume per up/down minutes
        df['avg_volume_up_5m'] = volume_up.rolling(5).sum() / up_minutes_count_5.replace(0, 1)
        df['avg_volume_down_5m'] = volume_down.rolling(5).sum() / down_minutes_count_5.replace(0, 1)
        df['avg_volume_up_15m'] = volume_up.rolling(15).sum() / up_minutes_count_15.replace(0, 1)
        df['avg_volume_down_15m'] = volume_down.rolling(15).sum() / down_minutes_count_15.replace(0, 1)
        
        # Volume spike frequency - multi-timeframe approach
        volume_ma_5 = ta.SMA(volume, timeperiod=5)
        volume_std_5 = ta.STDDEV(volume, timeperiod=5)
        volume_threshold_5 = volume_ma_5 + 2 * volume_std_5
        
        volume_ma_15 = ta.SMA(volume, timeperiod=15)  
        volume_std_15 = ta.STDDEV(volume, timeperiod=15)
        volume_threshold_15 = volume_ma_15 + 2 * volume_std_15
        
        # Combined spike intensity (more informative than separate frequencies)
        spike_5m = (volume > volume_threshold_5).astype(int)
        spike_15m = (volume > volume_threshold_15).astype(int)
        
        # Volume spike persistence: how sustained are the spikes?
        df['volume_spike_persistence'] = (spike_5m + spike_15m).rolling(5).mean()  # 0=no spikes, 1=sustained spikes across timeframes
        
        # Volume-price correlation
        price_roc = ta.ROC(close, timeperiod=1)
        try:
            df['volume_price_corr_5m'] = ta.CORREL(volume, price_roc, timeperiod=5)
            df['volume_price_corr_15m'] = ta.CORREL(volume, price_roc, timeperiod=15)
        except:
            # Fallback to pandas correlation
            df['volume_price_corr_5m'] = volume.rolling(5).corr(price_roc)
            df['volume_price_corr_15m'] = volume.rolling(15).corr(price_roc)
        
        volume_ma_vol_5 = ta.SMA(volume, timeperiod=5)
        volume_ma_vol_15 = ta.SMA(volume, timeperiod=15)
        volume_std_5 = ta.STDDEV(volume, timeperiod=5)
        volume_std_15 = ta.STDDEV(volume, timeperiod=15)
        df['volume_volatility_5m'] = np.where(volume_ma_vol_5 > 0, 
                                           volume_std_5 / volume_ma_vol_5, 
                                           0)  # Default to 0 when no volume
        df['volume_volatility_15m'] = np.where(volume_ma_vol_15 > 0, 
                                           volume_std_15 / volume_ma_vol_15, 
                                           0)  # Default to 0 when no volume
        
        # Calculate how much current volume correlates with previous volume over 5-period windows
        volume_curr_5 = volume
        volume_prev_5 = volume.shift(1)
        df['volume_persistence_5m'] = volume_curr_5.rolling(5).corr(volume_prev_5).fillna(0)
        volume_curr_15 = volume
        volume_prev_15 = volume.shift(1)
        df['volume_persistence_15m'] = volume_curr_15.rolling(15).corr(volume_prev_15).fillna(0)
        
        return df

    def _calculate_session_volume_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate actual volume ratios for different trading sessions based on timestamps
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert to Eastern Time for session calculation (market hours are in ET)
        df_temp = df.copy()
        # Handle both tz-aware and tz-naive timestamps
        if df_temp['timestamp'].dt.tz is None:
            df_temp['timestamp_et'] = df_temp['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            df_temp['timestamp_et'] = df_temp['timestamp'].dt.tz_convert('US/Eastern')
        df_temp['date'] = df_temp['timestamp_et'].dt.date
        df_temp['hour'] = df_temp['timestamp_et'].dt.hour
        df_temp['minute'] = df_temp['timestamp_et'].dt.minute
        df_temp['time_minutes'] = df_temp['hour'] * 60 + df_temp['minute']
        
        # Define session boundaries (in minutes from midnight)
        # Morning: 9:30 AM - 12:00 PM (570 - 720 minutes)
        # Lunch: 12:00 PM - 2:00 PM (720 - 840 minutes)  
        # Afternoon: 2:00 PM - 4:00 PM (840 - 960 minutes)
        morning_start = 9 * 60 + 30  # 9:30 AM = 570 minutes
        morning_end = 12 * 60        # 12:00 PM = 720 minutes
        lunch_start = 12 * 60        # 12:00 PM = 720 minutes
        lunch_end = 14 * 60          # 2:00 PM = 840 minutes
        afternoon_start = 14 * 60    # 2:00 PM = 840 minutes
        afternoon_end = 16 * 60      # 4:00 PM = 960 minutes
        
        # Calculate daily volume for each session
        def calculate_daily_session_volumes(group):
            # Filter to regular market hours only
            market_hours = group[
                (group['time_minutes'] >= morning_start) & 
                (group['time_minutes'] < afternoon_end)
            ].copy()
            
            if len(market_hours) == 0:
                return pd.Series({
                    'morning_volume': 0,
                    'lunch_volume': 0, 
                    'afternoon_volume': 0,
                    'total_daily_volume': 0
                })
            
            morning_vol = market_hours[
                (market_hours['time_minutes'] >= morning_start) & 
                (market_hours['time_minutes'] < morning_end)
            ]['volume'].sum()
            
            lunch_vol = market_hours[
                (market_hours['time_minutes'] >= lunch_start) & 
                (market_hours['time_minutes'] < lunch_end)
            ]['volume'].sum()
            
            afternoon_vol = market_hours[
                (market_hours['time_minutes'] >= afternoon_start) & 
                (market_hours['time_minutes'] < afternoon_end)
            ]['volume'].sum()
            
            total_vol = morning_vol + lunch_vol + afternoon_vol
            
            return pd.Series({
                'morning_volume': morning_vol,
                'lunch_volume': lunch_vol,
                'afternoon_volume': afternoon_vol, 
                'total_daily_volume': total_vol
            })
        
        # Group by date and calculate session volumes
        daily_volumes = df_temp.groupby('date').apply(calculate_daily_session_volumes, include_groups=False)
        
        # Merge back to original dataframe
        df_temp = df_temp.merge(daily_volumes, left_on='date', right_index=True, how='left')
        
        # Calculate ratios (avoid division by zero)
        df['morning_volume_ratio'] = np.where(
            df_temp['total_daily_volume'] > 0,
            df_temp['morning_volume'] / df_temp['total_daily_volume'],
            0.0
        )
        
        df['lunch_volume_ratio'] = np.where(
            df_temp['total_daily_volume'] > 0,
            df_temp['lunch_volume'] / df_temp['total_daily_volume'],
            0.0
        )
        
        df['afternoon_volume_ratio'] = np.where(
            df_temp['total_daily_volume'] > 0,
            df_temp['afternoon_volume'] / df_temp['total_daily_volume'],
            0.0
        )
        
        # Calculate which session each minute belongs to
        df_temp['session'] = np.where(
            (df_temp['time_minutes'] >= morning_start) & (df_temp['time_minutes'] < morning_end), 'morning',
            np.where(
                (df_temp['time_minutes'] >= lunch_start) & (df_temp['time_minutes'] < lunch_end), 'lunch',
                np.where(
                    (df_temp['time_minutes'] >= afternoon_start) & (df_temp['time_minutes'] < afternoon_end), 'afternoon',
                    'pre_post'
                )
            )
        )
        
        # Calculate volume relative to typical session volume
        df['session_volume_relative'] = np.where(
            (df_temp['session'] == 'morning') & (df['morning_volume_ratio'] > 0),
            df['volume'] / (df_temp['morning_volume'] / 150),  # 150 minutes in morning session
            np.where(
                (df_temp['session'] == 'lunch') & (df['lunch_volume_ratio'] > 0),
                df['volume'] / (df_temp['lunch_volume'] / 120),  # 120 minutes in lunch session
                np.where(
                    (df_temp['session'] == 'afternoon') & (df['afternoon_volume_ratio'] > 0),
                    df['volume'] / (df_temp['afternoon_volume'] / 120),  # 120 minutes in afternoon
                    1.0  # Default for pre/post market
                )
            )
        )
        
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators
        
        기술적 지표 추가
        """
        
        def compute_pvi_nvi_vectorized(close, volume, start=1000):
            pct_change = close.pct_change().fillna(0)

            # Create masks for when volume increases or decreases
            vol_up = volume > volume.shift(1)
            vol_down = volume < volume.shift(1)

            # Initialize arrays
            pvi = np.ones(len(close)) * start
            nvi = np.ones(len(close)) * start

            # Apply daily multipliers where conditions are met
            pvi[1:] = np.where(vol_up[1:], 1 + pct_change[1:], 1)
            nvi[1:] = np.where(vol_down[1:], 1 + pct_change[1:], 1)

            # Compute cumulative product
            pvi = pd.Series(np.cumprod(pvi), index=close.index)
            nvi = pd.Series(np.cumprod(nvi), index=close.index)

            return pvi, nvi

        print('adding technical indicators (65-112)...')
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Trend indicators
        df['rsi_7m'] = ta.RSI(close, timeperiod=7)  
        df['rsi_14m'] = ta.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = ta.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['ema_5'] = ta.EMA(close, timeperiod=5)
        df['ema_15'] = ta.EMA(close, timeperiod=15)
        df['adx'] = ta.ADX(high, low, close)
        df['plus_di'] = ta.PLUS_DI(high, low, close)
        df['minus_di'] = ta.MINUS_DI(high, low, close)
        aroon_down, aroon_up = ta.AROON(high, low)
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        df['cci'] = ta.CCI(high, low, close)
        
        # Momentum indicators
        slowk, slowd = ta.STOCH(high, low, close)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['williams_r'] = ta.WILLR(high, low, close)
        df['roc_5m'] = ta.ROC(close, timeperiod=5)
        df['roc_10m'] = ta.ROC(close)
        diff = close.diff()
        r = 13
        s = 7
        double_smoothed_diff = ta.EMA(ta.EMA(diff, r), s)
        double_smoothed_abs_diff = ta.EMA(ta.EMA(diff.abs(), r), s)
        df['tsi'] = 100 * np.where(double_smoothed_abs_diff != 0, 
                                   double_smoothed_diff / double_smoothed_abs_diff, 0)
        df['ultosc'] = ta.ULTOSC(high, low, close)
        df['mfi'] = ta.MFI(high, low, close, volume)
        df['cmo'] = ta.CMO(close)
        df['kama'] = ta.KAMA(close, timeperiod=20)
        
        # Volatility indicators
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_percent_b'] = np.where((bb_upper - bb_lower) > 0, (close - bb_lower) / (bb_upper - bb_lower), 0.5)
        df['bb_width'] = np.where(bb_middle > 0, (bb_upper - bb_lower) / bb_middle, 0.0)
        
        # Keltner Channels
        ema_20 = ta.EMA(close, timeperiod=20)
        atr_10 = ta.ATR(high, low, close, timeperiod=10)
        df['kc_upper'] = ema_20 + 2 * atr_10
        df['kc_lower'] = ema_20 - 2 * atr_10
        
        # Donchian Channels
        df['dc_upper_5m'] = high.rolling(5).max()
        df['dc_lower_5m'] = low.rolling(5).min()
        df['dc_upper_15m'] = high.rolling(15).max()
        df['dc_lower_15m'] = low.rolling(15).min()
        
        # VIX-style volatility
        df['intraday_vol'] = ta.STDDEV(ta.ROC(close, timeperiod=1), timeperiod=20) * np.sqrt(MINUTES_PER_TRADING_DAY)
        
        # Volume indicators
        df['obv'] = ta.OBV(close, volume)
        df['ad_line'] = ta.AD(high, low, close, volume)
        df['cmf'] = ta.ADOSC(high, low, close, volume)
        df['vroc_5m'] = ta.ROC(volume, timeperiod=5)
        df['vroc_15m'] = ta.ROC(volume, timeperiod=15)
        
        # Volume oscillator
        volume_fast = ta.EMA(volume, timeperiod=5) 
        volume_slow = ta.EMA(volume, timeperiod=15)
        df['volume_oscillator'] = np.where(volume_slow > 0, 
                                          (volume_fast - volume_slow) / volume_slow, 0)
        
        # Additional volume indicators
        df['pvi'], df['nvi'] = compute_pvi_nvi_vectorized(close, volume)
        
        # Force Index
        df['force_index'] = volume * ta.ROC(close, timeperiod=1)
        
        # ENHANCEMENT: Combined signal strength features
        # Multi-timeframe signal convergence
        rsi_5m = ta.RSI(close, timeperiod=5)
        rsi_15m = ta.RSI(close, timeperiod=15)
        macd_5m, macd_signal_5m, _ = ta.MACD(close, fastperiod=5, slowperiod=12, signalperiod=4)
        
        # Signal convergence strength (when multiple indicators agree)
        bullish_rsi_5m = (rsi_5m > 50).astype(int)
        bullish_rsi_15m = (rsi_15m > 50).astype(int)
        bullish_macd = (macd_5m > macd_signal_5m).astype(int)
        bullish_momentum = (df['momentum_gradient_5m'] > 0).astype(int) if 'momentum_gradient_5m' in df.columns else 0
        
        # Combined signal strength (0-1, higher = more indicators agree)
        df['bullish_signal_strength'] = (bullish_rsi_5m + bullish_rsi_15m + bullish_macd + bullish_momentum) / 4
        df['bearish_signal_strength'] = 1 - df['bullish_signal_strength']
        
        # Signal persistence (how long has signal been consistent)
        df['signal_persistence_bullish'] = (df['bullish_signal_strength'] > 0.6).astype(int).rolling(5).sum() / 5
        df['signal_persistence_bearish'] = (df['bearish_signal_strength'] > 0.6).astype(int).rolling(5).sum() / 5
        
        return df

    def _add_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal patterns
        
        시간 패턴 추가
        """
        print('adding temporal patterns...')
        timestamp = df['timestamp']
        
        # Convert to different time components
        minute = timestamp.dt.minute
        hour = timestamp.dt.hour
        day = timestamp.dt.dayofweek  # 0=Monday, 6=Sunday
        month = timestamp.dt.month
        quarter = timestamp.dt.quarter
        
        # Cyclical encoding
        df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
        
        return df

    def _add_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market context features
        
        시장 컨텍스트 추가
        """
        print('adding market context...')
        
        # Market state
        volume_ma_5 = ta.SMA(df['volume'], timeperiod=5)
        volume_ma_15 = ta.SMA(df['volume'], timeperiod=15)
        df['high_volume_regime_5m'] = (df['volume'] > volume_ma_5).astype(int)
        df['high_volume_regime_15m'] = (df['volume'] > volume_ma_15).astype(int)
        
        return df

    def _add_portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add portfolio state features
        
        포트폴리오 상태 추가
        """
        print('adding portfolio features...')
        
        returns_roc = ta.ROC(df['close'], timeperiod=1)
        df['var_95'] = returns_roc.rolling(20).quantile(0.05)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional derived features and ratios
        """
        print('adding derived features...')
        
        # Price ratios relative to moving averages
        close = df['close']
        df['price_to_ema_5'] = close / df['ema_5'] - 1
        df['price_to_ema_15'] = close / df['ema_15'] - 1
        
        # Technical indicator combinations
        df['rsi_macd_combo'] = (df['rsi_14m'] / 100 + 
                               np.where(df['close'] > 0, df['macd'] / df['close'], 0)).fillna(0)
        df['bb_rsi_combo'] = (df['bb_percent_b'] * df['rsi_14m'] / 100).fillna(0)
        
        # Volume-price relationships
        df['volume_price_trend_5m'] = df['volume_ratio_5m'] * np.sign(ta.ROC(close, timeperiod=1))
        df['volume_price_trend_15m'] = df['volume_ratio_15m'] * np.sign(ta.ROC(close, timeperiod=1))
        
        # Volatility regimes
        vol_median_5 = df['volatility_5m'].rolling(5).median() 
        vol_median_15 = df['volatility_15m'].rolling(15).median() 

        df['vol_regime_numeric_5m'] = np.where(vol_median_5 > 0, 
                                               df['volatility_5m'] / vol_median_5, 
                                               1.0) 
        df['vol_regime_numeric_15m'] = np.where(vol_median_15 > 0, 
                                                df['volatility_15m'] / vol_median_15, 
                                                1.0) 
        
        # Additional price-based features
        high = df['high']
        low = df['low']
        open_price = df['open']
        df['high_low_ratio'] = np.where(low > 0, high / low, 1.0)
        df['close_open_ratio'] = np.where(open_price > 0, close / open_price, 1.0)
        df['price_range_normalized'] = np.where(close > 0, (high - low) / close, 0.0)
        
        # Additional derived features
        df['price_to_ema_15_alt'] = close / ta.EMA(close, timeperiod=15) - 1
        
        # Handle division by zero when Bollinger Bands have zero width
        bb_width = df['bb_upper'] - df['bb_lower']
        df['price_position_bb'] = np.where(bb_width > 0,
                                          (close - df['bb_lower']) / bb_width,
                                          0.5)  # Default to 0.5 (middle position) when bands have no width
        
        df['atr_ratio_5m'] = df['atr_5m'] / close 
        df['atr_ratio_15m'] = df['atr_15m'] / close 
        
        # Momentum persistence ratio (bullish vs bearish strength)
        df['momentum_persistence_ratio_5m'] = np.where(
            df['momentum_persistence_bearish_5m'] > 0,
            df['momentum_persistence_bullish_5m'] / df['momentum_persistence_bearish_5m'],
            df['momentum_persistence_bullish_5m'] * 2  # Default when no bearish momentum
        )
        df['momentum_persistence_ratio_15m'] = np.where(
            df['momentum_persistence_bearish_15m'] > 0,
            df['momentum_persistence_bullish_15m'] / df['momentum_persistence_bearish_15m'],
            df['momentum_persistence_bullish_15m'] * 2  # Default when no bearish momentum
        )

        # Momentum consistency (how directional vs choppy the market is)
        df['momentum_consistency_5m'] = df['momentum_persistence_bullish_5m'] + df['momentum_persistence_bearish_5m']
        df['momentum_consistency_15m'] = df['momentum_persistence_bullish_15m'] + df['momentum_persistence_bearish_15m']
        # High values = strong directional movement, Low values = choppy/sideways

        # Momentum shift indicator (detect momentum changes)
        df['momentum_shift_5m'] = (df['momentum_persistence_bullish_5m'] - df['momentum_persistence_bearish_5m']).diff()
        df['momentum_shift_15m'] = (df['momentum_persistence_bullish_15m'] - df['momentum_persistence_bearish_15m']).diff()
        # Positive = shifting toward bullish, Negative = shifting toward bearish
        
        # ENHANCEMENT: Peak/Valley detection for support/resistance
        smoothed_price = signal.savgol_filter(close, 15, 3)
        
        # Detect significant peaks and valleys
        peaks, _ = signal.find_peaks(smoothed_price, prominence=close.std() * 0.5, distance=10)
        valleys, _ = signal.find_peaks(-smoothed_price, prominence=close.std() * 0.5, distance=10)
        
        # Distance to nearest significant levels
        df['distance_to_resistance'] = self._calculate_distance_to_levels(close.index, peaks, close.values)
        df['distance_to_support'] = self._calculate_distance_to_levels(close.index, valleys, close.values)
        
        # Proximity indicators (0-1, higher = closer to level)
        df['resistance_proximity'] = np.exp(-df['distance_to_resistance'] / 10)  # Exponential decay
        df['support_proximity'] = np.exp(-df['distance_to_support'] / 10)
        
        return df

    def _calculate_distance_to_levels(self, index, level_indices, prices):
        """Calculate distance to nearest significant price level"""
        distances = np.full(len(index), np.inf)
        
        for i, idx in enumerate(index):
            if len(level_indices) > 0:
                # Find nearest level in time
                nearest_level_idx = level_indices[np.argmin(np.abs(level_indices - i))]
                # Calculate price distance to that level
                level_price = prices[nearest_level_idx] if nearest_level_idx < len(prices) else prices[-1]
                distances[i] = abs(prices[i] - level_price) / level_price if level_price > 0 else 0
        
        return distances

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features to [-1, 1] range where appropriate
        특성을 적절한 경우 [-1, 1] 범위로 정규화
        """
        print('normalizing features...')
        
        # Features that should be normalized
        normalize_features = [
            'rsi_7m', 'rsi_14m', 'stoch_k', 'stoch_d', 'williams_r',
            'bb_percent_b', 'cci', 'mfi'
        ]
        
        for feature in normalize_features:
            if feature in df.columns:
                if feature.startswith('rsi') or feature.startswith('stoch') or feature in ['mfi']:
                    # Scale from [0, 100] to [-1, 1]
                    df[feature] = (df[feature] - 50) / 50
                elif feature == 'williams_r':
                    # Scale from [-100, 0] to [-1, 1]
                    df[feature] = df[feature] / 50
                elif feature == 'bb_percent_b':
                    # Scale from [0, 1] to [-1, 1]
                    df[feature] = df[feature] * 2 - 1
                elif feature == 'cci':
                    # Clip and normalize CCI
                    df[feature] = df[feature].clip(-200, 200) / 200
        
        return df

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle infinite and NaN values comprehensively
        무한값과 NaN 값을 포괄적으로 처리
        """
        print('handling infinite and NaN values...')
        
        # Get numeric columns only (exclude timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with NaN first
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Count and report infinite/NaN values
        inf_counts = df[numeric_cols].isnull().sum()
        problematic_cols = inf_counts[inf_counts > 0]
        
        if len(problematic_cols) > 0:
            print(f"Found NaN/inf values in {len(problematic_cols)} columns:")
            for col, count in problematic_cols.items():
                print(f"  {col}: {count} values")
        
        # Handle NaN values with appropriate strategies
        for col in numeric_cols:
            if df[col].isnull().any():
                # For ratio features, use 1.0 (neutral ratio)
                if any(keyword in col.lower() for keyword in ['ratio', '_to_', 'normalized']):
                    df[col] = df[col].fillna(1.0)
                # For percentage features, use 0.0 (neutral percentage)
                elif any(keyword in col.lower() for keyword in ['percent', 'pct', '_b']):
                    df[col] = df[col].fillna(0.0)
                # For oscillators and indicators, use median
                elif any(keyword in col.lower() for keyword in ['rsi', 'stoch', 'williams', 'cci', 'mfi', 'macd']):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                # For volume features, use 0.0
                elif 'volume' in col.lower():
                    df[col] = df[col].fillna(0.0)
                # For volatility features, use median
                elif any(keyword in col.lower() for keyword in ['vol', 'atr', 'std']):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                # For return features, use 0.0 (no return)
                elif 'return' in col.lower() or 'roc' in col.lower():
                    df[col] = df[col].fillna(0.0)
                # For momentum features, use 0.0 (no momentum)
                elif 'momentum' in col.lower():
                    df[col] = df[col].fillna(0.0)
                # For all other features, use forward fill then backward fill, then 0
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        # Final check for any remaining NaN/inf values
        remaining_nans = df[numeric_cols].isnull().sum().sum()
        remaining_infs = np.isinf(df[numeric_cols]).sum().sum()
        
        if remaining_nans > 0 or remaining_infs > 0:
            print(f"WARNING: {remaining_nans} NaN and {remaining_infs} infinite values still remain")
            # Force replace any remaining problematic values
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf, np.nan], 0.0)
        else:
            print("✅ All infinite and NaN values successfully handled")
        
        return df

    def _drop_rows_before_timestamp(self, df: pd.DataFrame, timestamp: str) -> pd.DataFrame:
        """
        Drop rows before a specified timestamp
        지정된 타임스탬프 이전의 행들 삭제
        """
        print('dropping rows before timestamp...')
        cutoff_timestamp = pd.to_datetime(timestamp).tz_localize('UTC')
        return df[df['timestamp'] >= cutoff_timestamp]

    def _save_feature_engineer_ticker(self, ticker: str, timestamp: str):
        """
        Feature engineers data for a single ticker with comprehensive features
        Uses only data available from OHLCV database (subset of 156-dimensional features)
        """
        print(f'starting comprehensive feature engineering for {ticker}')
        
        # Get and prepare data
        df = self._get_data(ticker)
        df = self._handle_gaps(df)
        
        # Add feature categories (only what can be calculated with OHLCV data)
        df = self._add_price_features(df)           # Price-based features
        df = self._add_volume_features(df)          # Volume-based features  
        df = self._add_technical_indicators(df)     # Technical indicators
        df = self._add_temporal_patterns(df)        # Time-based features
        df = self._add_market_context(df)           # Basic market context (limited)
        df = self._add_portfolio_features(df)       # Basic risk metrics only
        df = self._add_derived_features(df)         # Additional derived features
        # df = self._normalize_features(df)           # Essential for DQN training
        # df = self._handle_infinite_values(df)       # Handle infinite values
        df = self._drop_rows_before_timestamp(df, timestamp)
        
        # Save results
        base_dir = DATA_DIR / 'feature_engineered_v2'
        create_directory(base_dir)
        save_to_csv(df, f'{base_dir}/{ticker}.csv', index=False)

        print(f'completed feature engineering for {ticker}')
        print(f'final dataset shape: {df.shape}')
        print(f'features created: {df.shape[1] - 1} (excluding timestamp)')

    def save_feature_engineered_tickers(self, tickers: List[str] = TICKERS):
        """
        Feature engineers data for multiple tickers with comprehensive features
        Creates subset of features using available OHLCV data
        """    
        for ticker in tickers:
            try:
                self._save_feature_engineer_ticker(ticker, self.timestamp)
                print(f"✅ Successfully processed {ticker}")
            except Exception as e:
                print(f"❌ Failed to process {ticker}: {e}")
                continue
        
        print("="*80)
        print("INTRADAY FEATURE ENGINEERING COMPLETE")
        print("="*80)

if __name__ == '__main__':
    feature_engineer = FeatureEngineer(use_json=True)
    feature_engineer.save_feature_engineered_tickers(['TSLA'])