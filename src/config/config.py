"""
SAC 트레이딩 시스템 설정 파일
"""
import os
import logging
import torch
from datetime import datetime
from pathlib import Path

from src.config.apikeys import POLYGON_APIKEY
from src.config.database_values import *
from src.config.flask import *

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 데이터 관련 설정
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# API 설정
API_CALL_DELAY = 12  # 초 단위 (API 호출 제한 고려)

# 대상 주식 종목
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'BAC', 'GS', 'JNJ', 'PFE', 
           'UNH', 'AMZN', 'TSLA', 'MCD', 'GOOG', 'META', 'NFLX', 'PG', 
           'KO', 'WMT', 'XOM', 'CVX', 'COP', 'NEE', 'DUK', 'SO', 
           'LIN', 'FCX', 'NEM', 'CAT', 'UPS', 'HON', 'AMT', 'PLD', 
           'PLTR', 'VST', 'MRNA', 'WBA', 'WSM', 'ALB', 'APA', 'KKR']

# 데이터 수집 설정
DATA_START_DATE = '1430899200000' # 2015-05-06 4:00 AM UTC
DATA_FREQUENCY = "minute"  # 일별 데이터

# 데이터 전처리 설정
WINDOW_SIZE = 20  # 관측 윈도우 크기
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 트레이딩 환경 설정
INITIAL_BALANCE = 10000.0  # 초기 자본금
MAX_TRADING_UNITS = 10  # 최대 거래 단위
TRANSACTION_FEE_PERCENT = 0.001  # 거래 수수료 (0.1%)
MAX_POSITION_SIZE = 0.7

# SAC 모델 하이퍼파라미터
HIDDEN_DIM = 256
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
LEARNING_RATE_ALPHA = 3e-4
GAMMA = 0.99  # 할인 계수
TAU = 0.005  # 타겟 네트워크 소프트 업데이트 계수
ALPHA_INIT = 0.2  # 초기 엔트로피 계수
TARGET_UPDATE_INTERVAL = 1

# DQN 모델 하이퍼파라미터
REPLAY_BUFFER_SIZE = 300000
UPDATE_TARGET_EVERY = 388
EPSILON_EARLY_STOPPING_THRESHOLD = 0.1

# 학습 설정
BATCH_SIZE = 128
CUTOFF_TIMESTAMP = '2021-05-06 08:00:00'
NUM_EPISODES = 1000
EVALUATE_INTERVAL = 20
TRAIN_INTERVAL = 4

MINUTES_PER_TRADING_DAY = 390  # 9:30 AM to 4:00 PM EST (regular market)
MINUTES_PER_EXTENDED_DAY = 960  # 4:00 AM to 8:00 PM EST (full extended hours)
MINUTES_PER_ENHANCED_DAY = 450  # 8:30 AM to 4:00 PM EST (1hr pre + regular)

# 평가 설정
ANNUAL_RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'vwap']
AUXILIARY_FEATURES = [
    'stochrsi_k_14_1min', 'stochrsi_d_14_1min', 
    'rsi_14_1min', 
    'macd_12_26_9_1min', 'macd_signal_12_26_9_1min', 'macd_hist_12_26_9_1min',
    'roc_10_1min',
    'obv_1min',
    'ema_3_1min', 'ema_9_1min', 'ema_21_1min',
    'plusdi_20_1min', 'minusdi_20_1min', 'adx_20_1min',
    'bband_upper_20_1min', 'bband_lower_20_1min',
    'atr_14_1min', 'cci_20_1min', 'mfi_14_1min',
    'volume_rolling_std_15', 'log_return_rolling_std_15',
    'close_diff_1', 'log_return_1'
]
FILTERED_TEMPORAL_FEATURES = ['minute_sin', 'minute_cos', 
                              'hour_sin', 'hour_cos', 
                              'day_sin', 'day_cos', 
                              'month_sin', 'month_cos', 
                              'quarter_sin', 'quarter_cos']
STOCK_FEATURES = CORE_FEATURES + AUXILIARY_FEATURES + FILTERED_TEMPORAL_FEATURES

# dqn v2
PRICE_FEATURES = ['open', 'high', 'low', 'close', 'vwap']
VOLUME_FEATURES = ['volume']
TECHNICAL_FEATURES = [
    'stochrsi_k_14_1min', 'stochrsi_d_14_1min', 'rsi_14_1min',
    'macd_12_26_9_1min', 'macd_signal_12_26_9_1min', 'macd_hist_12_26_9_1min',
    'roc_10_1min', 'obv_1min', 'ema_3_1min', 'ema_9_1min', 'ema_21_1min',
    'plusdi_20_1min', 'minusdi_20_1min', 'adx_20_1min',
    'bband_upper_20_1min', 'bband_lower_20_1min',
    'atr_14_1min', 'cci_20_1min', 'mfi_14_1min'
]
VOLATILLITY_FEATURES = ['volume_rolling_std_15', 'log_return_rolling_std_15']
RETURNS_FEATURES = ['close_diff_1', 'log_return_1']
TEMPORAL_FEATURES = ['minute_sin', 'minute_cos', 
                     'hour_sin', 'hour_cos',
                     'day_sin', 'day_cos',
                     'month_sin', 'month_cos',
                     'quarter_sin', 'quarter_cos']

# feature engineering v2
STOCK_FEATURES_V2 = ['open','high','low','close','volume','vwap',
                     'return_1m','return_2m','return_3m','return_5m','return_7m','return_10m','return_12m','return_15m',
                     'open_gap','high_level','low_level','close_change',
                     'momentum_1m','momentum_5m','momentum_15m','momentum_1m_accel','momentum_5m_accel','momentum_15m_accel',
                     'momentum_persistence_bullish_1m','momentum_persistence_bearish_1m','momentum_persistence_bullish_5m','momentum_persistence_bearish_5m','momentum_persistence_bullish_15m','momentum_persistence_bearish_15m',
                     'price_smoothed_5m','price_smoothed_15m','price_trend_deviation_5m','price_trend_deviation_15m','momentum_gradient_5m',
                     'volatility_5m','volatility_15m',
                     'atr_5m','atr_15m',
                     'parkinson_vol',
                     'vol_regime_5m','vol_regime_15m','volume_ma_5m','volume_ma_15m','volume_ratio_5m','volume_ratio_15m',
                     'price_vwap_distance','price_vwap_abs_distance',
                     'morning_volume_ratio','lunch_volume_ratio','afternoon_volume_ratio','session_volume_relative',
                     'avg_volume_up_5m','avg_volume_down_5m','avg_volume_up_15m','avg_volume_down_15m',
                     'volume_spike_persistence','volume_price_corr_5m','volume_price_corr_15m','volume_volatility_5m','volume_volatility_15m','volume_persistence_5m','volume_persistence_15m',
                     'rsi_7m','rsi_14m',
                     'macd','macd_signal','macd_histogram',
                     'ema_5','ema_15',
                     'adx',
                     'plus_di','minus_di',
                     'aroon_up','aroon_down',
                     'cci',
                     'stoch_k','stoch_d',
                     'williams_r',
                     'roc_5m','roc_10m',
                     'tsi','ultosc','mfi','cmo','kama',
                     'bb_upper','bb_lower','bb_percent_b','bb_width',
                     'kc_upper','kc_lower',
                     'dc_upper_5m','dc_lower_5m','dc_upper_15m','dc_lower_15m',
                     'intraday_vol','obv','ad_line','cmf',
                     'vroc_5m','vroc_15m',
                     'volume_oscillator','pvi','nvi','force_index',
                     'bullish_signal_strength','bearish_signal_strength','signal_persistence_bullish','signal_persistence_bearish',
                     'minute_sin','minute_cos','hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos','quarter_sin','quarter_cos',
                     'high_volume_regime_5m','high_volume_regime_15m',
                     'var_95',
                     'price_to_ema_5','price_to_ema_15',
                     'rsi_macd_combo','bb_rsi_combo',
                     'volume_price_trend_5m','volume_price_trend_15m','vol_regime_numeric_5m','vol_regime_numeric_15m',
                     'high_low_ratio','close_open_ratio',
                     'price_range_normalized','price_to_ema_15_alt','price_position_bb',
                     'atr_ratio_5m','atr_ratio_15m',
                     'momentum_persistence_ratio_5m','momentum_persistence_ratio_15m','momentum_consistency_5m','momentum_consistency_15m','momentum_shift_5m','momentum_shift_15m',
                     'distance_to_resistance','distance_to_support','resistance_proximity','support_proximity']


WEB_DATABASE_URI = f'postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_WEB}'

# 로깅 설정
def setup_logger(name, log_file, level=logging.INFO):
    """로거 설정 함수"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

# 기본 로거 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"marklygon_{timestamp}.log"
LOGGER = setup_logger("marklygon", LOG_FILE)

# 백테스트 설정
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2025-01-01" 