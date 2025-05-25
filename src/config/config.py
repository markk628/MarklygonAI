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
WINDOW_SIZE = 60  # 관측 윈도우 크기
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 트레이딩 환경 설정
INITIAL_BALANCE = 10000.0  # 초기 자본금
MAX_TRADING_UNITS = 10  # 최대 거래 단위
TRANSACTION_FEE_PERCENT = 0.001  # 거래 수수료 (0.1%)

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


REPLAY_BUFFER_SIZE = 350000

# 학습 설정
BATCH_SIZE = 1024
NUM_EPISODES = 100
EVALUATE_INTERVAL = 5
TRAIN_INTERVAL = 4

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUTOFF_TIMESTAMP = '2021-05-06 08:00:00'

PRICE_FEATURES = ['open', 'high', 'low', 'close', 'vwap']
VOLUME_FEATURES = ['transactions', 'volume']
MOMENTUM_FEATURES = ['stochrsi_k_14_1min', 'stochrsi_d_14_1min', 
                     'stochrsi_k_21_1min', 'stochrsi_d_21_1min', 
                     'rsi_7_1min', 'rsi_14_1min', 'rsi_21_1min', 
                     'macd_5_13_4_1min', 'macd_signal_5_13_4_1min', 'macd_hist_5_13_4_1min', 
                     'macd_12_26_9_1min', 'macd_signal_12_26_9_1min', 'macd_hist_12_26_9_1min', 
                     'roc_5_1min', 'roc_10_1min', 'roc_20_1min', 
                     'ultosc_5_15_30_1min', 'ultosc_7_21_42_1min', 'ultosc_10_30_60_1min', 'obv_1min']
TREND_FEATURES = ['ema_3_1min', 'ema_5_1min', 'ema_9_1min', 'ema_21_1min', 'ema_50_1min', 
                  'sma_5_1min', 'sma_10_1min', 'sma_20_1min', 'sma_50_1min', 
                  'plusdi_10_1min', 'plusdi_20_1min', 'plusdi_30_1min', 
                  'minusdi_10_1min', 'minusdi_20_1min', 'minusdi_30_1min', 
                  'adx_10_1min', 'adx_20_1min', 'adx_30_1min']
VOLATILLITY_FEATURES = ['bband_upper_10_1min', 'bband_middle_10_1min', 'bband_lower_10_1min', 
                        'bband_upper_20_1min', 'bband_middle_20_1min', 'bband_lower_20_1min', 
                        'bband_upper_50_1min', 'bband_middle_50_1min', 'bband_lower_50_1min', 
                        'atr_14_1min', 'atr_21_1min', 
                        'close_rolling_std_5', 'close_rolling_std_15', 'close_rolling_std_30',
                        'volume_rolling_std_5', 'volume_rolling_std_15', 'volume_rolling_std_30', 
                        'log_return_rolling_std_15', 'log_return_rolling_std_30']
OCILLATOR_FEATURES = ['cci_10_1min', 'cci_20_1min', 'cci_30_1min', 
                      'mfi_7_1min', 'mfi_14_1min', 'mfi_21_1min']
LAGGED_FEATURES = ['open_lag_1', 'high_lag_1', 'low_lag_1', 'close_lag_1', 'volume_lag_1', 'vwap_lag_1', 
                   'open_lag_5', 'high_lag_5', 'low_lag_5', 'close_lag_5', 'volume_lag_5', 'vwap_lag_5', 
                   'open_lag_10', 'high_lag_10', 'low_lag_10', 'close_lag_10', 'volume_lag_10', 'vwap_lag_10']
ROLLING_FEAATURES = ['close_rolling_mean_5', 'volume_rolling_mean_5', 
                     'close_rolling_mean_15', 'volume_rolling_mean_15', 
                     'close_rolling_mean_30', 'volume_rolling_mean_30']
PRICE_RANGE_FEATURES = ['close_diff_1', 'close_pct_change_1', 'log_return_1', 'high_low_range', 'close_open_range', 'high_low_ratio', 'close_open_ratio']
TEMPORAL_FEATURES = ['minute', 'minute_sin', 'minute_cos', 
                     'hour', 'hour_sin', 'hour_cos', 
                     'day', 'day_sin', 'day_cos', 
                     'month', 'month_sin', 'month_cos', 
                     'quarter', 'quarter_sin', 'quarter_cos',
                     'time_since_last_significant_change']
FEATURES_RANKED = ['close_pct_change_1', 'log_return_1', 'close_diff_1', 'stochrsi_k_14_1min', 'stochrsi_k_21_1min', 
                   'close_open_ratio', 'close_open_range', 'cci_10_1min', 'obv_1min', 'rsi_7_1min', 'macd_hist_5_13_4_1min', 
                   'cci_20_1min', 'roc_5_1min', 'rsi_14_1min', 'stochrsi_d_14_1min', 'stochrsi_d_21_1min', 'cci_30_1min', 
                   'rsi_21_1min', 'ultosc_5_15_30_1min', 'roc_10_1min', 'mfi_7_1min', 'minusdi_10_1min', 'plusdi_10_1min', 
                   'ultosc_7_21_42_1min', 'roc_20_1min', 'volume', 'transactions', 'mfi_14_1min', 'ultosc_10_30_60_1min', 
                   'minusdi_20_1min', 'plusdi_20_1min', 'macd_5_13_4_1min', 'mfi_21_1min', 'macd_hist_12_26_9_1min', 
                   'minusdi_30_1min', 'plusdi_30_1min', 'close_rolling_std_5', 'volume_rolling_mean_5', 'volume_rolling_std_5', 
                   'volume_rolling_mean_15', 'volume_rolling_std_15', 'volume_rolling_mean_30', 'volume_rolling_std_30', 
                   'high_low_ratio', 'volume_lag_1', 'atr_14_1min', 'atr_21_1min', 'log_return_rolling_std_15', 'volume_lag_5', 
                   'close_rolling_std_15', 'volume_lag_10', 'log_return_rolling_std_30', 'macd_signal_5_13_4_1min', 'vwap', 
                   'high_low_range', 'vwap_lag_1', 'macd_12_26_9_1min', 'close_rolling_std_30', 'hour', 'close_rolling_mean_5', 
                   'sma_5_1min', 'vwap_lag_5', 'vwap_lag_10', 'close', 'close_lag_1', 'ema_3_1min', 'bband_middle_10_1min', 
                   'sma_10_1min', 'open', 'open_lag_1', 'close_lag_5', 'bband_lower_10_1min', 'close_lag_10', 'bband_upper_10_1min', 
                   'hour_sin', 'high', 'macd_signal_12_26_9_1min', 'low', 'open_lag_5', 'high_lag_1', 'low_lag_1', 'open_lag_10', 
                   'close_rolling_mean_15', 'bband_middle_20_1min', 'sma_20_1min', 'time_since_last_significant_change', 
                   'ema_5_1min', 'high_lag_5', 'bband_upper_20_1min', 'low_lag_5', 'bband_lower_20_1min', 'high_lag_10', 
                   'low_lag_10', 'close_rolling_mean_30', 'ema_9_1min', 'bband_lower_50_1min', 'bband_upper_50_1min', 
                   'sma_50_1min', 'bband_middle_50_1min', 'ema_21_1min', 'ema_50_1min', 'adx_10_1min', 'hour_cos', 'adx_20_1min', 
                   'adx_30_1min', 'minute', 'month_cos', 'month_sin', 'month', 'quarter', 'quarter_sin', 'day', 'day_cos', 
                   'minute_sin', 'day_sin', 'minute_cos', 'quarter_cos']

DATABASE_URI = f'postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DATABASE}'

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