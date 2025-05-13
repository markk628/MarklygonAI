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
TICKERS = ['APPL', 'MSFT', 'NVDA', 'JPM', 'BAC', 'GS', 'JNJ', 'PFE', 
           'UNH', 'AMZN', 'TSLA', 'MCD', 'GOOG', 'META', 'NFLX', 'PG', 
           'KO', 'WMT', 'XOM', 'CVX', 'COP', 'NEE', 'DUK', 'SO', 
           'LIN', 'FCX', 'NEM', 'CAT', 'UPS', 'HON', 'AMT', 'PLD', 
           'PLTR', 'VST', 'MRNA', 'WBA', 'WSM', 'ALB', 'APA', 'KKR']

# 데이터 수집 설정
DATA_START_DATE = '1430899200000' # 2015-05-06 4:00 AM UTC
DATA_FREQUENCY = "minute"  # 일별 데이터

# 데이터 전처리 설정
WINDOW_SIZE = 30  # 관측 윈도우 크기
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
REPLAY_BUFFER_SIZE = 1000000

# 학습 설정
BATCH_SIZE = 256
NUM_EPISODES = 1000
EVALUATE_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50
MAX_STEPS_PER_EPISODE = 1000

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUTOFF_TIMESTAMP = '2021-05-06 08:00:00'

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