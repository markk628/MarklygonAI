"""
Eddie의 Signal Pipeline + Pattern Pipeline 설정
TCN+Attention과 TimesNet+Wavelet 통합 시스템
"""
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Union


@dataclass
class TCNConfig:
    """TCN (Temporal Convolutional Network) 설정"""
    num_inputs: int = 50  # 입력 피처 수 (TA-Lib 지표들)
    num_channels: List[int] = None  # [64, 128, 256, 128, 64]
    kernel_size: int = 3
    dropout: float = 0.2
    activation: str = 'relu'
    use_norm: bool = True
    
    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [64, 128, 256, 128, 64]


@dataclass
class AttentionConfig:
    """Multi-Head Attention 설정"""
    d_model: int = 64  # TCN 출력 차원과 맞춰야 함
    num_heads: int = 8
    dropout: float = 0.1
    use_positional_encoding: bool = True
    max_seq_length: int = 60


@dataclass
class SignalGeneratorConfig:
    """Signal Generator (TCN+Attention) 통합 설정"""
    tcn: TCNConfig = None
    attention: AttentionConfig = None
    seq_len: int = 60  # 60분 윈도우
    output_dim: int = 1  # 매도 강도 [-1, 1]
    use_uncertainty: bool = True  # 모델 불확실성 추정
    
    def __post_init__(self):
        if self.tcn is None:
            self.tcn = TCNConfig()
        if self.attention is None:
            self.attention = AttentionConfig()


@dataclass
class TimesNetConfig:
    """TimesNet 설정"""
    seq_len: int = 60
    pred_len: int = 1
    top_k: int = 5  # 상위 K개 주기 선택
    d_model: int = 128
    d_ff: int = 256
    num_kernels: int = 6
    num_layers: int = 2  # Number of TimesNet layers
    dropout: float = 0.1
    use_multi_scale: bool = True


@dataclass
class WaveletConfig:
    """Wavelet Transform 설정"""
    wavelet: str = 'db4'  # Daubechies 4
    levels: int = 5
    noise_threshold: float = 0.1
    denoise_method: str = 'soft'  # 'soft' or 'hard'
    extract_energy: bool = True
    extract_entropy: bool = True


@dataclass
class PatternAnalyzerConfig:
    """Pattern Analyzer (TimesNet+Wavelet) 통합 설정"""
    timesnet: TimesNetConfig = None
    wavelet: WaveletConfig = None
    feature_dim: int = 64
    integration_method: str = 'attention'  # 'concat', 'attention', 'fusion'
    use_market_regime: bool = True
    
    def __post_init__(self):
        if self.timesnet is None:
            self.timesnet = TimesNetConfig()
        if self.wavelet is None:
            self.wavelet = WaveletConfig()


@dataclass
class TALibConfig:
    """TA-Lib 지표 설정"""
    # 선별된 상위 지표들
    selected_indicators: List[str] = None
    correlation_threshold: float = 0.8
    use_pca: bool = True
    pca_components: int = 20
    normalize: bool = True
    
    def __post_init__(self):
        if self.selected_indicators is None:
            # 상관관계 낮고 정보가치 높은 상위 30개 지표
            self.selected_indicators = [
                # Momentum
                'RSI_14', 'STOCHRSI_14', 'MACD_12_26_9', 'ROC_10', 'MOM_10',
                # Trend  
                'SMA_20', 'EMA_12', 'EMA_26', 'ADX_14', 'AROON_25',
                # Volatility
                'BBANDS_20_2', 'ATR_14', 'NATR_14', 'KELTNER_20_2',
                # Volume
                'OBV', 'MFI_14', 'AD', 'CMF_20',
                # Overlap
                'SAR', 'MIDPOINT_14', 'MIDPRICE_14',
                # Pattern
                'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING',
                # Cycle
                'HT_DCPERIOD', 'HT_TRENDMODE',
                # Statistics
                'VAR_5', 'STDDEV_5', 'TSF_14', 'CORREL_30'
            ]


@dataclass
class TradingEnvironmentConfig:
    """Trading Environment 설정"""
    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # 최대 포지션 크기 (자본 대비)
    transaction_cost: float = 0.001  # 거래 비용 0.1%
    slippage: float = 0.0001  # 슬리피지 0.01%
    
    # 상태 공간 구성
    state_features: Dict[str, int] = None
    
    # 액션 공간 구성
    action_space_type: str = 'continuous'  # 'discrete' or 'continuous'
    num_discrete_actions: int = 3  # BUY, SELL, HOLD
    
    # 보상 함수 가중치
    return_weight: float = 10.0
    sharpe_weight: float = 5.0
    mdd_penalty: float = -20.0
    cost_penalty: float = -100.0
    volatility_penalty: float = -50.0
    
    def __post_init__(self):
        if self.state_features is None:
            self.state_features = {
                'signal_generator': 2,      # sell_intensity + uncertainty
                'pattern_analyzer': 30,     # TimesNet + Wavelet features
                'talib_indicators': 30,     # 선별된 기술지표
                'position_info': 5,         # 포지션, PnL, 드로우다운 등
                'market_microstructure': 8, # 호가, 거래량, 변동성 등
                'market_regime': 4          # 변동성, 트렌드, 유동성, 상관관계
            }


@dataclass
class TrainingConfig:
    """훈련 설정"""
    # Signal Generator 훈련
    signal_epochs: int = 100
    signal_batch_size: int = 256  # 64 → 256 (4x increase)
    signal_lr: float = 2e-3  # Learning rate slightly increased for larger batch
    signal_weight_decay: float = 1e-5
    
    # Pattern Analyzer 훈련  
    pattern_epochs: int = 200
    pattern_batch_size: int = 128  # 32 → 128 (4x increase)
    pattern_lr: float = 1e-3  # Learning rate increased for larger batch
    pattern_weight_decay: float = 1e-4
    
    # SAC 훈련
    sac_episodes: int = 50000
    sac_buffer_size: int = 100000
    sac_batch_size: int = 256
    sac_lr_actor: float = 3e-4
    sac_lr_critic: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    
    # 검증 설정
    validation_split: float = 0.2
    test_split: float = 0.1
    walk_forward_validation: bool = True
    
    # 장치 설정
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    
    # 체크포인트 설정
    save_every: int = 10
    early_stopping_patience: int = 20
    best_metric: str = 'val_sharpe'  # 'val_loss', 'val_sharpe', 'val_return'


@dataclass
class EddieConfig:
    """Eddie 전체 시스템 통합 설정"""
    signal_generator: SignalGeneratorConfig = None
    pattern_analyzer: PatternAnalyzerConfig = None
    talib: TALibConfig = None
    environment: TradingEnvironmentConfig = None
    training: TrainingConfig = None
    
    # 데이터 설정
    data_path: str = "./data/feature_engineered"
    results_path: str = "./outputs/eddie_robust"
    model_save_path: str = "./models/eddie"
    
    # 실험 설정
    experiment_name: str = "eddie_signal_pattern_integration"
    version: str = "v1.0"
    seed: int = 42
    
    # 로깅 설정
    use_wandb: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.signal_generator is None:
            self.signal_generator = SignalGeneratorConfig()
        if self.pattern_analyzer is None:
            self.pattern_analyzer = PatternAnalyzerConfig()
        if self.talib is None:
            self.talib = TALibConfig()
        if self.environment is None:
            self.environment = TradingEnvironmentConfig()
        if self.training is None:
            self.training = TrainingConfig()


# 기본 설정 인스턴스
DEFAULT_CONFIG = EddieConfig()

# 메모리 효율적 설정 (안정적 훈련용)
QUICK_CONFIG = EddieConfig(
    signal_generator=SignalGeneratorConfig(
        tcn=TCNConfig(num_channels=[32, 64, 32]),  # 단순한 네트워크
        attention=AttentionConfig(d_model=32, num_heads=4),  # 적은 헤드
        seq_len=50  # 기본 시퀀스 길이
    ),
    pattern_analyzer=PatternAnalyzerConfig(
        timesnet=TimesNetConfig(seq_len=50, d_model=64, d_ff=128),  # 작은 모델
        feature_dim=32
    ),
    training=TrainingConfig(
        signal_epochs=20,  # 짧은 에포크
        pattern_epochs=30,  # 짧은 에포크
        signal_batch_size=32,  # 작은 배치
        pattern_batch_size=16,  # 작은 배치
        signal_lr=1e-3,  # 적절한 학습률
        pattern_lr=5e-4,  # 적절한 학습률
        mixed_precision=True,  # 메모리 절약
        sac_episodes=5000
    )
)

# 고성능 설정 (최종 모델용)
HIGH_PERFORMANCE_CONFIG = EddieConfig(
    signal_generator=SignalGeneratorConfig(
        tcn=TCNConfig(num_channels=[128, 256, 512, 256, 128]),
        attention=AttentionConfig(d_model=128, num_heads=16),
        seq_len=120
    ),
    pattern_analyzer=PatternAnalyzerConfig(
        timesnet=TimesNetConfig(seq_len=120, d_model=256),
        feature_dim=128
    ),
    training=TrainingConfig(
        signal_epochs=300,
        pattern_epochs=500,
        sac_episodes=100000
    )
)

# 최대 GPU 활용 설정 (RTX 4060 Ti 17.2GB 풀 활용)
MAX_GPU_CONFIG = EddieConfig(
    signal_generator=SignalGeneratorConfig(
        tcn=TCNConfig(num_channels=[256, 512, 1024, 2048, 1024, 512, 256]),  # 매우 깊고 넓은 네트워크
        attention=AttentionConfig(d_model=256, num_heads=32),  # 큰 어텐션 모델
        seq_len=120  # 긴 시퀀스
    ),
    pattern_analyzer=PatternAnalyzerConfig(
        timesnet=TimesNetConfig(
            seq_len=120, 
            d_model=512,  # 훨씬 큰 모델
            d_ff=2048,  # 매우 큰 feedforward
            num_kernels=16,  # 더 많은 커널
            num_layers=6  # 더 깊은 TimesNet
        ),
        feature_dim=256
    ),
    training=TrainingConfig(
        signal_epochs=50,
        pattern_epochs=100,
        signal_batch_size=2048,  # 매우 큰 배치 크기
        pattern_batch_size=1024,  # 큰 배치 크기
        signal_lr=1e-2,  # 매우 큰 배치에 맞는 높은 학습률
        pattern_lr=5e-3,  # 큰 배치에 맞는 높은 학습률
        mixed_precision=True,  # 필수
        sac_episodes=50000
    )
)

# 울트라 고성능 설정 (리소스 최대 활용)
ULTRA_HIGH_PERFORMANCE_CONFIG = EddieConfig(
    signal_generator=SignalGeneratorConfig(
        tcn=TCNConfig(num_channels=[512, 1024, 2048, 4096, 2048, 1024, 512]),  # 극도로 깊은 네트워크
        attention=AttentionConfig(d_model=512, num_heads=64),  # 초대형 어텐션
        seq_len=240  # 매우 긴 시퀀스 (4시간)
    ),
    pattern_analyzer=PatternAnalyzerConfig(
        timesnet=TimesNetConfig(
            seq_len=240, 
            d_model=1024,  # 초대형 모델
            d_ff=4096,  # 초대형 feedforward
            num_kernels=32,  # 최대 커널 수
            num_layers=8  # 매우 깊은 TimesNet
        ),
        feature_dim=512
    ),
    training=TrainingConfig(
        signal_epochs=100,
        pattern_epochs=200,
        signal_batch_size=4096,  # 초대형 배치
        pattern_batch_size=2048,  # 초대형 배치
        signal_lr=2e-2,  # 초대형 배치에 맞는 매우 높은 학습률
        pattern_lr=1e-2,  # 초대형 배치에 맞는 높은 학습률
        mixed_precision=True,  # 필수
        sac_episodes=100000
    )
) 