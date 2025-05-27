"""
Eddie System Usage Examples
Eddie Signal + Pattern Pipeline 사용 예제 모음
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Eddie imports
from config import EddieConfig, DEFAULT_CONFIG, QUICK_CONFIG
from train_pipeline import EddieTrainer, EddieDataset
from signal_pipeline.signal_generator import SignalGenerator
from pattern_pipeline.pattern_analyzer import PatternAnalyzer
from utils.integration import (
    EddiePredictor, 
    EddieDataProcessor, 
    EddieRealTimeInterface,
    load_eddie_predictor,
    quick_predict
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_training():
    """
    예제 1: 기본 훈련 과정
    """
    print("=" * 60)
    print("예제 1: Eddie 기본 훈련")
    print("=" * 60)
    
    # 1. 설정 선택
    config = QUICK_CONFIG  # 빠른 테스트용
    
    # 2. 훈련 설정 커스터마이징
    config.training.signal_epochs = 5  # 예제용으로 짧게
    config.training.pattern_epochs = 5
    config.use_wandb = False  # 예제에서는 wandb 비활성화
    
    print(f"Configuration: {config.experiment_name}")
    print(f"Signal epochs: {config.training.signal_epochs}")
    print(f"Pattern epochs: {config.training.pattern_epochs}")
    
    try:
        # 3. 트레이너 초기화
        trainer = EddieTrainer(config, use_wandb=False)
        
        # 4. 훈련 실행 (에포크 수가 적어서 빠르게 완료)
        print("\n훈련 시작...")
        trainer.train(num_epochs=5)
        
        print("✅ 훈련 완료!")
        
    except FileNotFoundError as e:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        print("   실제 훈련을 위해서는 feature engineered 데이터가 필요합니다.")
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")


def example_2_model_components():
    """
    예제 2: 개별 모델 컴포넌트 테스트
    """
    print("\n" + "=" * 60)
    print("예제 2: 개별 모델 컴포넌트 테스트")
    print("=" * 60)
    
    # 가상 데이터 생성
    batch_size, seq_len, num_features = 4, 60, 30
    mock_data = torch.randn(batch_size, seq_len, num_features)
    
    print(f"Mock data shape: {mock_data.shape}")
    
    # 1. Signal Generator 테스트
    print("\n1. Signal Generator 테스트")
    signal_gen = SignalGenerator(
        num_inputs=num_features,
        seq_len=seq_len,
        use_uncertainty=True,
        use_adaptive_attention=True
    )
    
    with torch.no_grad():
        signal_outputs = signal_gen(mock_data)
        
        print(f"   Sell intensity: {signal_outputs['sell_intensity']}")
        print(f"   Uncertainty: {signal_outputs.get('uncertainty', 'N/A')}")
        print(f"   Market regime: {signal_outputs.get('market_regime', 'N/A')}")
    
    # 2. Pattern Analyzer 테스트
    print("\n2. Pattern Analyzer 테스트")
    pattern_analyzer = PatternAnalyzer(
        num_features=num_features,
        seq_len=seq_len,
        use_market_regime=True,
        integration_method='attention'
    )
    
    with torch.no_grad():
        pattern_outputs = pattern_analyzer(mock_data, return_detailed=True)
        
        print(f"   Volatility: {pattern_outputs['volatility']}")
        print(f"   Trend strength: {pattern_outputs['trend_strength']}")
        print(f"   Pattern confidence: {pattern_outputs['pattern_confidence']}")
        
        if 'regime_probabilities' in pattern_outputs:
            regime_probs = pattern_outputs['regime_probabilities']
            print(f"   Market regime probs: {regime_probs}")
    
    print("✅ 모델 컴포넌트 테스트 완료!")


def example_3_prediction_interface():
    """
    예제 3: 예측 인터페이스 사용
    """
    print("\n" + "=" * 60)
    print("예제 3: 예측 인터페이스 사용")
    print("=" * 60)
    
    # 가상 데이터로 예측 인터페이스 테스트
    seq_len, num_features = 60, 30
    mock_indicators = np.random.randn(seq_len, num_features)
    
    print(f"Mock indicators shape: {mock_indicators.shape}")
    
    # 사전 훈련된 모델이 없는 경우를 시뮬레이션
    try:
        # 가상의 모델 경로 (실제로는 존재하지 않음)
        model_path = "./models/eddie/best_model.pt"
        
        if not Path(model_path).exists():
            print("❌ 사전 훈련된 모델이 없습니다.")
            print("   실제 사용을 위해서는 먼저 훈련을 완료해야 합니다.")
            print(f"   모델 경로: {model_path}")
            
            # 모델 구조만 보여주기
            print("\n📋 예상되는 예측 결과 형식:")
            mock_prediction = {
                'sell_intensity': -0.234,
                'uncertainty': 0.156,
                'volatility': 0.678,
                'trend_strength': -0.345,
                'pattern_confidence': 0.789,
                'market_regime_probs': np.array([0.1, 0.6, 0.2, 0.1]),
                'market_regime': 1,
                'confidence_score': 0.723,
                'timestamp': datetime.now().isoformat()
            }
            
            for key, value in mock_prediction.items():
                if isinstance(value, np.ndarray):
                    print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {value}")
            
        else:
            # 실제 모델이 있는 경우 예측 수행
            predictor = load_eddie_predictor(model_path)
            prediction = predictor.predict(mock_indicators, return_detailed=True)
            
            print("🎯 예측 결과:")
            for key, value in prediction.items():
                if key != 'detailed':
                    print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ 예측 중 오류: {e}")


def example_4_trading_signals():
    """
    예제 4: 트레이딩 신호 생성
    """
    print("\n" + "=" * 60)
    print("예제 4: 트레이딩 신호 생성")
    print("=" * 60)
    
    # 다양한 시나리오의 가상 예측 결과
    scenarios = [
        {
            'name': '강한 매도 신호',
            'sell_intensity': -0.85,
            'uncertainty': 0.15,
            'volatility': 0.45,
            'pattern_confidence': 0.82
        },
        {
            'name': '약한 매수 신호',
            'sell_intensity': 0.25,
            'uncertainty': 0.35,
            'volatility': 0.62,
            'pattern_confidence': 0.58
        },
        {
            'name': '불확실한 신호',
            'sell_intensity': -0.45,
            'uncertainty': 0.75,
            'volatility': 0.89,
            'pattern_confidence': 0.32
        },
        {
            'name': '중립 신호',
            'sell_intensity': 0.05,
            'uncertainty': 0.25,
            'volatility': 0.35,
            'pattern_confidence': 0.71
        }
    ]
    
    # 임계값 설정
    threshold_config = {
        'strong_sell_threshold': -0.7,
        'sell_threshold': -0.3,
        'hold_threshold_low': -0.1,
        'hold_threshold_high': 0.1,
        'buy_threshold': 0.3,
        'strong_buy_threshold': 0.7,
        'min_confidence': 0.6,
        'max_uncertainty': 0.3
    }
    
    print("📊 트레이딩 신호 분석:")
    print(f"   임계값 설정: {threshold_config}")
    print()
    
    for scenario in scenarios:
        print(f"🔍 시나리오: {scenario['name']}")
        
        # 신호 결정 로직 시뮬레이션
        sell_intensity = scenario['sell_intensity']
        uncertainty = scenario['uncertainty']
        volatility = scenario['volatility']
        pattern_confidence = scenario['pattern_confidence']
        
        # 종합 신뢰도 계산
        confidence_score = (
            (1.0 - uncertainty) * 0.4 +
            pattern_confidence * 0.4 +
            abs(sell_intensity) * 0.2
        )
        
        # 액션 결정
        action = 'HOLD'
        strength = 0.0
        reasons = []
        
        # 신뢰도 체크
        if confidence_score < threshold_config['min_confidence']:
            reasons.append(f"낮은 신뢰도: {confidence_score:.3f}")
        
        if uncertainty > threshold_config['max_uncertainty']:
            reasons.append(f"높은 불확실성: {uncertainty:.3f}")
        
        # 신호 결정
        if len(reasons) == 0:
            if sell_intensity <= threshold_config['strong_sell_threshold']:
                action = 'STRONG_SELL'
                strength = abs(sell_intensity)
                reasons.append(f"강한 매도: {sell_intensity:.3f}")
            elif sell_intensity <= threshold_config['sell_threshold']:
                action = 'SELL'
                strength = abs(sell_intensity) * 0.7
                reasons.append(f"매도: {sell_intensity:.3f}")
            elif sell_intensity >= threshold_config['strong_buy_threshold']:
                action = 'STRONG_BUY'
                strength = sell_intensity
                reasons.append(f"강한 매수: {sell_intensity:.3f}")
            elif sell_intensity >= threshold_config['buy_threshold']:
                action = 'BUY'
                strength = sell_intensity * 0.7
                reasons.append(f"매수: {sell_intensity:.3f}")
        
        # 변동성 조정
        if volatility > 0.8:
            strength *= 0.8
            reasons.append(f"고변동성 조정: {volatility:.3f}")
        
        print(f"   입력: sell_intensity={sell_intensity:.3f}, uncertainty={uncertainty:.3f}")
        print(f"   출력: action={action}, strength={strength:.3f}")
        print(f"   사유: {'; '.join(reasons) if reasons else '정상 신호'}")
        print()


def example_5_realtime_simulation():
    """
    예제 5: 실시간 데이터 처리 시뮬레이션
    """
    print("\n" + "=" * 60)
    print("예제 5: 실시간 데이터 처리 시뮬레이션")
    print("=" * 60)
    
    # 실시간 인터페이스는 모델이 있어야 동작하므로, 시뮬레이션만 수행
    print("📡 실시간 데이터 스트림 시뮬레이션")
    
    # 가상의 TA-Lib 지표 이름들
    indicator_names = [
        'RSI_14', 'MACD_12_26_9', 'BB_upper', 'BB_lower', 'SMA_20',
        'EMA_12', 'ATR_14', 'OBV', 'MFI_14', 'ADX_14',
        'STOCH_K', 'STOCH_D', 'CCI_20', 'ROC_10', 'MOM_10'
    ]
    
    # 시간별 데이터 스트림 시뮬레이션
    buffer = []
    predictions = []
    
    print(f"   지표: {', '.join(indicator_names[:5])}... (총 {len(indicator_names)}개)")
    print()
    
    for minute in range(70):  # 70분간 데이터 스트림
        # 가상 지표 데이터 생성 (실제로는 시장 데이터에서 받아옴)
        current_time = datetime.now() + timedelta(minutes=minute)
        indicators = {name: np.random.randn() for name in indicator_names}
        
        # 버퍼에 추가
        buffer.append({
            'timestamp': current_time,
            **indicators
        })
        
        # 60분 윈도우 유지
        if len(buffer) > 60:
            buffer.pop(0)
        
        # 60분 데이터가 모이면 예측 수행 (시뮬레이션)
        if len(buffer) >= 60:
            # 실제로는 Eddie 모델이 예측을 수행
            mock_prediction = {
                'timestamp': current_time.isoformat(),
                'sell_intensity': np.random.uniform(-1, 1),
                'uncertainty': np.random.uniform(0, 1),
                'volatility': np.random.uniform(0, 1),
                'confidence_score': np.random.uniform(0, 1),
                'action': np.random.choice(['BUY', 'SELL', 'HOLD'])
            }
            predictions.append(mock_prediction)
            
            if minute % 10 == 0:  # 10분마다 상태 출력
                print(f"   시간 {minute:02d}: "
                      f"action={mock_prediction['action']}, "
                      f"sell_intensity={mock_prediction['sell_intensity']:.3f}, "
                      f"confidence={mock_prediction['confidence_score']:.3f}")
    
    print(f"\n📈 시뮬레이션 완료: {len(predictions)}개 예측 생성")
    
    # 간단한 통계
    if predictions:
        actions = [p['action'] for p in predictions]
        sell_intensities = [p['sell_intensity'] for p in predictions]
        
        action_counts = {action: actions.count(action) for action in ['BUY', 'SELL', 'HOLD']}
        avg_sell_intensity = np.mean(sell_intensities)
        
        print(f"   액션 분포: {action_counts}")
        print(f"   평균 매도 강도: {avg_sell_intensity:.3f}")


def example_6_data_preprocessing():
    """
    예제 6: 데이터 전처리 과정
    """
    print("\n" + "=" * 60)
    print("예제 6: 데이터 전처리 과정")
    print("=" * 60)
    
    # 가상의 TA-Lib 지표 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    
    # 30개 지표 시뮬레이션
    indicators_data = {}
    for i in range(30):
        # 각 지표는 서로 다른 특성을 가짐
        if i < 10:  # 모멘텀 지표들 (oscillating)
            indicators_data[f'momentum_{i}'] = np.sin(np.linspace(0, 20*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        elif i < 20:  # 트렌드 지표들 (trending)
            indicators_data[f'trend_{i}'] = np.cumsum(np.random.normal(0, 0.01, 1000)) + np.random.normal(0, 0.05, 1000)
        else:  # 변동성 지표들 (volatility)
            indicators_data[f'volatility_{i}'] = np.abs(np.random.normal(0, 1, 1000))
    
    df = pd.DataFrame(indicators_data, index=dates)
    
    print(f"📊 원본 데이터 형태: {df.shape}")
    print(f"   시간 범위: {df.index[0]} ~ {df.index[-1]}")
    print(f"   지표 수: {len(df.columns)}")
    
    # 전처리기 생성
    processor = EddieDataProcessor(DEFAULT_CONFIG)
    
    print("\n🔧 전처리 과정:")
    
    # 1. 특징 전처리
    print("   1. 특징 정규화 및 PCA 적용...")
    features = processor.prepare_features(df, fit_scaler=True)
    print(f"      전처리 후 형태: {features.shape}")
    
    # 2. 시퀀스 생성
    print("   2. 시퀀스 데이터 생성...")
    sequences = processor.create_sequences(features, seq_len=60)
    print(f"      시퀀스 형태: {sequences.shape}")
    
    # 3. 통계 정보
    print("\n📈 전처리 통계:")
    print(f"   원본 특징 수: {df.shape[1]}")
    print(f"   PCA 후 특징 수: {features.shape[1]}")
    print(f"   생성된 시퀀스 수: {sequences.shape[0]}")
    print(f"   시퀀스 길이: {sequences.shape[1]}")
    
    # 4. 특징 이름 확인
    if processor.feature_names:
        print(f"   특징 이름: {processor.feature_names[:5]}... (총 {len(processor.feature_names)}개)")
    
    print("✅ 데이터 전처리 완료!")


def example_7_model_analysis():
    """
    예제 7: 모델 분석 및 해석
    """
    print("\n" + "=" * 60)
    print("예제 7: 모델 분석 및 해석")
    print("=" * 60)
    
    # 가상 분석 결과
    print("🔬 모델 아키텍처 분석:")
    
    # Signal Generator 분석
    print("\n1. Signal Generator (TCN + Attention)")
    print("   - TCN 레이어: 5층 (64→128→256→128→64)")
    print("   - Attention heads: 8개")
    print("   - 파라미터 수: ~1.2M")
    print("   - 출력: sell_intensity [-1,1], uncertainty [0,1]")
    
    # Pattern Analyzer 분석
    print("\n2. Pattern Analyzer (TimesNet + Wavelet)")
    print("   - TimesNet 주기 분석: top-5 주파수")
    print("   - Wavelet 레벨: 5단계 (db4)")
    print("   - 파라미터 수: ~2.1M")
    print("   - 출력: volatility, trend_strength, pattern_confidence")
    
    # 성능 분석 (가상)
    print("\n📊 성능 메트릭 (시뮬레이션):")
    
    metrics = {
        'Signal Generator': {
            'MSE': 0.0847,
            'MAE': 0.2314,
            'R²': 0.6521,
            'Uncertainty Calibration': 0.7834
        },
        'Pattern Analyzer': {
            'Volatility MSE': 0.0623,
            'Trend Accuracy': 0.7145,
            'Regime Accuracy': 0.6892,
            'Pattern Confidence': 0.7456
        },
        'Integrated System': {
            'Combined Loss': 0.3421,
            'Consistency Score': 0.8123,
            'Trading Signal Accuracy': 0.6987
        }
    }
    
    for component, metric_dict in metrics.items():
        print(f"\n   {component}:")
        for metric, value in metric_dict.items():
            print(f"     {metric}: {value:.4f}")
    
    # 복잡성 분석
    print("\n⚠️  복잡성 분석:")
    print("   - 총 파라미터: ~3.3M")
    print("   - GPU 메모리 요구량: ~8GB (배치=64)")
    print("   - 추론 시간: ~15ms (GPU), ~80ms (CPU)")
    print("   - 훈련 시간: ~12시간 (100 epochs)")
    
    # 한계점
    print("\n🚨 시스템 한계점:")
    limitations = [
        "높은 계산 복잡도로 인한 실시간 처리 부담",
        "대량의 고품질 훈련 데이터 필요",
        "과적합 위험성 (복잡한 아키텍처)",
        "시장 레짐 변화에 대한 적응 시간 필요",
        "불확실성 추정의 캘리브레이션 이슈"
    ]
    
    for i, limitation in enumerate(limitations, 1):
        print(f"   {i}. {limitation}")


def main():
    """
    모든 예제 실행
    """
    print("🚀 Eddie System Usage Examples")
    print("=" * 80)
    
    examples = [
        ("기본 훈련", example_1_basic_training),
        ("모델 컴포넌트", example_2_model_components),
        ("예측 인터페이스", example_3_prediction_interface),
        ("트레이딩 신호", example_4_trading_signals),
        ("실시간 시뮬레이션", example_5_realtime_simulation),
        ("데이터 전처리", example_6_data_preprocessing),
        ("모델 분석", example_7_model_analysis)
    ]
    
    try:
        for name, example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"❌ {name} 예제 실행 중 오류: {e}")
                continue
        
        print("\n" + "=" * 80)
        print("🎉 모든 예제 실행 완료!")
        print("\n📚 추가 정보:")
        print("   - 상세 문서: README.md")
        print("   - 설정 파일: config.py")
        print("   - 훈련 스크립트: train_pipeline.py")
        print("   - 통합 유틸리티: utils/integration.py")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 예기치 않은 오류: {e}")


if __name__ == "__main__":
    main() 