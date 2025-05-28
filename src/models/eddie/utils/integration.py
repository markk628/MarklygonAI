"""
Eddie Integration Utilities
MarklygonAI 메인 시스템과의 통합을 위한 유틸리티
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import logging
from datetime import datetime
import json

try:
    from ..config import EddieConfig, DEFAULT_CONFIG
    from ..signal_pipeline.signal_generator import SignalGenerator
    from ..pattern_pipeline.pattern_analyzer import PatternAnalyzer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import EddieConfig, DEFAULT_CONFIG
    from signal_pipeline.signal_generator import SignalGenerator
    from pattern_pipeline.pattern_analyzer import PatternAnalyzer


class EddiePredictor:
    """
    Eddie 통합 예측기 - 운영환경용 인터페이스
    Signal + Pattern Pipeline 통합 추론
    """
    
    def __init__(
        self,
        model_path: str,
        config: EddieConfig = None,
        device: torch.device = None
    ):
        self.model_path = Path(model_path)
        self.config = config or DEFAULT_CONFIG
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = logging.getLogger('EddiePredictor')
        
        # Load models
        self._load_models()
        
        # Feature cache for efficiency
        self._feature_cache = {}
        self._cache_size = 1000
        
    def _load_models(self):
        """모델 로드"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize models
            self.signal_generator = SignalGenerator(
                num_inputs=self.config.talib.pca_components,
                tcn_channels=self.config.signal_generator.tcn.num_channels,
                tcn_kernel_size=self.config.signal_generator.tcn.kernel_size,
                tcn_dropout=self.config.signal_generator.tcn.dropout,
                attention_heads=self.config.signal_generator.attention.num_heads,
                attention_dropout=self.config.signal_generator.attention.dropout,
                seq_len=self.config.signal_generator.seq_len,
                use_uncertainty=self.config.signal_generator.use_uncertainty
            ).to(self.device)
            
            self.pattern_analyzer = PatternAnalyzer(
                num_features=self.config.talib.pca_components,
                seq_len=self.config.signal_generator.seq_len,
                feature_dim=self.config.pattern_analyzer.feature_dim,
                integration_method=self.config.pattern_analyzer.integration_method,
                use_market_regime=self.config.pattern_analyzer.use_market_regime
            ).to(self.device)
            
            # Load states
            self.signal_generator.load_state_dict(checkpoint['signal_generator_state'])
            self.pattern_analyzer.load_state_dict(checkpoint['pattern_analyzer_state'])
            
            # Set to evaluation mode
            self.signal_generator.eval()
            self.pattern_analyzer.eval()
            
            self.logger.info(f"Models loaded successfully from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(
        self,
        indicators_data: Union[np.ndarray, torch.Tensor],
        return_detailed: bool = False,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        실시간 예측 수행
        
        Args:
            indicators_data: (seq_len, num_features) or (1, seq_len, num_features)
            return_detailed: 상세 정보 반환 여부
            use_cache: 캐시 사용 여부
            
        Returns:
            predictions: 예측 결과 딕셔너리
        """
        # Input validation and preprocessing
        if isinstance(indicators_data, np.ndarray):
            indicators_data = torch.FloatTensor(indicators_data)
        
        if indicators_data.dim() == 2:
            indicators_data = indicators_data.unsqueeze(0)  # Add batch dimension
        
        # Check cache
        cache_key = self._get_cache_key(indicators_data) if use_cache else None
        if cache_key and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Move to device
        indicators_data = indicators_data.to(self.device)
        
        with torch.no_grad():
            # Signal prediction
            signal_outputs = self.signal_generator(indicators_data)
            
            # Pattern analysis
            pattern_outputs = self.pattern_analyzer(indicators_data, return_detailed=return_detailed)
            
            # Combine results
            predictions = {
                # Main signals
                'sell_intensity': signal_outputs['sell_intensity'].cpu().item(),
                'uncertainty': signal_outputs.get('uncertainty', torch.tensor(0.0)).cpu().item(),
                
                # Pattern features
                'volatility': pattern_outputs['volatility'].cpu().item(),
                'trend_strength': pattern_outputs['trend_strength'].cpu().item(),
                'pattern_confidence': pattern_outputs['pattern_confidence'].cpu().item(),
                
                # Market regime
                'market_regime_probs': pattern_outputs.get('regime_probabilities', torch.zeros(4)).cpu().numpy(),
                'market_regime': int(torch.argmax(pattern_outputs.get('regime_probabilities', torch.zeros(4))).item()),
                
                # Meta information
                'timestamp': datetime.now().isoformat(),
                'model_version': self.config.version,
                'confidence_score': self._calculate_confidence_score(signal_outputs, pattern_outputs)
            }
            
            # Add detailed information if requested
            if return_detailed:
                predictions['detailed'] = {
                    'signal_features': signal_outputs.get('features', torch.zeros(1)).cpu().numpy(),
                    'pattern_features': pattern_outputs.get('final_features', torch.zeros(1)).cpu().numpy(),
                    'regime_confidence': pattern_outputs.get('regime_confidence', torch.tensor(0.0)).cpu().item()
                }
        
        # Cache result
        if cache_key:
            self._update_cache(cache_key, predictions)
        
        return predictions
    
    def predict_batch(
        self,
        batch_indicators: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        배치 예측 (효율적인 대량 처리)
        
        Args:
            batch_indicators: (batch_size, seq_len, num_features)
            batch_size: 처리 배치 크기
            
        Returns:
            batch_predictions: 예측 결과 리스트
        """
        if isinstance(batch_indicators, np.ndarray):
            batch_indicators = torch.FloatTensor(batch_indicators)
        
        batch_indicators = batch_indicators.to(self.device)
        total_samples = batch_indicators.size(0)
        
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_data = batch_indicators[i:end_idx]
                
                # Batch prediction
                signal_outputs = self.signal_generator(batch_data)
                pattern_outputs = self.pattern_analyzer(batch_data)
                
                # Convert to individual predictions
                for j in range(batch_data.size(0)):
                    pred = {
                        'sell_intensity': signal_outputs['sell_intensity'][j].cpu().item(),
                        'uncertainty': signal_outputs.get('uncertainty', torch.zeros_like(signal_outputs['sell_intensity']))[j].cpu().item(),
                        'volatility': pattern_outputs['volatility'][j].cpu().item(),
                        'trend_strength': pattern_outputs['trend_strength'][j].cpu().item(),
                        'pattern_confidence': pattern_outputs['pattern_confidence'][j].cpu().item(),
                        'market_regime_probs': pattern_outputs.get('regime_probabilities', torch.zeros(batch_data.size(0), 4))[j].cpu().numpy(),
                        'market_regime': int(torch.argmax(pattern_outputs.get('regime_probabilities', torch.zeros(batch_data.size(0), 4))[j]).item()),
                        'timestamp': datetime.now().isoformat(),
                        'sample_index': i + j
                    }
                    all_predictions.append(pred)
        
        return all_predictions
    
    def get_trading_signal(
        self,
        indicators_data: Union[np.ndarray, torch.Tensor],
        threshold_config: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        트레이딩 신호 생성 (실제 매매용)
        
        Args:
            indicators_data: 지표 데이터
            threshold_config: 임계값 설정
            
        Returns:
            trading_signal: 매매 신호 정보
        """
        # Default thresholds
        if threshold_config is None:
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
        
        # Get prediction
        prediction = self.predict(indicators_data, return_detailed=True)
        
        # Extract key values
        sell_intensity = prediction['sell_intensity']
        confidence = prediction['confidence_score']
        uncertainty = prediction['uncertainty']
        pattern_confidence = prediction['pattern_confidence']
        volatility = prediction['volatility']
        
        # Determine trading action
        action = 'HOLD'
        strength = 0.0
        reason = []
        
        # Check confidence filters
        if confidence < threshold_config['min_confidence']:
            reason.append(f"Low confidence: {confidence:.3f}")
        
        if uncertainty > threshold_config['max_uncertainty']:
            reason.append(f"High uncertainty: {uncertainty:.3f}")
        
        # Determine action based on sell_intensity
        if len(reason) == 0:  # Only trade if confidence is sufficient
            if sell_intensity <= threshold_config['strong_sell_threshold']:
                action = 'STRONG_SELL'
                strength = abs(sell_intensity)
                reason.append(f"Strong sell signal: {sell_intensity:.3f}")
                
            elif sell_intensity <= threshold_config['sell_threshold']:
                action = 'SELL'
                strength = abs(sell_intensity) * 0.7
                reason.append(f"Sell signal: {sell_intensity:.3f}")
                
            elif sell_intensity >= threshold_config['strong_buy_threshold']:
                action = 'STRONG_BUY'
                strength = sell_intensity
                reason.append(f"Strong buy signal: {sell_intensity:.3f}")
                
            elif sell_intensity >= threshold_config['buy_threshold']:
                action = 'BUY'
                strength = sell_intensity * 0.7
                reason.append(f"Buy signal: {sell_intensity:.3f}")
                
            elif (threshold_config['hold_threshold_low'] <= sell_intensity <= 
                  threshold_config['hold_threshold_high']):
                action = 'HOLD'
                strength = 0.0
                reason.append(f"Neutral signal: {sell_intensity:.3f}")
        
        # Volatility adjustment
        if volatility > 0.8:
            strength *= 0.8  # Reduce strength in high volatility
            reason.append(f"High volatility adjustment: {volatility:.3f}")
        
        # Pattern confidence adjustment
        if pattern_confidence < 0.5:
            strength *= 0.9
            reason.append(f"Low pattern confidence: {pattern_confidence:.3f}")
        
        return {
            'action': action,
            'strength': np.clip(strength, 0.0, 1.0),
            'sell_intensity': sell_intensity,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'volatility': volatility,
            'pattern_confidence': pattern_confidence,
            'market_regime': prediction['market_regime'],
            'reason': '; '.join(reason),
            'timestamp': prediction['timestamp'],
            'raw_prediction': prediction
        }
    
    def _calculate_confidence_score(
        self,
        signal_outputs: Dict[str, torch.Tensor],
        pattern_outputs: Dict[str, torch.Tensor]
    ) -> float:
        """종합 신뢰도 점수 계산"""
        factors = []
        
        # Signal uncertainty (lower is better)
        if 'uncertainty' in signal_outputs:
            uncertainty = signal_outputs['uncertainty'].cpu().item()
            factors.append(1.0 - uncertainty)
        
        # Pattern confidence
        pattern_conf = pattern_outputs.get('pattern_confidence', torch.tensor(0.5)).cpu().item()
        factors.append(pattern_conf)
        
        # Market regime confidence
        if 'regime_probabilities' in pattern_outputs:
            regime_probs = pattern_outputs['regime_probabilities'].cpu()
            regime_conf = torch.max(regime_probs).item()  # Highest probability
            factors.append(regime_conf)
        
        # Signal strength (closer to extremes is more confident)
        sell_intensity = signal_outputs['sell_intensity'].cpu().item()
        signal_strength = abs(sell_intensity)
        factors.append(signal_strength)
        
        # Weighted average
        if factors:
            weights = [0.3, 0.3, 0.2, 0.2][:len(factors)]
            weights = weights / np.sum(weights)  # Normalize
            confidence = np.average(factors, weights=weights)
        else:
            confidence = 0.5
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _get_cache_key(self, data: torch.Tensor) -> str:
        """캐시 키 생성"""
        # Use hash of data for cache key
        data_hash = hash(data.cpu().numpy().tobytes())
        return f"eddie_pred_{data_hash}"
    
    def _update_cache(self, key: str, prediction: Dict):
        """캐시 업데이트"""
        if len(self._feature_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._feature_cache))
            del self._feature_cache[oldest_key]
        
        self._feature_cache[key] = prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_path': str(self.model_path),
            'config_version': self.config.version,
            'device': str(self.device),
            'signal_generator_params': sum(p.numel() for p in self.signal_generator.parameters()),
            'pattern_analyzer_params': sum(p.numel() for p in self.pattern_analyzer.parameters()),
            'cache_size': len(self._feature_cache),
            'sequence_length': self.config.signal_generator.seq_len,
            'feature_count': self.config.talib.pca_components
        }


class EddieDataProcessor:
    """
    Eddie 데이터 전처리기
    TA-Lib 지표를 Eddie 입력 형식으로 변환
    """
    
    def __init__(self, config: EddieConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.feature_scaler = None
        self.feature_names = None
        
    def prepare_features(
        self,
        talib_indicators: pd.DataFrame,
        fit_scaler: bool = False
    ) -> torch.Tensor:
        """
        TA-Lib 지표를 Eddie 입력 형식으로 변환
        
        Args:
            talib_indicators: TA-Lib 지표 DataFrame
            fit_scaler: 스케일러 학습 여부
            
        Returns:
            features: 전처리된 특징 텐서
        """
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.decomposition import PCA
        
        # Handle missing values
        talib_indicators = talib_indicators.fillna(method='ffill').fillna(method='bfill')
        
        # Extract selected indicators
        if self.config.talib.selected_indicators:
            available_indicators = [col for col in self.config.talib.selected_indicators 
                                  if col in talib_indicators.columns]
            if available_indicators:
                indicators_subset = talib_indicators[available_indicators]
            else:
                indicators_subset = talib_indicators
        else:
            indicators_subset = talib_indicators
        
        # Normalization
        if self.config.talib.normalize:
            if fit_scaler or self.feature_scaler is None:
                self.feature_scaler = RobustScaler()
                normalized_data = self.feature_scaler.fit_transform(indicators_subset.values)
            else:
                normalized_data = self.feature_scaler.transform(indicators_subset.values)
        else:
            normalized_data = indicators_subset.values
        
        # PCA dimensionality reduction
        if self.config.talib.use_pca:
            if fit_scaler or not hasattr(self, 'pca'):
                self.pca = PCA(n_components=self.config.talib.pca_components)
                features = self.pca.fit_transform(normalized_data)
            else:
                features = self.pca.transform(normalized_data)
        else:
            features = normalized_data
        
        # Store feature names
        if fit_scaler:
            if self.config.talib.use_pca:
                self.feature_names = [f'PCA_{i}' for i in range(self.config.talib.pca_components)]
            else:
                self.feature_names = list(indicators_subset.columns)
        
        return torch.FloatTensor(features)
    
    def create_sequences(
        self,
        features: torch.Tensor,
        seq_len: int = None
    ) -> torch.Tensor:
        """
        시퀀스 데이터 생성
        
        Args:
            features: (T, num_features) 특징 텐서
            seq_len: 시퀀스 길이
            
        Returns:
            sequences: (N, seq_len, num_features) 시퀀스 텐서
        """
        if seq_len is None:
            seq_len = self.config.signal_generator.seq_len
        
        T, num_features = features.shape
        if T < seq_len:
            raise ValueError(f"Data length {T} is shorter than sequence length {seq_len}")
        
        num_sequences = T - seq_len + 1
        sequences = torch.zeros(num_sequences, seq_len, num_features)
        
        for i in range(num_sequences):
            sequences[i] = features[i:i + seq_len]
        
        return sequences
    
    def save_preprocessor(self, path: str):
        """전처리기 저장"""
        preprocessor_data = {
            'config': self.config,
            'feature_scaler': self.feature_scaler,
            'pca': getattr(self, 'pca', None),
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
    
    def load_preprocessor(self, path: str):
        """전처리기 로드"""
        with open(path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.config = preprocessor_data['config']
        self.feature_scaler = preprocessor_data['feature_scaler']
        if 'pca' in preprocessor_data:
            self.pca = preprocessor_data['pca']
        self.feature_names = preprocessor_data['feature_names']


class EddieRealTimeInterface:
    """
    Eddie 실시간 인터페이스
    실시간 데이터 스트림 처리
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str = None,
        buffer_size: int = 200
    ):
        self.predictor = EddiePredictor(model_path)
        self.processor = EddieDataProcessor()
        
        if preprocessor_path:
            self.processor.load_preprocessor(preprocessor_path)
        
        # Real-time data buffer
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.predictions_history = []
        
    def update_data(self, new_indicators: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        실시간 데이터 업데이트 및 예측
        
        Args:
            new_indicators: 새로운 지표 데이터
            
        Returns:
            prediction: 예측 결과 (충분한 데이터가 있을 때만)
        """
        # Add to buffer
        self.data_buffer.append(new_indicators)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
        
        # Check if we have enough data for prediction
        seq_len = self.predictor.config.signal_generator.seq_len
        if len(self.data_buffer) < seq_len:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        # Preprocess
        try:
            features = self.processor.prepare_features(df, fit_scaler=False)
            
            # Get latest sequence
            latest_sequence = features[-seq_len:].unsqueeze(0)  # Add batch dimension
            
            # Predict
            prediction = self.predictor.predict(latest_sequence)
            
            # Store in history
            self.predictions_history.append(prediction)
            if len(self.predictions_history) > 100:  # Keep last 100 predictions
                self.predictions_history.pop(0)
            
            return prediction
            
        except Exception as e:
            logging.error(f"Real-time prediction error: {e}")
            return None
    
    def get_trading_recommendation(self, new_indicators: Dict[str, float]) -> Dict[str, Any]:
        """실시간 매매 추천"""
        prediction = self.update_data(new_indicators)
        
        if prediction is None:
            return {
                'action': 'WAIT',
                'reason': 'Insufficient data for prediction',
                'data_length': len(self.data_buffer),
                'required_length': self.predictor.config.signal_generator.seq_len
            }
        
        # Get trading signal
        latest_sequence = self._get_latest_sequence()
        if latest_sequence is not None:
            trading_signal = self.predictor.get_trading_signal(latest_sequence)
            return trading_signal
        else:
            return {'action': 'WAIT', 'reason': 'Data preprocessing failed'}
    
    def _get_latest_sequence(self) -> Optional[torch.Tensor]:
        """최신 시퀀스 데이터 반환"""
        seq_len = self.predictor.config.signal_generator.seq_len
        if len(self.data_buffer) < seq_len:
            return None
        
        try:
            df = pd.DataFrame(self.data_buffer)
            features = self.processor.prepare_features(df, fit_scaler=False)
            return features[-seq_len:].unsqueeze(0)
        except:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.predictions_history:
            return {'message': 'No predictions yet'}
        
        # Calculate basic statistics
        recent_predictions = self.predictions_history[-20:]  # Last 20 predictions
        
        sell_intensities = [p['sell_intensity'] for p in recent_predictions]
        confidences = [p['confidence_score'] for p in recent_predictions]
        uncertainties = [p['uncertainty'] for p in recent_predictions]
        
        return {
            'total_predictions': len(self.predictions_history),
            'recent_predictions': len(recent_predictions),
            'avg_sell_intensity': np.mean(sell_intensities),
            'avg_confidence': np.mean(confidences),
            'avg_uncertainty': np.mean(uncertainties),
            'sell_intensity_std': np.std(sell_intensities),
            'confidence_std': np.std(confidences),
            'buffer_utilization': len(self.data_buffer) / self.buffer_size,
            'last_prediction_time': recent_predictions[-1]['timestamp'] if recent_predictions else None
        }


# Utility functions for easy integration
def load_eddie_predictor(model_path: str, device: str = 'auto') -> EddiePredictor:
    """Eddie 예측기 간단 로드"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    return EddiePredictor(model_path, device=device)


def quick_predict(model_path: str, indicators_data: np.ndarray) -> Dict[str, Any]:
    """빠른 예측 (일회성 사용)"""
    predictor = load_eddie_predictor(model_path)
    return predictor.predict(indicators_data)


def create_realtime_interface(model_path: str, preprocessor_path: str = None) -> EddieRealTimeInterface:
    """실시간 인터페이스 생성"""
    return EddieRealTimeInterface(model_path, preprocessor_path) 