"""
Pattern Analyzer: TimesNet + Wavelet Integration
다중 스케일 패턴 분석을 위한 통합 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .timesnet import MultiVariateTimesNet, TimesNet
from .wavelet_transform import WaveletFeatureExtractor, TorchWaveletTransform


class MarketRegimeDetector(nn.Module):
    """
    시장 레짐 탐지 모듈
    4가지 시장 상태를 분류: 상승 트렌드, 하락 트렌드, 횡보, 고변동성
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 4),  # 4 market regimes
            nn.Softmax(dim=-1)
        )
        
        # 레짐별 특성 학습
        self.regime_embeddings = nn.Embedding(4, hidden_dim // 4)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch_size, input_dim) - Combined pattern features
            
        Returns:
            regime_info: Dictionary containing regime probabilities and embeddings
        """
        # 시장 레짐 분류
        regime_probs = self.regime_classifier(features)  # (batch_size, 4)
        
        # 가장 가능성 높은 레짐
        regime_indices = torch.argmax(regime_probs, dim=1)  # (batch_size,)
        
        # 레짐별 임베딩
        regime_embeds = self.regime_embeddings(regime_indices)  # (batch_size, hidden_dim//4)
        
        return {
            'regime_probabilities': regime_probs,
            'regime_indices': regime_indices,
            'regime_embeddings': regime_embeds,
            'regime_confidence': torch.max(regime_probs, dim=1)[0]  # 최대 확률값
        }


class TemporalPatternFusion(nn.Module):
    """
    시간적 패턴 융합 모듈
    TimesNet과 Wavelet 특징을 시간 차원에서 융합
    """
    
    def __init__(
        self,
        timesnet_dim: int,
        wavelet_dim: int,
        output_dim: int,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.timesnet_dim = timesnet_dim
        self.wavelet_dim = wavelet_dim
        self.output_dim = output_dim
        
        # 차원 정렬
        self.timesnet_projection = nn.Linear(timesnet_dim, output_dim)
        self.wavelet_projection = nn.Linear(wavelet_dim, output_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Self-attention for temporal modeling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        timesnet_features: torch.Tensor,
        wavelet_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            timesnet_features: (batch_size, seq_len, timesnet_dim)
            wavelet_features: (batch_size, wavelet_dim) - Global features
            
        Returns:
            fused_features: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = timesnet_features.size()
        
        # 차원 정렬
        timesnet_proj = self.timesnet_projection(timesnet_features)  # (batch_size, seq_len, output_dim)
        wavelet_proj = self.wavelet_projection(wavelet_features)  # (batch_size, output_dim)
        
        # Wavelet features를 시퀀스 차원으로 확장
        wavelet_expanded = wavelet_proj.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, output_dim)
        
        # Cross-modal attention (TimesNet을 Query, Wavelet을 Key/Value)
        cross_attended, cross_weights = self.cross_attention(
            timesnet_proj, wavelet_expanded, wavelet_expanded
        )
        
        # Self-attention for temporal dependencies
        temporal_attended, temporal_weights = self.temporal_attention(
            cross_attended, cross_attended, cross_attended
        )
        
        # Gated fusion
        combined = torch.cat([cross_attended, temporal_attended], dim=-1)  # (batch_size, seq_len, output_dim*2)
        gate = self.fusion_gate(combined)  # (batch_size, seq_len, output_dim)
        
        # Apply gate
        gated_timesnet = gate * cross_attended
        gated_temporal = (1 - gate) * temporal_attended
        
        # Final fusion
        final_combined = torch.cat([gated_timesnet, gated_temporal], dim=-1)
        fused_features = self.final_projection(final_combined)
        
        return fused_features


class ScaleSpecificAnalyzer(nn.Module):
    """
    스케일별 특화 분석기
    서로 다른 시간 스케일에서의 패턴을 전문적으로 분석
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale_type: str = 'short'  # 'short', 'medium', 'long'
    ):
        super().__init__()
        
        self.scale_type = scale_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 스케일별 특화 레이어
        if scale_type == 'short':
            # 단기 패턴 (1-5분): 빠른 변화 감지
            self.analyzer = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
            
        elif scale_type == 'medium':
            # 중기 패턴 (5-30분): 트렌드 변화
            self.analyzer = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Conv1d(output_dim, output_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
            
        else:  # long
            # 장기 패턴 (30분+): 구조적 변화
            self.analyzer = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=15, padding=7),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Conv1d(output_dim, output_dim, kernel_size=15, padding=7),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_context = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim, seq_len)
            
        Returns:
            features: (batch_size, output_dim, seq_len)
        """
        # 스케일별 분석
        features = self.analyzer(x)  # (batch_size, output_dim, seq_len)
        
        # Global context attention
        global_feat = self.global_pool(features).squeeze(-1)  # (batch_size, output_dim)
        attention_weights = self.global_context(global_feat).unsqueeze(-1)  # (batch_size, output_dim, 1)
        
        # Apply attention
        attended_features = features * attention_weights
        
        return attended_features


class PatternAnalyzer(nn.Module):
    """
    Pattern Analyzer: TimesNet + Wavelet 통합 분석기
    
    다중 스케일 패턴 분석을 통해 SAC 환경의 상태 공간을 구성하는 핵심 모듈
    """
    
    def __init__(
        self,
        num_features: int = 30,  # TA-Lib 지표 수
        seq_len: int = 60,
        timesnet_config: Dict = None,
        wavelet_config: Dict = None,
        feature_dim: int = 64,
        integration_method: str = 'attention',
        use_market_regime: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.integration_method = integration_method
        self.use_market_regime = use_market_regime
        
        # Default configurations
        if timesnet_config is None:
            timesnet_config = {
                'seq_len': seq_len,
                'pred_len': 1,
                'top_k': 5,
                'd_model': 128,
                'd_ff': 256,
                'num_kernels': 6,
                'num_layers': 2,
                'dropout': 0.1
            }
        
        if wavelet_config is None:
            wavelet_config = {
                'wavelet': 'db4',
                'levels': 5,
                'feature_dim': feature_dim,
                'use_denoising': True
            }
        
        # TimesNet for multi-periodicity analysis
        self.timesnet = MultiVariateTimesNet(
            num_features=num_features,
            **timesnet_config
        )
        
        # Wavelet for multi-resolution analysis
        self.wavelet_extractors = nn.ModuleList([
            WaveletFeatureExtractor(**wavelet_config)
            for _ in range(min(num_features, 10))  # Extract wavelet features for top 10 indicators
        ])
        
        # Scale-specific analyzers
        timesnet_dim = timesnet_config['d_model']
        # Ensure the combined output matches timesnet_dim
        scale_output_dim = timesnet_dim // 3
        remaining_dim = timesnet_dim - (scale_output_dim * 2)  # Account for integer division
        self.short_analyzer = ScaleSpecificAnalyzer(timesnet_dim, scale_output_dim, 'short')
        self.medium_analyzer = ScaleSpecificAnalyzer(timesnet_dim, scale_output_dim, 'medium')
        self.long_analyzer = ScaleSpecificAnalyzer(timesnet_dim, remaining_dim, 'long')
        
        # Feature integration
        if integration_method == 'attention':
            self.temporal_fusion = TemporalPatternFusion(
                timesnet_dim=timesnet_dim,
                wavelet_dim=feature_dim,
                output_dim=feature_dim,
                num_heads=8
            )
            
        elif integration_method == 'concat':
            self.concat_projection = nn.Sequential(
                nn.Linear(timesnet_dim + feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
        elif integration_method == 'fusion':
            self.learned_fusion = nn.Sequential(
                nn.Linear(timesnet_dim + feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
        
        # Market regime detection
        if use_market_regime:
            regime_input_dim = timesnet_config['d_model'] + feature_dim  # TimesNet + Wavelet features
            regime_hidden_dim = 128  # MarketRegimeDetector's hidden_dim
            self.regime_detector = MarketRegimeDetector(regime_input_dim, regime_hidden_dim)
            regime_embed_dim = regime_hidden_dim // 4  # 128 // 4 = 32
        else:
            regime_embed_dim = 0
        
        # Final feature aggregation
        self.feature_aggregator = nn.Sequential(
            nn.Linear(feature_dim + regime_embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output heads for different purposes
        self.volatility_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # Volatility in [0, 1]
        )
        
        self.trend_strength_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Tanh()  # Trend strength in [-1, 1]
        )
        
        self.pattern_confidence_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )
        
    def forward(
        self,
        indicators_data: torch.Tensor,
        return_detailed: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            indicators_data: (batch_size, seq_len, num_features) - TA-Lib indicators
            return_detailed: Whether to return detailed intermediate features
            
        Returns:
            analysis_results: Dictionary containing pattern analysis results
        """
        batch_size, seq_len, num_features = indicators_data.size()
        
        # 1. TimesNet Analysis (Multi-periodicity)
        timesnet_features = self.timesnet(indicators_data)  # (batch_size, seq_len, d_model)
        
        # 2. Wavelet Analysis (Multi-resolution)
        wavelet_features = []
        for i, wavelet_extractor in enumerate(self.wavelet_extractors):
            if i < num_features:
                # Extract wavelet features for each indicator
                indicator_series = indicators_data[:, :, i]  # (batch_size, seq_len)
                wavelet_feat = wavelet_extractor(indicator_series)  # (batch_size, feature_dim)
                wavelet_features.append(wavelet_feat)
        
        # Average wavelet features across indicators
        if wavelet_features:
            avg_wavelet_features = torch.stack(wavelet_features, dim=1).mean(dim=1)  # (batch_size, feature_dim)
        else:
            avg_wavelet_features = torch.zeros(batch_size, self.feature_dim, device=indicators_data.device)
        
        # 3. Scale-specific Analysis
        timesnet_transposed = timesnet_features.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        short_features = self.short_analyzer(timesnet_transposed)  # (batch_size, feature_dim//3, seq_len)
        medium_features = self.medium_analyzer(timesnet_transposed)  # (batch_size, feature_dim//3, seq_len)
        long_features = self.long_analyzer(timesnet_transposed)  # (batch_size, feature_dim//3, seq_len)
        
        # Combine scale-specific features
        scale_features = torch.cat([short_features, medium_features, long_features], dim=1)  # (batch_size, feature_dim, seq_len)
        scale_features = scale_features.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        
        # 4. Feature Integration
        if self.integration_method == 'attention':
            integrated_features = self.temporal_fusion(
                scale_features, avg_wavelet_features
            )  # (batch_size, seq_len, feature_dim)
            
        elif self.integration_method == 'concat':
            # Expand wavelet features to sequence dimension
            wavelet_expanded = avg_wavelet_features.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([scale_features, wavelet_expanded], dim=-1)
            integrated_features = self.concat_projection(combined)
            
        elif self.integration_method == 'fusion':
            wavelet_expanded = avg_wavelet_features.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([scale_features, wavelet_expanded], dim=-1)
            integrated_features = self.learned_fusion(combined)
        
        # 5. Global feature aggregation
        global_features = torch.mean(integrated_features, dim=1)  # (batch_size, feature_dim)
        
        # 6. Market Regime Detection
        results = {'pattern_features': global_features}
        
        if self.use_market_regime:
            # Use both TimesNet and Wavelet features for regime detection
            regime_input = torch.cat([
                torch.mean(timesnet_features, dim=1),  # Global TimesNet features
                avg_wavelet_features  # Wavelet features
            ], dim=-1)
            
            regime_info = self.regime_detector(regime_input)
            results.update(regime_info)
            
            # Incorporate regime embeddings
            enhanced_features = torch.cat([
                global_features,
                regime_info['regime_embeddings']
            ], dim=-1)
            
            final_features = self.feature_aggregator(enhanced_features)
        else:
            final_features = global_features
        
        results['final_features'] = final_features
        
        # 7. Additional Pattern Metrics
        results['volatility'] = self.volatility_head(final_features).squeeze(-1)
        results['trend_strength'] = self.trend_strength_head(final_features).squeeze(-1)
        results['pattern_confidence'] = self.pattern_confidence_head(final_features).squeeze(-1)
        
        # 8. Detailed features (if requested)
        if return_detailed:
            results['detailed'] = {
                'timesnet_features': timesnet_features,
                'wavelet_features': avg_wavelet_features,
                'scale_features': {
                    'short': short_features,
                    'medium': medium_features,
                    'long': long_features
                },
                'integrated_features': integrated_features
            }
        
        return results
    
    def get_market_regime_features(self, pattern_features: torch.Tensor) -> torch.Tensor:
        """시장 레짐 특징 추출 (SAC 환경용)"""
        if self.use_market_regime:
            regime_input = torch.cat([pattern_features, pattern_features], dim=-1)  # Dummy expansion
            regime_info = self.regime_detector(regime_input)
            return regime_info['regime_probabilities']
        else:
            return torch.zeros(pattern_features.size(0), 4, device=pattern_features.device)
    
    def extract_sac_features(self, indicators_data: torch.Tensor) -> torch.Tensor:
        """SAC 환경을 위한 압축된 특징 추출"""
        results = self.forward(indicators_data, return_detailed=False)
        
        # Combine key features for SAC state space
        sac_features = torch.cat([
            results['final_features'],  # Main pattern features
            results['volatility'].unsqueeze(-1),  # Volatility
            results['trend_strength'].unsqueeze(-1),  # Trend strength
            results['pattern_confidence'].unsqueeze(-1),  # Pattern confidence
        ], dim=-1)
        
        if self.use_market_regime:
            sac_features = torch.cat([
                sac_features,
                results['regime_probabilities']  # Market regime probabilities
            ], dim=-1)
        
        return sac_features


def test_pattern_analyzer():
    """Pattern Analyzer 테스트"""
    print("Testing Pattern Analyzer...")
    
    # Test parameters
    batch_size, seq_len, num_features = 4, 60, 30
    device = torch.device('cpu')
    
    # Create test data
    indicators_data = torch.randn(batch_size, seq_len, num_features)
    
    # Create pattern analyzer
    pattern_analyzer = PatternAnalyzer(
        num_features=num_features,
        seq_len=seq_len,
        feature_dim=64,
        integration_method='attention',
        use_market_regime=True
    )
    
    # Test forward pass
    results = pattern_analyzer(indicators_data, return_detailed=True)
    
    print(f"Pattern features shape: {results['pattern_features'].shape}")
    print(f"Final features shape: {results['final_features'].shape}")
    print(f"Volatility shape: {results['volatility'].shape}")
    print(f"Trend strength shape: {results['trend_strength'].shape}")
    print(f"Pattern confidence shape: {results['pattern_confidence'].shape}")
    
    if 'regime_probabilities' in results:
        print(f"Regime probabilities shape: {results['regime_probabilities'].shape}")
        print(f"Regime confidence shape: {results['regime_confidence'].shape}")
    
    # Test SAC feature extraction
    sac_features = pattern_analyzer.extract_sac_features(indicators_data)
    print(f"SAC features shape: {sac_features.shape}")
    
    # Check output ranges
    assert torch.all(results['volatility'] >= 0) and torch.all(results['volatility'] <= 1)
    assert torch.all(results['trend_strength'] >= -1) and torch.all(results['trend_strength'] <= 1)
    assert torch.all(results['pattern_confidence'] >= 0) and torch.all(results['pattern_confidence'] <= 1)
    
    if 'regime_probabilities' in results:
        assert torch.allclose(results['regime_probabilities'].sum(dim=1), torch.ones(batch_size))
    
    print("Pattern Analyzer test passed!")


if __name__ == "__main__":
    test_pattern_analyzer() 