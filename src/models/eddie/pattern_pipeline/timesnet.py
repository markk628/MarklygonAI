"""
TimesNet: A General Time Series Analysis Framework
다중 주기성 시계열 분석을 위한 TimesNet 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from typing import List, Tuple, Optional, Dict
import math


class TimesBlock(nn.Module):
    """
    TimesNet의 핵심 블록
    1D 시계열을 2D 이미지로 변환하여 CNN으로 패턴 추출
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int = 6
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        
        # 1D → 2D 변환을 위한 컨볼루션
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_ff,
                     kernel_size=[1, 1], bias=False),
            nn.BatchNorm2d(d_ff),
            nn.ReLU()
        )
        
        # 다양한 크기의 커널로 다중 스케일 특징 추출
        self.inception_blocks = nn.ModuleList([
            InceptionBlock(d_ff) for _ in range(num_kernels)
        ])
        
        # 2D → 1D 변환
        self.projection = nn.Conv2d(
            in_channels=d_ff * num_kernels, 
            out_channels=1,
            kernel_size=[1, 1], 
            bias=False
        )
        
        # 시퀀스 길이 조정
        if self.seq_len % self.pred_len != 0:
            self.padding = nn.ReplicationPad1d((0, self.pred_len - self.seq_len % self.pred_len))
        else:
            self.padding = None
            
        # 주기성 가중치
        self.period_weights = nn.Parameter(torch.ones(top_k))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 1. 주기성 분석
        period_list, period_weights = self.analyze_periods(x)
        
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            
            # 1D → 2D 변환
            x_2d = self.time_to_image(x, period)
            
            # CNN 특징 추출
            x_2d = self.conv(x_2d.unsqueeze(1))  # Add channel dimension
            
            # Inception blocks 적용
            inception_outputs = []
            for inception in self.inception_blocks:
                inception_outputs.append(inception(x_2d))
            
            # 모든 inception 출력 결합
            x_2d = torch.cat(inception_outputs, dim=1)
            
            # 2D → 1D 변환
            x_2d = self.projection(x_2d).squeeze(1)
            
            # 2D → 1D 복원
            x_1d = self.image_to_time(x_2d, period, seq_len)
            
            res.append(x_1d * period_weights[i])
        
        # 가중 합산
        res = torch.stack(res, dim=-1)
        
        # 학습 가능한 주기 가중치 적용
        period_weights = F.softmax(self.period_weights, dim=0)
        res = torch.sum(res * period_weights.view(1, 1, 1, -1), dim=-1)
        
        return res
    
    def analyze_periods(self, x: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        FFT를 사용한 주기성 분석
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            period_list: 상위 K개 주기
            period_weights: 각 주기의 가중치
        """
        batch_size, seq_len, d_model = x.size()
        
        # 평균을 제거하여 DC 성분 제거
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        
        # FFT 계산 (마지막 차원에 대해)
        x_fft = fft.rfft(x_centered, dim=1)
        
        # 파워 스펙트럼 계산
        power_spectrum = torch.abs(x_fft).mean(dim=-1)  # (batch_size, freq_bins)
        power_spectrum = power_spectrum.mean(dim=0)  # (freq_bins,)
        
        # DC 성분 제거
        power_spectrum[0] = 0
        
        # 상위 K개 주파수 찾기
        _, top_indices = torch.topk(power_spectrum, self.top_k)
        
        # 주파수를 주기로 변환
        period_list = []
        period_weights_list = []
        
        for idx in top_indices:
            if idx.item() == 0:
                period = seq_len  # Avoid division by zero
            else:
                period = seq_len // idx.item()
            
            # 최소/최대 주기 제한
            period = max(2, min(period, seq_len // 2))
            period_list.append(period)
            period_weights_list.append(power_spectrum[idx].item())
        
        # 가중치 정규화
        period_weights = torch.tensor(period_weights_list, device=x.device)
        period_weights = F.softmax(period_weights, dim=0)
        
        return period_list, period_weights
    
    def time_to_image(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """
        1D 시계열을 2D 이미지로 변환
        
        Args:
            x: (batch_size, seq_len, d_model)
            period: 주기 길이
            
        Returns:
            x_2d: (batch_size, height, width)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 패딩 적용 (필요한 경우)
        if self.padding is not None:
            x = self.padding(x.transpose(1, 2)).transpose(1, 2)
            seq_len = x.size(1)
        
        # Reshape to 2D: (batch_size * d_model, period, seq_len // period)
        if seq_len % period != 0:
            # 주기로 나누어떨어지지 않는 경우 패딩
            pad_len = period - (seq_len % period)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.size(1)
        
        height = seq_len // period
        x_2d = x.view(batch_size, height, period, d_model)
        
        # 차원별로 평균내어 단일 채널로 변환
        x_2d = x_2d.mean(dim=-1)  # (batch_size, height, period)
        
        return x_2d
    
    def image_to_time(self, x_2d: torch.Tensor, period: int, target_len: int) -> torch.Tensor:
        """
        2D 이미지를 1D 시계열로 복원
        
        Args:
            x_2d: (batch_size, height, width)
            period: 주기 길이
            target_len: 목표 시퀀스 길이
            
        Returns:
            x: (batch_size, target_len, d_model)
        """
        batch_size, height, width = x_2d.size()
        
        # 1D로 flatten
        x_1d = x_2d.view(batch_size, height * width)
        
        # 목표 길이에 맞게 조정
        current_len = x_1d.size(1)
        
        if current_len < target_len:
            # 부족한 경우 반복 패딩
            repeat_times = (target_len // current_len) + 1
            x_1d = x_1d.repeat(1, repeat_times)
        
        # 정확한 길이로 자르기
        x_1d = x_1d[:, :target_len]
        
        # 차원 확장 (d_model=1로 가정)
        x_1d = x_1d.unsqueeze(-1)
        
        return x_1d


class InceptionBlock(nn.Module):
    """
    Inception-style block for multi-scale feature extraction
    """
    
    def __init__(self, d_ff: int):
        super().__init__()
        
        # 1x1 conv
        self.conv1x1 = nn.Conv2d(d_ff, d_ff // 4, kernel_size=1, bias=False)
        
        # 3x3 conv
        self.conv3x3_reduce = nn.Conv2d(d_ff, d_ff // 4, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(d_ff // 4, d_ff // 4, kernel_size=3, padding=1, bias=False)
        
        # 5x5 conv
        self.conv5x5_reduce = nn.Conv2d(d_ff, d_ff // 4, kernel_size=1, bias=False)
        self.conv5x5 = nn.Conv2d(d_ff // 4, d_ff // 4, kernel_size=5, padding=2, bias=False)
        
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(d_ff, d_ff // 4, kernel_size=1, bias=False)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, d_ff, height, width)
            
        Returns:
            output: (batch_size, d_ff, height, width)
        """
        # 1x1 conv
        branch1 = self.conv1x1(x)
        
        # 3x3 conv
        branch2 = self.conv3x3_reduce(x)
        branch2 = self.conv3x3(branch2)
        
        # 5x5 conv
        branch3 = self.conv5x5_reduce(x)
        branch3 = self.conv5x5(branch3)
        
        # Max pooling
        branch4 = self.maxpool(x)
        branch4 = self.conv_pool(branch4)
        
        # Concatenate all branches
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        output = self.bn(output)
        output = F.relu(output)
        
        return output


class TimesNet(nn.Module):
    """
    Complete TimesNet model for time series analysis
    """
    
    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 1,
        top_k: int = 5,
        d_model: int = 128,
        d_ff: int = 256,
        num_kernels: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)  # Assume univariate input
        
        # TimesNet layers
        self.times_blocks = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, 1) - Univariate time series
            
        Returns:
            output: (batch_size, seq_len, d_model) - Feature representations
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Apply TimesNet layers
        for i, (times_block, layer_norm) in enumerate(zip(self.times_blocks, self.layer_norms)):
            # Residual connection
            residual = x
            
            # TimesNet block
            x = times_block(x)
            
            # Dropout and residual connection
            x = self.dropout(x) + residual
            
            # Layer normalization
            x = layer_norm(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (optional, for sequence prediction tasks)
        
        Args:
            x: (batch_size, seq_len, 1)
            
        Returns:
            predictions: (batch_size, pred_len, 1)
        """
        features = self.forward(x)  # (batch_size, seq_len, d_model)
        
        # Use last few timesteps for prediction
        pred_features = features[:, -self.pred_len:, :]
        predictions = self.output_projection(pred_features)
        
        return predictions


class MultiVariateTimesNet(nn.Module):
    """
    Multi-variate TimesNet for multiple indicators
    """
    
    def __init__(
        self,
        num_features: int,
        seq_len: int = 60,
        pred_len: int = 1,
        top_k: int = 5,
        d_model: int = 128,
        d_ff: int = 256,
        num_kernels: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Feature-wise TimesNet
        self.feature_timesnets = nn.ModuleList([
            TimesNet(seq_len, pred_len, top_k, d_model, d_ff, num_kernels, num_layers, dropout)
            for _ in range(num_features)
        ])
        
        # Cross-feature attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * num_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_features)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, num_features = x.size()
        
        # Process each feature separately
        feature_outputs = []
        for i in range(min(num_features, len(self.feature_timesnets))):
            feature_data = x[:, :, i:i+1]  # (batch_size, seq_len, 1)
            feature_output = self.feature_timesnets[i](feature_data)  # (batch_size, seq_len, d_model)
            feature_outputs.append(feature_output)
        
        # Stack feature outputs
        stacked_features = torch.stack(feature_outputs, dim=2)  # (batch_size, seq_len, actual_num_features, d_model)
        
        # Cross-feature attention for each timestep
        attended_features = []
        for t in range(seq_len):
            timestep_features = stacked_features[:, t, :, :]  # (batch_size, num_features, d_model)
            
            # Self-attention across features
            attended, _ = self.cross_attention(
                timestep_features, timestep_features, timestep_features
            )
            
            attended_features.append(attended)
        
        # Reconstruct sequence dimension
        attended_sequence = torch.stack(attended_features, dim=1)  # (batch_size, seq_len, num_features, d_model)
        
        # Flatten and fuse features
        actual_num_features = len(feature_outputs)
        attended_flat = attended_sequence.view(batch_size, seq_len, -1)  # (batch_size, seq_len, actual_num_features * d_model)
        
        # Handle dimension mismatch
        expected_input_dim = self.feature_fusion[0].in_features
        actual_input_dim = actual_num_features * self.d_model
        
        if actual_input_dim != expected_input_dim:
            # Create a projection layer if dimensions don't match
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(actual_input_dim, expected_input_dim).to(attended_flat.device)
            attended_flat = self.input_projection(attended_flat)
        
        fused_output = self.feature_fusion(attended_flat)  # (batch_size, seq_len, d_model)
        
        return fused_output


def test_timesnet():
    """TimesNet 모델 테스트"""
    print("Testing TimesNet models...")
    
    # Test parameters
    batch_size, seq_len, num_features = 4, 60, 30
    
    # Test univariate TimesNet
    x_uni = torch.randn(batch_size, seq_len, 1)
    
    timesnet = TimesNet(
        seq_len=seq_len,
        pred_len=1,
        top_k=5,
        d_model=64,
        d_ff=128,
        num_layers=2
    )
    
    output_uni = timesnet(x_uni)
    print(f"Univariate TimesNet output shape: {output_uni.shape}")
    assert output_uni.shape == (batch_size, seq_len, 64)
    
    # Test prediction
    pred_uni = timesnet.predict(x_uni)
    print(f"Univariate TimesNet prediction shape: {pred_uni.shape}")
    assert pred_uni.shape == (batch_size, 1, 1)
    
    # Test multivariate TimesNet
    x_multi = torch.randn(batch_size, seq_len, num_features)
    
    mv_timesnet = MultiVariateTimesNet(
        num_features=num_features,
        seq_len=seq_len,
        d_model=64,
        d_ff=128,
        num_layers=2
    )
    
    output_multi = mv_timesnet(x_multi)
    print(f"Multivariate TimesNet output shape: {output_multi.shape}")
    assert output_multi.shape == (batch_size, seq_len, 64)
    
    print("All TimesNet tests passed!")


if __name__ == "__main__":
    test_timesnet() 