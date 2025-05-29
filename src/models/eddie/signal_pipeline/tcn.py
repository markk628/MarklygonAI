"""
Temporal Convolutional Network (TCN) Implementation
매도 강도 예측을 위한 시계열 특징 추출
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from typing import List, Optional, Tuple


class Chomp1d(nn.Module):
    """1D Convolution의 causal padding을 위한 모듈"""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN의 기본 블록: Dilated Convolution + Residual Connection"""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        activation: str = 'relu',
        use_norm: bool = True
    ):
        super(TemporalBlock, self).__init__()
        
        # 첫 번째 dilated convolution
        conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        if use_norm:
            self.conv1 = weight_norm(conv1)
        else:
            self.conv1 = conv1
            
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)
        
        # 두 번째 dilated convolution  
        conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        if use_norm:
            self.conv2 = weight_norm(conv2)
        else:
            self.conv2 = conv2
            
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
            
        self.init_weights()

    def init_weights(self):
        """가중치 초기화"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, seq_len)
        Returns:
            output: (batch_size, channels, seq_len)
        """
        # 첫 번째 convolution path
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        # 두 번째 convolution path
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.activation(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    시계열 데이터에서 장기 의존성을 효과적으로 학습
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        activation: str = 'relu',
        use_norm: bool = True
    ):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, ...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
                activation=activation,
                use_norm=use_norm
            )]

        self.network = nn.Sequential(*layers)
        self.output_size = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        Returns:
            output: (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class PositionalEncoding(nn.Module):
    """Transformer-style positional encoding for TCN"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            x + positional_encoding: (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class EnhancedTCN(nn.Module):
    """
    Enhanced TCN with additional features:
    - Global context modeling
    - Multi-scale feature extraction
    - Feature normalization
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: str = 'relu',
        use_norm: bool = True,
        use_global_context: bool = True,
        use_multi_scale: bool = True
    ):
        super(EnhancedTCN, self).__init__()
        
        self.use_global_context = use_global_context
        self.use_multi_scale = use_multi_scale
        
        # Main TCN
        self.tcn = TCN(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            use_norm=use_norm
        )
        
        self.output_size = num_channels[-1]
        
        # Multi-scale TCN branches (different kernel sizes)
        if use_multi_scale:
            self.tcn_small = TCN(
                num_inputs=num_inputs,
                num_channels=[ch // 2 for ch in num_channels],
                kernel_size=2,
                dropout=dropout,
                activation=activation,
                use_norm=use_norm
            )
            
            self.tcn_large = TCN(
                num_inputs=num_inputs,
                num_channels=[ch // 2 for ch in num_channels],
                kernel_size=5,
                dropout=dropout,
                activation=activation,
                use_norm=use_norm
            )
            
            # Feature fusion
            fusion_input_size = self.output_size + (num_channels[-1] // 2) * 2
            self.fusion = nn.Conv1d(fusion_input_size, self.output_size, 1)
        
        # Global context modeling
        if use_global_context:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.global_mlp = nn.Sequential(
                nn.Linear(self.output_size, self.output_size // 4),
                nn.ReLU(),
                nn.Linear(self.output_size // 4, self.output_size),
                nn.Sigmoid()
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
        Returns:
            output: (batch_size, output_size, seq_len)
        """
        # Main TCN branch
        main_out = self.tcn(x)
        
        if self.use_multi_scale:
            # Multi-scale branches
            small_out = self.tcn_small(x)
            large_out = self.tcn_large(x)
            
            # Concatenate and fuse
            combined = torch.cat([main_out, small_out, large_out], dim=1)
            fused = self.fusion(combined)
        else:
            fused = main_out
        
        # Global context attention
        if self.use_global_context:
            # Global average pooling
            global_context = self.global_pool(fused)  # (batch_size, channels, 1)
            global_context = global_context.squeeze(-1)  # (batch_size, channels)
            
            # Global attention weights
            attention_weights = self.global_mlp(global_context)  # (batch_size, channels)
            attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, channels, 1)
            
            # Apply attention
            fused = fused * attention_weights
        
        # Layer normalization (transpose for LayerNorm)
        fused = fused.transpose(1, 2)  # (batch_size, seq_len, channels)
        fused = self.layer_norm(fused)
        fused = fused.transpose(1, 2)  # (batch_size, channels, seq_len)
        
        return fused


class TCNWithUncertainty(nn.Module):
    """
    TCN with uncertainty estimation using Monte Carlo Dropout
    매도 강도 예측과 함께 모델의 불확실성도 추정
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: str = 'relu',
        use_norm: bool = True,
        uncertainty_samples: int = 10
    ):
        super(TCNWithUncertainty, self).__init__()
        
        self.uncertainty_samples = uncertainty_samples
        
        self.tcn = EnhancedTCN(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            use_norm=use_norm
        )
        
        # Output heads
        self.mean_head = nn.Conv1d(self.tcn.output_size, 1, 1)
        self.log_var_head = nn.Conv1d(self.tcn.output_size, 1, 1)
        
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_inputs, seq_len)
            return_uncertainty: Whether to return uncertainty estimation
        Returns:
            If return_uncertainty=False: (batch_size, 1, seq_len) - mean prediction
            If return_uncertainty=True: (mean, uncertainty) tuple
        """
        features = self.tcn(x)
        
        if not return_uncertainty:
            # Training mode: single forward pass
            mean = self.mean_head(features)
            return torch.tanh(mean)  # Sell intensity in [-1, 1]
        
        else:
            # Inference mode: Monte Carlo sampling for uncertainty
            self.train()  # Enable dropout for uncertainty estimation
            
            predictions = []
            for _ in range(self.uncertainty_samples):
                features_sample = self.tcn(x)
                mean_sample = self.mean_head(features_sample)
                predictions.append(torch.tanh(mean_sample))
            
            # Stack predictions
            predictions = torch.stack(predictions, dim=0)  # (samples, batch_size, 1, seq_len)
            
            # Calculate mean and uncertainty
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)
            
            return mean_pred, uncertainty

    def forward_with_variance(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit variance prediction
        
        Returns:
            mean: (batch_size, 1, seq_len)
            log_var: (batch_size, 1, seq_len)
        """
        features = self.tcn(x)
        
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        
        return torch.tanh(mean), log_var


def test_tcn():
    """TCN 모델 테스트"""
    print("Testing TCN models...")
    
    # Test data
    batch_size, num_features, seq_len = 4, 30, 60
    x = torch.randn(batch_size, num_features, seq_len)
    
    # Test basic TCN
    tcn = TCN(
        num_inputs=num_features,
        num_channels=[64, 128, 64],
        kernel_size=3,
        dropout=0.2
    )
    
    output = tcn(x)
    print(f"Basic TCN output shape: {output.shape}")
    assert output.shape == (batch_size, 64, seq_len)
    
    # Test Enhanced TCN
    enhanced_tcn = EnhancedTCN(
        num_inputs=num_features,
        num_channels=[64, 128, 64],
        kernel_size=3,
        use_global_context=True,
        use_multi_scale=True
    )
    
    output = enhanced_tcn(x)
    print(f"Enhanced TCN output shape: {output.shape}")
    assert output.shape == (batch_size, 64, seq_len)
    
    # Test TCN with Uncertainty
    tcn_uncertainty = TCNWithUncertainty(
        num_inputs=num_features,
        num_channels=[64, 128, 64],
        uncertainty_samples=5
    )
    
    # Training mode
    output_train = tcn_uncertainty(x, return_uncertainty=False)
    print(f"TCN uncertainty (training) output shape: {output_train.shape}")
    assert output_train.shape == (batch_size, 1, seq_len)
    
    # Inference mode with uncertainty
    mean, uncertainty = tcn_uncertainty(x, return_uncertainty=True)
    print(f"TCN uncertainty (inference) mean shape: {mean.shape}")
    print(f"TCN uncertainty (inference) uncertainty shape: {uncertainty.shape}")
    assert mean.shape == (batch_size, 1, seq_len)
    assert uncertainty.shape == (batch_size, 1, seq_len)
    
    print("All TCN tests passed!")


if __name__ == "__main__":
    test_tcn() 