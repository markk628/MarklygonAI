"""
Multi-Head Attention Implementation for Signal Generation
TCN 출력에서 중요한 시간 포인트를 식별하여 매도 강도 예측 성능 향상
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time series
    시계열 데이터의 시간적 위치 정보를 인코딩
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for sinusoidal patterns
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            x + positional encoding: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for time series
    시계열에서 중요한 시간 포인트 간의 관계를 학습
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.init_weights()
    
    def init_weights(self):
        """Xavier initialization for better convergence"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def scaled_dot_product_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention mechanism
        
        Args:
            query: (batch_size, num_heads, seq_len, d_k)
            key: (batch_size, num_heads, seq_len, d_k)  
            value: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, seq_len, seq_len) or None
            
        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len) or None
            return_attention: Whether to return attention weights
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, d_model = x.size()
        
        # Store residual for skip connection
        residual = x
        
        # Linear projections and reshape for multi-head attention
        query = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)
        
        if return_attention:
            return output, attention_weights
        return output


class TemporalMultiHeadAttention(nn.Module):
    """
    Temporal-aware Multi-Head Attention
    시계열 특성을 고려한 어텐션 (최근 데이터에 높은 가중치)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        temporal_decay: float = 0.1,
        use_relative_position: bool = True
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.temporal_decay = temporal_decay
        self.use_relative_position = use_relative_position
        
        if use_relative_position:
            # Learnable relative position embeddings
            self.relative_pos_embed = nn.Embedding(512, d_model)
        
    def create_temporal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create temporal decay mask (recent timesteps have higher weights)
        
        Returns:
            mask: (1, 1, seq_len, seq_len)
        """
        # Create distance matrix
        positions = torch.arange(seq_len, device=device).float()
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        
        # Apply exponential decay (recent positions have higher weights)
        temporal_weights = torch.exp(-self.temporal_decay * distance_matrix)
        
        return temporal_weights.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Add relative positional encoding
        if self.use_relative_position:
            pos_ids = torch.arange(seq_len, device=x.device)
            pos_embed = self.relative_pos_embed(pos_ids)
            x = x + pos_embed.unsqueeze(0)
        
        # Create temporal mask
        temporal_mask = self.create_temporal_mask(seq_len, x.device)
        
        # Apply attention with temporal mask
        return self.attention(x, mask=temporal_mask, return_attention=return_attention)


class CrossTimeAttention(nn.Module):
    """
    Cross-Time Attention for comparing different time periods
    서로 다른 시간 구간 간의 패턴 비교 (예: 아침 vs 오후 패턴)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        time_window: int = 60,  # 60분 윈도우
        num_windows: int = 4,   # 4개 구간으로 분할
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.time_window = time_window
        self.num_windows = num_windows
        self.window_size = time_window // num_windows
        
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.window_embed = nn.Embedding(num_windows, d_model)
        
        # Window-specific transformations
        self.window_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_windows)
        ])
        
        self.fusion = nn.Linear(d_model * num_windows, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        original_seq_len = seq_len
        
        # Ensure sequence length is compatible with time window
        if seq_len < self.time_window:
            # Pad if sequence is shorter than expected
            pad_length = self.time_window - seq_len
            x = F.pad(x, (0, 0, 0, pad_length))
            seq_len = self.time_window
        elif seq_len > self.time_window:
            # Truncate if sequence is longer
            x = x[:, :self.time_window, :]
            seq_len = self.time_window
        
        # Calculate window size and ensure it divides evenly
        window_size = self.time_window // self.num_windows
        effective_length = window_size * self.num_windows
        
        # Use only the effective length
        x_windowed = x[:, :effective_length, :]
        
        # Reshape into windows
        x_windows = x_windowed.view(
            batch_size, self.num_windows, window_size, d_model
        )
        
        # Add window embeddings
        window_ids = torch.arange(self.num_windows, device=x.device)
        window_embeds = self.window_embed(window_ids)  # (num_windows, d_model)
        
        # Process each window
        window_outputs = []
        for i in range(self.num_windows):
            window_data = x_windows[:, i, :, :]  # (batch_size, window_size, d_model)
            window_data = window_data + window_embeds[i].unsqueeze(0).unsqueeze(0)
            
            # Apply window-specific projection
            window_data = self.window_projections[i](window_data)
            
            # Apply self-attention within window
            window_output = self.cross_attention(window_data)
            
            # Global pooling for window representation
            window_repr = window_output.mean(dim=1)  # (batch_size, d_model)
            window_outputs.append(window_repr)
        
        # Cross-window attention
        window_stack = torch.stack(window_outputs, dim=1)  # (batch_size, num_windows, d_model)
        cross_attended = self.cross_attention(window_stack)
        
        # Fuse window representations
        fused = cross_attended.view(batch_size, -1)  # (batch_size, num_windows * d_model)
        output_repr = self.fusion(fused)  # (batch_size, d_model)
        
        # Broadcast to original sequence length
        output = output_repr.unsqueeze(1).expand(-1, original_seq_len, -1)
        
        return output


class AdaptiveAttention(nn.Module):
    """
    Adaptive Attention that adjusts focus based on market conditions
    시장 상황에 따라 어텐션 패턴을 적응적으로 조정
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_market_states: int = 4,  # 4가지 시장 상태 (상승, 하락, 횡보, 변동성)
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_market_states = num_market_states
        
        # Market state classifier
        self.market_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_market_states),
            nn.Softmax(dim=-1)
        )
        
        # State-specific attention modules
        self.state_attentions = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_market_states)
        ])
        
        # State-specific temperature parameters
        self.state_temperatures = nn.Parameter(torch.ones(num_market_states))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, input_d_model = x.size()
        
        # Debug: Check input dimensions
        if input_d_model != self.d_model:
            # If input d_model doesn't match expected, create a projection layer
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(input_d_model, self.d_model).to(x.device)
            x = self.input_projection(x)
        
        # Classify market state based on global representation
        global_repr = x.mean(dim=1)  # (batch_size, self.d_model)
        
        market_probs = self.market_classifier(global_repr)  # (batch_size, num_market_states)
        
        # Apply state-specific attention
        state_outputs = []
        for i, attention in enumerate(self.state_attentions):
            # Modify attention temperature based on market state
            attention.temperature = self.state_temperatures[i].item()
            state_output = attention(x)
            state_outputs.append(state_output)
        
        # Weighted combination based on market state probabilities
        output = torch.zeros_like(x)
        for i, state_output in enumerate(state_outputs):
            weight = market_probs[:, i].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
            output += weight * state_output
        
        return output


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence summarization
    시퀀스를 고정 크기 벡터로 요약
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) - 1 for valid positions, 0 for padding
            
        Returns:
            pooled: (batch_size, d_model)
        """
        # Calculate attention weights
        attention_weights = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        return pooled


def test_attention_modules():
    """어텐션 모듈들 테스트"""
    print("Testing attention modules...")
    
    batch_size, seq_len, d_model = 4, 60, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test MultiHeadAttention
    mha = MultiHeadAttention(d_model=d_model, num_heads=8)
    output = mha(x)
    print(f"MultiHeadAttention output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test TemporalMultiHeadAttention
    tmha = TemporalMultiHeadAttention(d_model=d_model, num_heads=8)
    output = tmha(x)
    print(f"TemporalMultiHeadAttention output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test CrossTimeAttention
    cta = CrossTimeAttention(d_model=d_model, num_heads=8, time_window=60)
    output = cta(x)
    print(f"CrossTimeAttention output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test AdaptiveAttention
    ada = AdaptiveAttention(d_model=d_model, num_heads=8)
    output = ada(x)
    print(f"AdaptiveAttention output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Test AttentionPooling
    pool = AttentionPooling(d_model=d_model)
    output = pool(x)
    print(f"AttentionPooling output shape: {output.shape}")
    assert output.shape == (batch_size, d_model)
    
    print("All attention tests passed!")


if __name__ == "__main__":
    test_attention_modules() 