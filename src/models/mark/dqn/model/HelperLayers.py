import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionLayer(nn.Module):
    def __init__(self, temporal_feature_size, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(temporal_feature_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(temporal_feature_size)
        
    def forward(self, temporal_features):
        # temporal_features: (batch_size, num_temporal_components, feature_dim)
        attended_features, attention_weights = self.attention(
            temporal_features, temporal_features, temporal_features
        )
        # Residual connection + layer norm
        output = self.layer_norm(attended_features + temporal_features)
        return output, attention_weights

class CrossModalAttentionLayer(nn.Module):
    """Attention layer for cross-modal feature interaction"""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query_features, key_value_features):
        # query_features: (batch_size, 1, feature_dim)
        # key_value_features: (batch_size, num_features, feature_dim)
        attended_features, attention_weights = self.attention(
            query_features, key_value_features, key_value_features
        )
        output = self.layer_norm(attended_features + query_features)
        return output.squeeze(1), attention_weights

class SelfAttentionLayer(nn.Module):
    """Self-attention for feature refinement"""
    def __init__(self, feature_dim, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, features):
        # features: (batch_size, sequence_length, feature_dim)
        attended_features, attention_weights = self.attention(features, features, features)
        # Residual connection + layer norm
        attended_features = self.layer_norm(attended_features + features)
        # Feed forward with residual connection
        ff_output = self.feed_forward(attended_features)
        output = self.layer_norm(ff_output + attended_features)
        return output, attention_weights
    
# Additional activation functions
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))