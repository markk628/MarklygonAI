import torch
import torch.nn as nn

from src.config.config import BATCH_SIZE
from src.models.mark.dqn.model.HelperLayers import *


class HierarchicalTradingDQNNetwork(nn.Module):
    def __init__(self, sizes: dict[str, int], use_dueling: bool=True):
        super(HierarchicalTradingDQNNetwork, self).__init__()
        
        self.stock_data_window_size = sizes['stock_data_window_size']
        self.stock_data_feature_size = sizes['stock_data_feature_size']
        self.portfolio_metrics_size = sizes['portfolio_metrics_size']
        self.performance_metrics_size = sizes['performance_metrics_size']
        self.risk_metrics_size = sizes['risk_metrics_size']
        self.price_action_metrics_size = sizes['price_action_metrics_size']
        self.position_management_metrics_size = sizes['position_management_metrics_size']
        self.trading_behavior_metrics_size = sizes['trading_behavior_metrics_size']
        self.temporal_metrics_size = sizes['temporal_metrics_size']
        self.temporal_metrics_types_count = sizes['temporal_metrics_types_count']
        self.action_size = sizes['action_size']
        self.use_dueling = use_dueling
        
        # input dim for conv1d: (batch_size, in_channels, sequence_length)
        # in_channels: tock_data_feature_size (22)
        # sequence_length: stock_data_window_size (60)
        self.stock_data_conv1d_branch = nn.Sequential(
            # capture short-term patterns conv layer 
            nn.Conv1d(in_channels=self.stock_data_feature_size, 
                      out_channels=64, 
                      kernel_size=3, 
                      padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(alpha=1.0),
            nn.Dropout(0.15),
            
            # capture medium-term patterns conv layer
            nn.Conv1d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=5, 
                      padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(alpha=1.0),
            nn.Dropout(0.2),
            
            # capture longer-term  conv layer
            nn.Conv1d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=7, 
                      padding=3),
            nn.BatchNorm1d(256),
            nn.ELU(alpha=1.0),
            nn.Dropout(0.25),
        )
        
        # self-attention for temporal patterns in stock data
        self.stock_data_attention = SelfAttentionLayer(256, num_heads=4)
        
        # final pooling and linear processing
        self.stock_data_pool = nn.AdaptiveAvgPool1d(1)
        self.processed_stock_data_size = 256 
        
        self.stock_data_linear_branch = nn.Sequential(
            nn.Linear(self.processed_stock_data_size, 128),
            nn.LayerNorm(128),
            nn.ELU(alpha=1.0),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(alpha=1.0),
            nn.Dropout(0.1)
        )
        
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.portfolio_metrics_size, 32),
            nn.LayerNorm(32),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
            nn.LayerNorm(24),
            Swish(),
            nn.Dropout(0.05),
            nn.Linear(24, 16)
        )
        
        self.performance_branch = nn.Sequential(
            nn.Linear(self.performance_metrics_size, 48),
            nn.LayerNorm(48),
            nn.Mish(),
            nn.Dropout(0.15),
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(32, 16)
        )
        
        self.risk_branch = nn.Sequential(
            nn.Linear(self.risk_metrics_size, 32),
            nn.LayerNorm(32),
            nn.SELU(),
            nn.AlphaDropout(0.1),
            nn.Linear(32, 24),
            nn.LayerNorm(24),
            nn.SELU(),
            nn.AlphaDropout(0.05),
            nn.Linear(24, 16)
        )
        
        self.price_action_branch = nn.Sequential(
            nn.Linear(self.price_action_metrics_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(16, 8)
        )
        
        self.position_management_branch = nn.Sequential(
            nn.Linear(self.position_management_metrics_size, 28),
            nn.LayerNorm(28),
            Swish(),
            nn.Dropout(0.15),
            nn.Linear(28, 16),
            nn.LayerNorm(16),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(16, 8)
        )
        
        self.trading_behavior_branch = nn.Sequential(
            nn.Linear(self.trading_behavior_metrics_size, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8)
        )
        
        # temporal branches
        # project temporal features to common dimension
        self.temporal_feature_dim = 4
        self.micro_timing_projection = nn.Linear(self.temporal_metrics_size, self.temporal_feature_dim)
        self.intraday_timing_projection = nn.Linear(self.temporal_metrics_size, self.temporal_feature_dim)
        self.weekly_timing_projection = nn.Linear(self.temporal_metrics_size, self.temporal_feature_dim)
        
        # temporal attention layer
        self.temporal_attention = TemporalAttentionLayer(self.temporal_feature_dim, num_heads=2)
        
        # final temporal processing
        self.temporal_output_projection = nn.Sequential(
            nn.Linear(self.temporal_feature_dim * self.temporal_metrics_types_count, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(16, 12)
        )
        
        # cross modal attention branches
        # project features to common dimension
        self.feature_dim = 16
        self.stock_data_proj = nn.Linear(64, self.feature_dim)
        self.portfolio_proj = nn.Linear(16, self.feature_dim)
        self.performance_proj = nn.Linear(16, self.feature_dim)
        self.risk_proj = nn.Linear(16, self.feature_dim)
        self.price_action_proj = nn.Linear(8, self.feature_dim)
        self.position_mgmt_proj = nn.Linear(8, self.feature_dim)
        self.trading_behavior_proj = nn.Linear(8, self.feature_dim)
        
        # cross-modal attention
        # stock data attending to other features
        self.stock_cross_attention = CrossModalAttentionLayer(self.feature_dim, num_heads=4)
        
        # feature fusion attention
        self.feature_fusion_attention = SelfAttentionLayer(self.feature_dim, num_heads=4)
        
        # stock data (64) + attended stock features (16) + temporal features (12) + 6 other features (16 each)
        combined_feature_size = 64 + 16 + 12 + (6 * 16)
        
        if use_dueling:
            # value stream: estimates the state value V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(combined_feature_size, 128),
                nn.LayerNorm(128),
                nn.Mish(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

            # advantage stream: estimates the advantage A(s, a) for each action
            self.advantage_stream = nn.Sequential(
                nn.Linear(combined_feature_size, 128),
                nn.LayerNorm(128),
                nn.Mish(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(32, self.action_size)
            )
        else:
            # action layer
            self.action_layer = nn.Sequential(
                nn.Linear(combined_feature_size, 256),
                nn.LayerNorm(256),
                nn.Mish(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.Mish(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.Mish(),
                nn.Dropout(0.05),
                nn.Linear(32, self.action_size)
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization for different layer types"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # indices for state components
        portfolio_start_idx = self.stock_data_window_size * self.stock_data_feature_size
        performance_start_idx = portfolio_start_idx + self.portfolio_metrics_size
        risk_start_idx = performance_start_idx + self.performance_metrics_size
        price_action_start_idx = risk_start_idx + self.risk_metrics_size
        position_management_start_idx = price_action_start_idx + self.price_action_metrics_size
        trading_behavior_start_idx = position_management_start_idx + self.position_management_metrics_size
        micro_timing_start_idx = trading_behavior_start_idx + self.trading_behavior_metrics_size
        intraday_timing_start_idx = micro_timing_start_idx + self.temporal_metrics_size
        weekly_timing_start_idx = intraday_timing_start_idx + self.temporal_metrics_size
        
        # process stock data with temporal attention
        # extract and reshape market data, then process through covn1d -> self attention layer
        # input dim (batch_size, channels, seq_len)
        # output dim (batch_size, processed_market_data_size, 1)
        stock_data_flat = x[:, :portfolio_start_idx]
        stock_data_reshaped = stock_data_flat.view(-1, self.stock_data_feature_size, self.stock_data_window_size)
        conv_output = self.stock_data_conv1d_branch(stock_data_reshaped)
        
        # apply self-attention to temporal patterns
        # reshape for attention (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        conv_for_attention = conv_output.transpose(1, 2)
        attended_conv, _ = self.stock_data_attention(conv_for_attention)
        
        # pool and process
        # reshape for pooling (batch_size, seq_len, channels) -> (batch, channels, seq_len)
        attended_conv = attended_conv.transpose(1, 2)  # (batch, channels, seq_len)
        pooled_conv = self.stock_data_pool(attended_conv).squeeze(-1)
        stock_data_features = self.stock_data_linear_branch(pooled_conv)
        
        # process the rest
        portfolio_features = self.portfolio_branch(x[:, portfolio_start_idx: performance_start_idx])
        performance_features = self.performance_branch(x[:, performance_start_idx: risk_start_idx])
        risk_features = self.risk_branch(x[:, risk_start_idx: price_action_start_idx])
        price_action_features = self.price_action_branch(x[:, price_action_start_idx: position_management_start_idx])
        position_management_features = self.position_management_branch(x[:, position_management_start_idx: trading_behavior_start_idx])
        trading_behavior_features = self.trading_behavior_branch(x[:, trading_behavior_start_idx: micro_timing_start_idx])
        
        # process temporal features with attention
        micro_timing_raw = x[:, micro_timing_start_idx:intraday_timing_start_idx]
        intraday_timing_raw = x[:, intraday_timing_start_idx:weekly_timing_start_idx]
        weekly_timing_raw = x[:, weekly_timing_start_idx:]
        
        # project to common dimension
        micro_timing_proj = self.micro_timing_projection(micro_timing_raw)
        intraday_timing_proj = self.intraday_timing_projection(intraday_timing_raw)
        weekly_timing_proj = self.weekly_timing_projection(weekly_timing_raw)
        
        # stack temporal features for attention and apply
        # temporal_stack dim (batch_size, 3, temporal_feature_dim)
        temporal_stack = torch.stack([micro_timing_proj, intraday_timing_proj, weekly_timing_proj], dim=1)
        attended_temporal, _ = self.temporal_attention(temporal_stack)
        
        # flatten attended temporal features
        temporal_features = attended_temporal.view(batch_size, -1)
        temporal_features = self.temporal_output_projection(temporal_features)
        
        # cross modal attention
        # stock data attending to other features
        # project features to common dimension
        stock_data_proj = self.stock_data_proj(stock_data_features)
        portfolio_proj = self.portfolio_proj(portfolio_features)
        performance_proj = self.performance_proj(performance_features)
        risk_proj = self.risk_proj(risk_features)
        price_action_proj = self.price_action_proj(price_action_features)
        position_mgmt_proj = self.position_mgmt_proj(position_management_features)
        trading_behavior_proj = self.trading_behavior_proj(trading_behavior_features)
        
        # create feature matrix for cross-attention
        other_features = torch.stack([
            portfolio_proj, 
            performance_proj,
            risk_proj,
            price_action_proj,
            position_mgmt_proj,
            trading_behavior_proj
        ], dim=1)  # (batch_size, 6, feature_dim)
        
        # stock data as query, other features as key-value
        # reshape stock_data_proj (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        stock_query = stock_data_proj.unsqueeze(1)  
        
        # apply cross attention
        # stock data attending to other features
        stock_attended, _ = self.stock_cross_attention(stock_query, other_features)
        
        # # apply feature fusion attention
        # all_projected_features = torch.cat([
        #     stock_attended.unsqueeze(1), other_features
        # ], dim=1)  # (batch_size, 7, feature_dim)
        
        # fused_features, _ = self.feature_fusion_attention(all_projected_features)
        
        # # flatten fused features (skip the stock cross-attended feature since we have original stock_data_features)
        # fused_other_features = fused_features[:, 1:].reshape(batch_size, -1)  # Skip first (stock) feature
        
        # # combine all features
        # combined_features = torch.cat([
        #     stock_data_features,           # 64 features
        #     stock_attended,                # 16 features  
        #     temporal_features,             # 12 features
        #     fused_other_features,          # 6 * 16 = 96 features
        # ], dim=1)
        
        # return self.action_layer(combined_features)
        
        # Combine all feature representations
        # refine with self-attention over all features
        fused_features_seq = torch.stack([
            stock_attended,
            portfolio_proj,
            performance_proj,
            risk_proj,
            price_action_proj,
            position_mgmt_proj,
            trading_behavior_proj
        ], dim=1)  # shape: (batch_size, 7, feature_dim)

        refined_fusion, _ = self.feature_fusion_attention(fused_features_seq)
        refined_fusion_flat = refined_fusion.view(batch_size, -1)
        final_features = torch.cat([stock_data_features, temporal_features,refined_fusion_flat], dim=1)

        if self.use_dueling:
            # pass combined features through Value and Advantage streams
            # combine Value and Advantage to get q-values
            # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            value = self.value_stream(final_features)  # (batch_size, 1)
            advantage = self.advantage_stream(final_features)  # (batch_size, action_size)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.action_layer(final_features)
    
if __name__=='__main__':
    sizes = {
        'stock_data_window_size': 60,
        'stock_data_feature_size': 22,
        'portfolio_metrics_size': 6,
        'performance_metrics_size': 9,
        'risk_metrics_size': 4,
        'price_action_metrics_size': 14,
        'position_management_metrics_size': 9,
        'trading_behavior_metrics_size': 7,
        'temporal_metrics_size': 2,
        'temporal_metrics_types_count': 3,
        'action_size': 3
    }
    dqn = HierarchicalTradingDQNNetwork(sizes)
    print(dqn)

    # dummy data
    stock_data_flattened_dim = sizes['stock_data_window_size'] * sizes['stock_data_feature_size']
    temporal_states_dim = sizes['temporal_metrics_size'] * 3
    everything_else =  sizes['portfolio_metrics_size'] + sizes['performance_metrics_size'] + sizes['risk_metrics_size'] + sizes['price_action_metrics_size'] + sizes['position_management_metrics_size'] + sizes['trading_behavior_metrics_size'] + temporal_states_dim
    dummy_input = torch.randn(BATCH_SIZE, stock_data_flattened_dim + everything_else) 
    print(dummy_input.shape)
    
    output = dqn(dummy_input)
    print("Output Q-values shape:", output.shape)
    assert output.shape == (BATCH_SIZE, sizes['action_size'])