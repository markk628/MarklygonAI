import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, sizes):
        super(DQNNetwork, self).__init__()
        
        self.stock_data_window_size = sizes['stock_data_window_size']
        self.stock_data_feature_size = sizes['stock_data_feature_size']
        self.portfolio_metrics_size = sizes['portfolio_metrics_size']
        self.market_state_metrics_size = sizes['market_state_metrics_size']
        self.constraint_metrics_size = sizes['constraint_metrics_size']
        self.action_size = sizes['action_size']
        
        # --- state_layers using Conv1d for 2D market data ---
        # Input to Conv1d: (batch_size, in_channels, sequence_length)
        # in_channels will be stock_data_feature_size (22)
        # sequence_length will be stock_data_window_size (60)
        self.stock_data_conv1d_branch = nn.Sequential(
            nn.Conv1d(in_channels=self.stock_data_feature_size, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.AdaptiveMaxPool1d(1) # pool to get features ready for Linear layer
        )
        
        # `out_channels` of the last Conv1d layer
        self.processed_market_data_size = 256 
        
        # Now, the rest of the state_layers as Linear layers
        self.stock_data_linear_branch = nn.Sequential(
            nn.Linear(self.processed_market_data_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.portfolio_metrics_size, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.market_branch = nn.Sequential(
            nn.Linear(self.market_state_metrics_size, 64), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.constraint_branch = nn.Sequential(
            nn.Linear(self.constraint_metrics_size, 64), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # action layer
        # output of state_layers (64) + portfolio_branch (32) + market_branch (16) + of constraint_branch (16) = 128
        self.action_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, self.action_size)
        )
        
        # initialize optimized weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initializes weights using Kaiming Normal and biases to zero.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 1. Extract market data (2D)
        # It's initially flattened in the input `x`
        # Reshape it to (batch_size, num_channels, sequence_length)
        market_data_flat = x[:, :self.stock_data_window_size * self.stock_data_feature_size]
        
        # Reshape for Conv1d: (batch_size, num_channels, sequence_length)
        market_data_reshaped = market_data_flat.view(
            -1, self.stock_data_feature_size, self.stock_data_window_size
        )
        
        # Process through Conv1d layers
        # The output of AdaptiveMaxPool1d(1) will be (batch_size, processed_market_data_size, 1)
        # We need to squeeze the last dimension to make it (batch_size, processed_market_data_size)
        conv_output = self.stock_data_conv1d_branch(market_data_reshaped).squeeze(-1)
        
        # process features separately
        # Pass through the remaining linear layers for state features
        data_features = self.stock_data_linear_branch(conv_output)
        # 2. Extract and process other features
        # Adjust indices based on the new `stock_data_window_size` * `stock_data_feature_size`
        
        # Starting index for portfolio_info
        portfolio_start_idx = self.stock_data_window_size * self.stock_data_feature_size
        portfolio_features = self.portfolio_branch(x[:, portfolio_start_idx : portfolio_start_idx + self.portfolio_metrics_size])
        
        # Starting index for market_info_flat
        market_info_flat_start_idx = portfolio_start_idx + self.portfolio_metrics_size
        market_features = self.market_branch(x[:, market_info_flat_start_idx : market_info_flat_start_idx + self.market_state_metrics_size])
        
        # Starting index for constraint_info
        constraint_start_idx = market_info_flat_start_idx + self.market_state_metrics_size
        constraint_features = self.constraint_branch(x[:, constraint_start_idx:])

        # concatenate the outputs from all streams
        combined_features = torch.cat([data_features, portfolio_features, market_features, constraint_features], dim=1)

        return self.action_layer(combined_features)
    
if __name__=='__main__':
    dqn = DQNNetwork(1215, 6, 3)
    print(dqn)