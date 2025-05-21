import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class DQNNetwork(nn.Module):
    def __init__(self, 
                 market_data_timesteps: int,
                 market_data_features: int,
                 portfolio_info_size: int,
                 market_info_size: int, 
                 constraint_size: int,
                 action_size: int):
        super(DQNNetwork, self).__init__()
        
        self.market_data_timesteps = market_data_timesteps
        self.market_data_features = market_data_features
        self.portfolio_info_size = portfolio_info_size
        self.market_info_size = market_info_size
        self.constraint_size = constraint_size
        
        # --- state_layers using Conv1d for 2D market data ---
        # Input to Conv1d: (batch_size, in_channels, sequence_length)
        # in_channels will be market_data_features (22)
        # sequence_length will be market_data_timesteps (60)
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.market_data_features, out_channels=64, kernel_size=3, padding=1),
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
            # After convolutions, you'll likely want to flatten or pool
            # to get a fixed-size feature vector for the dense layers.
            # MaxPool1d can help reduce the sequence length.
            nn.AdaptiveMaxPool1d(1), # Pools across the time dimension to get a fixed size output
            nn.Flatten()
        )
        
        # Calculate the output size of the Conv1d layers after AdaptiveMaxPool1d(1)
        # It will be `out_channels` of the last Conv1d layer, which is 256.
        # This will be the feature_size for the subsequent linear layers.
        self.processed_market_data_size = 256 
        
        # Now, the rest of the state_layers as Linear layers
        self.state_linear_layers = nn.Sequential(
            nn.Linear(self.processed_market_data_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.portfolio_layers = nn.Sequential(
            nn.Linear(portfolio_info_size, 256),
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
        
        self.market_layers = nn.Sequential(
            nn.Linear(market_info_size, 64), 
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

        self.constraint_layers = nn.Sequential(
            nn.Linear(constraint_size, 64), 
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
        # output of state_layers (64) + portfolio_layers (32) + market_layers (16) + of constraint_layers (16) = 128
        self.action_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, action_size)
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
        market_data_flat = x[:, :self.market_data_timesteps * self.market_data_features]
        
        # Reshape for Conv1d: (batch_size, num_channels, sequence_length)
        market_data_reshaped = market_data_flat.view(
            -1, self.market_data_features, self.market_data_timesteps
        )
        
        # Process through Conv1d layers
        # The output of AdaptiveMaxPool1d(1) will be (batch_size, processed_market_data_size, 1)
        # We need to squeeze the last dimension to make it (batch_size, processed_market_data_size)
        conv_output = self.conv1d_layers(market_data_reshaped).squeeze(-1)
        
        # process features separately
        # Pass through the remaining linear layers for state features
        data_features = self.state_linear_layers(conv_output)
        # 2. Extract and process other features
        # Adjust indices based on the new `market_data_timesteps` * `market_data_features`
        
        # Starting index for portfolio_info
        portfolio_start_idx = self.market_data_timesteps * self.market_data_features
        portfolio_features = self.portfolio_layers(x[:, portfolio_start_idx : portfolio_start_idx + self.portfolio_info_size])
        
        # Starting index for market_info_flat
        market_info_flat_start_idx = portfolio_start_idx + self.portfolio_info_size
        market_features = self.market_layers(x[:, market_info_flat_start_idx : market_info_flat_start_idx + self.market_info_size_flat])
        
        # Starting index for constraint_info
        constraint_start_idx = market_info_flat_start_idx + self.market_info_size_flat
        constraint_features = self.constraint_layers(x[:, constraint_start_idx:])

        # concatenate the outputs from all streams
        combined_features = torch.cat([data_features, portfolio_features, market_features, constraint_features], dim=1)

        return self.action_layer(combined_features)
    
if __name__=='__main__':
    dqn = DQNNetwork(1215, 6, 3)
    print(dqn)