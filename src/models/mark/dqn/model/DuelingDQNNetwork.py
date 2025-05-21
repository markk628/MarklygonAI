import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class DuelingDQNNetwork(nn.Module):
    def __init__(self, 
                 market_data_timesteps: int,
                 market_data_features: int, 
                 portfolio_info_size: int,
                 market_info_size_flat: int, # Renamed to clarify it's the flattened market info
                 constraint_size: int,
                 action_size: int):
        super(DuelingDQNNetwork, self).__init__()

        self.market_data_timesteps = market_data_timesteps
        self.market_data_features = market_data_features
        self.portfolio_info_size = portfolio_info_size
        self.market_info_size_flat = market_info_size_flat # This is for other flattened market features if any
        self.constraint_size = constraint_size
        self.action_size = action_size

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
            nn.Linear(market_info_size_flat, 64), 
            nn.LeakyReLU(negative_slope=0.01), # try GELU
            nn.BatchNorm1d(64), # TODO try LayerNorm
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

        # --- Dueling DQN specific layers ---
        # input size for both value and advantage
        # output of state_linear_layers (64) + portfolio_layers (32) + market_layers (16) + constraint_layers (16) = 128
        self.combined_feature_size = 64 + 32 + 16 + 16
        
        # value stream: estimates the state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.combined_feature_size, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # advantage stream: estimates the advantage A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.combined_feature_size, 64),
            nn.LeakyReLU(negative_slope=0.01),
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
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Assuming x is a concatenated tensor:
        # [market_data (flattened), portfolio_info, market_info_flat, constraint_info]
        
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

        # pass combined features through Value and Advantage streams
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)

        # combine Value and Advantage to get q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    
if __name__=='__main__':
    # Original 'feature_size' (1215) was 60 * 22 + other flat features.
    # Now we explicitly pass the dimensions for the 2D market data.
    # Let's assume your previous 'feature_size' of 1215 was entirely the 60*22 market data.
    # If not, you need to adjust what 'market_info_size_flat' represents.
    
    # For a (60, 22) market data, total 1320 features if flattened.
    # Let's assume your original 'feature_size' of 1215 might have implicitly included 60 * X features, 
    # but the exact 2D shape (60, 22) is now explicit.
    
    # Let's re-evaluate the input sizes assuming:
    # 60 timesteps * 22 features = 1320 (this is your 'market_data' which will be handled by Conv1D)
    # The '15' for portfolio_info_size remains.
    # The '7' for market_info_size remains.
    # The '6' for constraint_size remains.
    # Total flattened input size = 1320 (market_data) + 15 (portfolio) + 7 (market_info_flat) + 6 (constraint) = 1348
    
    # When instantiating the network, pass the 2D dimensions for market data
    # market_data_timesteps = 60
    # market_data_features = 22
    # portfolio_info_size = 15
    # market_info_size_flat = 7 (This is for *other* flattened market info that isn't the 60x22 time series)
    # constraint_size = 6
    # action_size = 3
    
    dqn = DuelingDQNNetwork(
        market_data_timesteps=60, 
        market_data_features=22, 
        portfolio_info_size=16, 
        market_info_size_flat=7, 
        constraint_size=6, 
        action_size=3
    )
    print(dqn)

    # Example forward pass:
    # Batch size of 1
    batch_size = 1024
    
    # Create a dummy input tensor with the expected flattened total size
    # Total features = (60 * 22) + 15 + 7 + 6 = 1320 + 15 + 7 + 6 = 1348
    dummy_input = torch.randn(batch_size, (60 * 22) + 16 + 7 + 6) 
    print(dummy_input.shape)
    
    output = dqn(dummy_input)
    print("Output Q-values shape:", output.shape) # Should be (batch_size, action_size)
    assert output.shape == (batch_size, 3)