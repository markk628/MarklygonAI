import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class DuelingDQNNetwork(nn.Module):
    def __init__(self, 
                 state_size: int, 
                 constraint_size: int,
                 action_size: int):
        super(DuelingDQNNetwork, self).__init__()

        self.constraint_size = constraint_size
        self.state_layers = nn.Sequential(
            nn.Linear(state_size - constraint_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.constraint_layers = nn.Sequential(
            nn.Linear(constraint_size, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # --- Dueling DQN specific layers ---
        # input size for both value and advantage = output of state_layers (64) +  ouput of constraint_layers (16) = 80
        
        # value stream: estimates the state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # advantage stream: estimates the advantage A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, action_size)
        )

        # initialize optimized weights
        self.apply(self._init_weights)

    # TODO make a protocol/interface 
    def _init_weights(self, module):
        """
        Initializes weights using Kaiming Normal and biases to zero.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # process state features and constraint features separately
        state_features = self.state_layers(x[:, :-self.constraint_size])
        constraint_features = self.constraint_layers(x[:, -self.constraint_size:])

        # concatenate the outputs from both streams
        combined_features = torch.cat([state_features, constraint_features], dim=1)

        # pass combined features through Value and Advantage streams
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)

        # combine Value and Advantage to get q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    
if __name__=='__main__':
    dqn = DuelingDQNNetwork(1215, 6, 3)
    print(dqn)