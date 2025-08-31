import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque, defaultdict
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch.types import Number
from typing import Any, Dict, Tuple, Optional
from src.config.config import DATA_DIR 

# Set pandas display options and random seeds for reproducibility
pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)



# add code that gives penalty if agent tries to sell if while having nothing to sell
# properly initialize state
# start with can_buy bool
# generally speaking figure out how to handle wrong actions





class StockTradingEnv:
    """
    Environment for stock trading
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        initial_balance: int=10000, 
        transaction_fee: float=0.0015, 
        window_size: int=20,
        mode: str='train'  # 'train', 'validation', or 'test'
    ):  
        self.data: pd.DataFrame = data
        self.initial_balance: int = initial_balance
        self.transaction_fee: float = transaction_fee
        self.window_size: int = window_size
        self.mode: str = mode
        self._feature_cache = {}
        self.reset()
        
    def reset(self) -> NDArray:
        # Start at window_size to ensure enough historical data for the first state's features
        self.current_step = self.window_size 
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_cost = 0
        self.total_sales = 0
        
        # Initialize portfolio metrics tracking
        self.portfolio_values = [self.initial_balance]
        self.action_history = []
        self.price_history = []
        
        return self._get_state()
    
    def _get_features(self, current_idx) -> NDArray:
        """
        Extract features from a rolling window of historical data
        """
        # 캐시 키 생성
        cache_key = f"{current_idx}_{self.window_size}"
        
        # 캐시된 데이터가 있으면 반환
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        features_list = []
        # Iterate through current_idx - window_size up to current_idx
        for i in range(current_idx - self.window_size, current_idx):
            features_at_step_i = [self.data.iloc[i][feature_name] for feature_name in self.data.columns]
            features_list.append(features_at_step_i)
            
        features_list = np.array(features_list)
        self._feature_cache[cache_key] = features_list  
        return features_list

    def _get_state(self) -> NDArray:
        """
        Construct the current state with normalized features and portfolio information
        """
        # Features are from the window [self.current_step - self.window_size, self.current_step - 1]
        raw_features = self._get_features(self.current_step)

        # Rolling window scaling: Fit and transform on the features of the current window
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(raw_features)

        # Current price for portfolio valuation
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + self.shares_held * current_price
        
        # Track portfolio value history
        self.portfolio_values.append(portfolio_value)
        self.price_history.append(current_price)
        
        # Enhanced portfolio information
        balance_ratio = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        shares_value_ratio = (self.shares_held * current_price) / self.initial_balance if self.initial_balance > 0 else 0
        
        # Calculate profit/loss from current position
        avg_buy_price = self.total_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0
        position_pl = (current_price - avg_buy_price) * self.shares_held if self.shares_held > 0 else 0
        position_pl_ratio = position_pl / self.initial_balance if self.initial_balance > 0 else 0

        portfolio_info = np.array([
            portfolio_value / self.initial_balance,  # normalized portfolio value
            balance_ratio,                           # ratio of current and initial balance
            shares_value_ratio,                      # ratio of shares value to initial balance
            float(self.shares_held > 0),             # boolean if shares are held
            position_pl_ratio                        # profit/loss ratio on current position
        ])

        # Return state as a 1D array
        return np.concatenate((normalized_features.flatten(), portfolio_info))
    
    def step(self, action):
        """
        Execute one step in the environment based on the agent's action
        Actions: 0=Sell all, 1=Hold, 2=Buy max
        """
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        done = False
        info = {}  # Dictionary to store metrics about this step
        is_action_valid = True
        
        action_validity = 0 # TODO make logic to use this instead of is_action_valid
        
        # Track the action
        self.action_history.append(action)
        
        # Process different actions
        if action == 0 and self.shares_held > 0:  # Sell all
            sell_amount = self.shares_held * current_price
            fee = sell_amount * self.transaction_fee
            self.balance += (sell_amount - fee)

            self.total_shares_sold += self.shares_held
            self.total_sales += sell_amount

            # Calculate profit on this sale
            avg_buy_cost_per_share = (self.total_cost / self.total_shares_bought) if self.total_shares_bought > 0 else 0
            profit = sell_amount - (self.shares_held * avg_buy_cost_per_share) - fee
            
            # Normalize reward by initial balance
            reward = profit / self.initial_balance if self.initial_balance > 0 else 0
            self.shares_held = 0
            
            info['action_taken'] = 'sell'
            info['profit'] = profit
            info['fee'] = fee

        elif action == 1:  # Hold
            # Small penalty for holding to encourage action
            reward = -0.0001
            info['action_taken'] = 'hold'

        elif action == 2 and self.balance > 0:  # Buy as much as possible
            price_with_fee = current_price * (1 + self.transaction_fee)
            if price_with_fee <= 0:  # Safety check
                max_shares = 0
            else:
                max_shares = int(self.balance / price_with_fee)

            if max_shares > 0:
                buy_amount = max_shares * current_price
                fee = buy_amount * self.transaction_fee
                cost = buy_amount + fee

                if self.balance >= cost:  # Ensure affordability
                    self.balance -= cost
                    self.shares_held += max_shares
                    self.total_shares_bought += max_shares
                    self.total_cost += cost
                    # Penalty for transaction fee
                    reward = -fee / self.initial_balance if self.initial_balance > 0 else 0
                    
                    info['action_taken'] = 'buy'
                    info['shares_bought'] = max_shares
                    info['cost'] = cost
                    info['fee'] = fee
                else:
                    reward = -0.001  # Not enough balance
                    info['action_taken'] = 'failed_buy'
            else:
                reward = -0.001  # Insufficient balance for any shares
                info['action_taken'] = 'failed_buy'
        else:
            # do something with reward here if a invalid action is chosen (buy when no money left or sell when no shares held)
            is_action_valid = False
            info['action_taken'] = 'invalid_action'

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            done = True
            # Liquidate any remaining shares at the final price
            final_price_idx = len(self.data) - 1
            final_price = self.data.iloc[final_price_idx]['close']

            if self.shares_held > 0:
                sell_amount = self.shares_held * final_price
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)
                self.shares_held = 0
                info['final_liquidation'] = True
                info['liquidation_amount'] = sell_amount
                info['liquidation_fee'] = fee
            
            # Calculate final performance metrics
            final_portfolio_value = self.balance
            return_rate = (final_portfolio_value - self.initial_balance) / self.initial_balance
            
            # Add to final reward
            reward += return_rate
            
            # Add performance metrics to info
            info['final_balance'] = final_portfolio_value
            info['return_rate'] = return_rate
            
            # For test mode, calculate more detailed metrics
            if self.mode == 'test':
                # Calculate drawdown and other metrics
                portfolio_values = np.array(self.portfolio_values)
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
                sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
                
                info['max_drawdown'] = max_drawdown
                info['sharpe_ratio'] = sharpe_ratio
                info['portfolio_values'] = self.portfolio_values
                info['price_history'] = self.price_history
                info['action_history'] = self.action_history

        # Get the next state
        next_state = np.append(self._get_state(), is_action_valid) if not done else np.zeros(self.state_size)

        return next_state, reward, done, info
    
    def _calculate_max_drawdown(self, portfolio_values) -> np.float64:
        """
        Calculate the maximum drawdown from peak to trough
        """
        # Convert to numpy array if not already
        values = np.array(portfolio_values)
        # Calculate the running maximum
        running_max = np.maximum.accumulate(values)
        # Calculate drawdown in percentage terms
        drawdown = (running_max - values) / running_max
        # Get the maximum drawdown
        max_drawdown = np.max(drawdown)
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.02/252) -> np.float64:
        """
        Calculate the Sharpe ratio of the portfolio
        """
        # Convert to numpy array if not already
        values = np.array(portfolio_values)
        # Calculate daily returns
        daily_returns = np.diff(values) / values[:-1]
        # Calculate excess returns over risk-free rate
        excess_returns = daily_returns - risk_free_rate
        # Calculate Sharpe ratio (annualized)
        if np.std(excess_returns) == 0:
            return 0
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio

    @property
    def state_size(self) -> int:
        """
        Calculate state size dynamically based on features and portfolio info
        """
        if not hasattr(self, '_state_size_cached'):
            # Flattened normalized features
            features_part_len = self.window_size * self.data.columns.size
            # Portfolio info part (currently 5 values)
            portfolio_part_len = 5
            # is_action_valid
            additional_state_values_count = 1
            self._state_size_cached = features_part_len + portfolio_part_len + additional_state_values_count
        return self._state_size_cached


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with dropout and batch normalization
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        # Hidden layers
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
    
    def forward(self, x):
        # Check if input is a single sample and add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        return self.fc4(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture that separates state value and advantage functions
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQNNetwork, self).__init__()
        # Common feature layer
        print('state_size:', state_size)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        # Check if input is a single sample and add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay for more efficient learning
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha: float = alpha  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta: float = beta  # Importance sampling weight (0 = no correction, 1 = full correction)
        self.beta_increment: float = beta_increment  # Beta increases over time for more correction
        self.buffer = []
        self.priorities: NDArray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # Initial max priority for new transitions
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a new experience with max priority
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # New experiences get max priority to ensure they're sampled
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample experiences based on their priorities
        """
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get samples and calculate importance sampling weights
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** -self.beta
        weights /= weights.max()  # Normalize weights
        
        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(map(list, zip(*samples)))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities based on TD errors
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with prioritized experience replay and dueling architecture option
    """
    def __init__(
        self, 
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        update_frequency: int = 4,
        target_update_frequency: int = 100,
        use_dueling: bool = True,
        use_prioritized: bool = True
    ):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.batch_size: int = batch_size
        self.discount_factor: float = discount_factor  # gamma (γ)
        self.epsilon: float = epsilon  # epsilon (ε)
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.learning_rate: float = learning_rate
        self.use_dueling: bool = use_dueling
        self.use_prioritized: bool = use_prioritized

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Network initialization
        if use_dueling:
            self.main_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
        else:
            self.main_network = DQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size).to(self.device)
            
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() 

        # Optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        
        # Memory setup
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)

        # Training parameters
        self.update_counter = 0
        self.target_update_frequency = target_update_frequency
        self.update_frequency = update_frequency
        self.training_steps = 0
        
        # Metrics tracking
        self.loss_history = []
        self.avg_q_values = []
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        if self.use_prioritized:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True) -> Number:
        """
        Select action using epsilon-greedy policy
        """
        # Exploration during training
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        # Convert state to tensor
        state = torch.FloatTensor(state).to(self.device)
        
        # Get Q-values from network
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(state)
        self.main_network.train()
        
        # Track average Q-values during training
        if training:
            self.avg_q_values.append(q_values.mean().item())
            
        return torch.argmax(q_values, dim=1).item()

    def train(self):
        """
        Train the agent by sampling from replay buffer
        """
        # Skip if not enough samples
        if self.use_prioritized:
            if len(self.memory) < self.batch_size:
                return
        else:
            if len(self.memory) < self.batch_size:
                return
                
        self.training_steps += 1
        
        # Only update every update_frequency steps
        if self.training_steps % self.update_frequency != 0:
            return
            
        # Sample from memory
        if self.use_prioritized:
            batch, indices, is_weights = self.memory.sample(self.batch_size)
            if batch is None:  # Not enough samples
                return
                
            states, actions, rewards, next_states, dones = batch
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            states = np.array([experience[0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q-values
        q_values = self.main_network(states).gather(1, actions)

        # Double DQN: Get actions from main network
        with torch.no_grad():
            next_actions = self.main_network(next_states).max(1, keepdim=True)[1]
            # Get Q-values for those actions from target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Calculate target Q-values
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # Calculate loss
        if self.use_prioritized:
            # TD errors for updating priorities
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            # Weighted MSE loss
            loss = (is_weights * F.mse_loss(q_values, target_q_values, reduction='none')).mean()
        else:
            loss = F.mse_loss(q_values, target_q_values)
            
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in buffer
        if self.use_prioritized:
            self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant for stability

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            
        # Track loss
        self.loss_history.append(loss.item())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights from file"""
        self.main_network.load_state_dict(torch.load(name, map_location=self.device))
        self.target_network.load_state_dict(self.main_network.state_dict())
        print(f"Model loaded from {name}")

    def save(self, name):
        """Save model weights to file"""
        torch.save(self.main_network.state_dict(), name)
        print(f"Model saved to {name}")


def train_agent(env: StockTradingEnv, 
                agent: DQNAgent, 
                episodes: int=100, 
                validation_env: Optional[StockTradingEnv]=None,
                early_stopping_patience: int=10):
    """
    Train the agent with optional validation and early stopping
    """
    scores = []
    balances = []
    validation_scores = []
    
    best_validation_score = float('-inf')
    patience_counter = 0
    best_model_state = None
    
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            score += reward

        scores.append(score)
        balances.append(env.balance)
        
        print(f'Episode: {e}/{episodes} |',
              f'Score: {score:.4f} |',
              f'Balance: {env.balance:.2f} |',
              f'Epsilon: {agent.epsilon:.4f}'
        )
        
        # Validation if provided
        if validation_env is not None and (e + 1) % 5 == 0:  # Validate every 5 episodes
            validation_score = evaluate_agent(validation_env, agent, episodes=1, verbose=False)
            validation_scores.append(validation_score)
            
            # Early stopping logic
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu() for k, v in agent.main_network.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at episode {e+1}. Best validation score: {best_validation_score:.4f}")
                # Restore best model
                if best_model_state:
                    agent.main_network.load_state_dict(best_model_state)
                    agent.target_network.load_state_dict(best_model_state)
                break
    
    training_metrics = {
        'scores': scores,
        'balances': balances,
        'validation_scores': validation_scores,
        'loss_history': agent.loss_history,
        'avg_q_values': agent.avg_q_values
    }
    
    return training_metrics


def evaluate_agent(env: StockTradingEnv, agent: DQNAgent, episodes: int=10, verbose: bool=True):
    """
    Evaluate the agent's performance
    """
    total_return_rate = 0
    all_metrics = []

    for e in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state, training=False)  # Evaluation mode
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            
        # Get return rate for this episode
        return_rate = (env.balance - env.initial_balance) / env.initial_balance
        total_return_rate += return_rate
        
        # Store metrics for this episode
        if env.mode == 'test':
            metrics = {
                'episode': e+1,
                'return_rate': return_rate,
                'final_balance': env.balance,
                'episode_reward': episode_reward
            }
            if 'max_drawdown' in info:
                metrics.update({
                    'max_drawdown': info['max_drawdown'],
                    'sharpe_ratio': info['sharpe_ratio']
                })
            all_metrics.append(metrics)

        if verbose:
            print(f"Evaluation Episode {e+1}/{episodes}, Return Rate: {return_rate:.4f}, "
                  f"Final Balance: {env.balance:.2f}, Episode Score: {episode_reward:.4f}")

    avg_return_rate = total_return_rate / episodes
    if verbose:
        print(f"Average Return Rate over {episodes} evaluation episodes: {avg_return_rate:.4f}")

    return avg_return_rate if episodes > 1 else return_rate


def plot_training_results(metrics, model_name="DQN"):
    """
    Visualize training and performance metrics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training scores
    plt.subplot(2, 2, 1)
    plt.plot(metrics['scores'], label='Training Score')
    if metrics['validation_scores']:
        # Plot validation scores at their corresponding episodes
        validation_episodes = [i*5 for i in range(len(metrics['validation_scores']))]
        plt.plot(validation_episodes, metrics['validation_scores'], 'r-', label='Validation Score')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Score')
    plt.title(f'{model_name} Learning Curve - Cumulative Score')
    plt.legend()
    
    # Plot final balance after each episode
    plt.subplot(2, 2, 2)
    plt.plot(metrics['balances'])
    plt.xlabel('Episode')
    plt.ylabel('Final Balance ($)')
    plt.title('Portfolio Value at End of Episode')
    
    # Plot loss history
    if metrics['loss_history']:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['loss_history'])
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
    
    # Plot average Q-values
    if metrics['avg_q_values']:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['avg_q_values'])
        plt.xlabel('Action Selection')
        plt.ylabel('Average Q-Value')
        plt.title('Average Q-Values During Training')
    
    plt.tight_layout()
    return plt


def plot_backtest_results(env_data, portfolio_values, price_history, action_history, ticker="Stock"):
    """
    Visualize backtesting results including price chart, portfolio value,
    and buy/sell actions
    """
    plt.figure(figsize=(15, 10))
    
    # Plot stock price
    plt.subplot(2, 1, 1)
    plt.plot(price_history, label=f'{ticker} Price')
    
    # Mark buy and sell actions
    buy_indices = [i for i, a in enumerate(action_history) if a == 2]
    sell_indices = [i for i, a in enumerate(action_history) if a == 0]
    
    if buy_indices:
        plt.scatter(buy_indices, [price_history[i] for i in buy_indices], 
                   color='green', marker='^', s=100, label='Buy')
    if sell_indices:
        plt.scatter(sell_indices, [price_history[i] for i in sell_indices], 
                   color='red', marker='v', s=100, label='Sell')
    
    plt.xlabel('Trading Step')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} Price and Trading Actions')
    plt.legend()
    
    # Plot portfolio value
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values, label='Portfolio Value')
    
    # Calculate and plot buy-and-hold strategy for comparison
    initial_balance = portfolio_values[0]
    initial_price = price_history[0]
    shares_bought = initial_balance / initial_price
    buy_hold_values = [shares_bought * price for price in price_history]
    plt.plot(buy_hold_values, '--', label='Buy & Hold Strategy')
    
    plt.xlabel('Trading Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Comparison')
    plt.legend()
    
    plt.tight_layout()
    return plt


def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock data
    """
    drop_cols = ['timestamp', 'target']
    file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    
    return df

def split_data(data: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """
    Split data chronologically into train, validation, and test sets
    """
    # Calculate split indices
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    
    # Split data
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    val_data = data.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_data = data.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"Data split: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")
    
    return train_data, val_data, test_data


def main():
    # Parameters
    ticker = 'AAPL'
    window_size = 60  # Increased for better context
    initial_balance = 10000
    transaction_fee = 0.001  # Reduced from original
    
    # Load and prepare data
    data = load_stock_data(ticker)
    train_data, val_data, test_data = split_data(data)
    
    # Create environments
    train_env = StockTradingEnv(
        train_data, 
        initial_balance=initial_balance, 
        transaction_fee=transaction_fee, 
        window_size=window_size,
        mode='train'
    )
    
    val_env = StockTradingEnv(
        val_data, 
        initial_balance=initial_balance, 
        transaction_fee=transaction_fee, 
        window_size=window_size,
        mode='validation'
    )
    
    test_env = StockTradingEnv(
        test_data, 
        initial_balance=initial_balance, 
        transaction_fee=transaction_fee, 
        window_size=window_size,
        mode='test'
    )
    
    # Define agent
    episodes = 150
    steps_per_episode = len(train_data) - window_size
    state_size = train_env.state_size
    action_size = 3  # sell(0), hold(1), buy(2)
    epsilon = 1.0
    epsilon_min = 0.05
    total_steps = steps_per_episode * episodes
    epsilon_decay = (epsilon_min / epsilon) ** (1 / total_steps)
    
    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0005,  # Lower learning rate for stability
        discount_factor=0.97,  # Higher discount factor for longer-term rewards
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        batch_size=1024,       # Larger batch size for more stable learning
        memory_size=20000,    # Larger memory for better experience diversity
        update_frequency=4,   # Update every 4 steps for efficiency
        target_update_frequency=200,  # Less frequent target updates for stability
        use_dueling=True,     # Use dueling architecture
        use_prioritized=True  # Use prioritized replay
    )
    
    # Train agent with validation-based early stopping
    print("Starting training...")
    training_metrics = train_agent(
        train_env, 
        agent, 
        episodes=episodes,  # More episodes for better learning
        validation_env=val_env,
        early_stopping_patience=15  # Stop if no improvement for 15 validations
    )
    
    # Plot training results
    training_plot = plot_training_results(training_metrics)
    training_plot.savefig(f'dqn_{ticker}_training_results.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_return = evaluate_agent(test_env, agent, episodes=1)
    
    # Get detailed test metrics for final run
    state = test_env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
    
    # Plot backtest results
    if 'portfolio_values' in info:
        backtest_plot = plot_backtest_results(
            test_data, 
            info['portfolio_values'], 
            info['price_history'], 
            info['action_history'], 
            ticker=ticker
        )
        backtest_plot.savefig(f'dqn_{ticker}_backtest_results.png')
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/dqn_{ticker}_model.pth"
    agent.save(model_path)
    
    # Print final metrics
    print(f"\nTest Results for {ticker}:")
    print(f"Final Balance: ${info['final_balance']:.2f}")
    print(f"Return Rate: {info['return_rate']:.4f} ({info['return_rate']*100:.2f}%)")
    print(f"Max Drawdown: {info['max_drawdown']:.4f} ({info['max_drawdown']*100:.2f}%)")
    print(f"Sharpe Ratio: {info['sharpe_ratio']:.4f}")


if __name__ == '__main__':
    main()