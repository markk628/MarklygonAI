import matplotlib.pyplot  as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from sklearn.preprocessing import StandardScaler

from src.config.config import DATA_DIR

pd.set_option('display.max_columns', None)

class StockTradingEnv:
    def __init__(self, data, initial_balance=10000, transaction_fee=0.0015, window_size=20):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee # 거래 수수료 (0.15%)
        self.window_size = window_size

        # 상태 정규화를 위한 스케일러
        self.scaler = StandardScaler()
        # 처음 100개 데이터로 스케일러 학습
        features = self._get_features(self.window_size)
        self.scaler.fit(features[:100])
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_cost = 0
        self.total_sales = 0

        return self._get_state()

    def _get_features(self, start_idx) -> np.array:
        features = []
        for i in range(start_idx - self.window_size, start_idx):
            row = [self.data.iloc[i][col] for col in self.data.columns]
            features.append(row)
        return np.array(features)

    def _get_state(self):
        # 현재 상태 정보 구성
        features = self._get_features(self.current_step)

        # 정규화
        normalized_features = self.scaler.transform(features)

        # 포트폴리오 정보 추가
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + self.shares_held * current_price
        portfolio_info = np.array([
            portfolio_value,
            self.balance / self.initial_balance, # 현재 잔고 비율
            self.shares_held * current_price / self.initial_balance, # 보유 주식 가치 비율
            self.shares_held > 0, # 주식 보유 여부 (불리언)
        ])

        # 상태를 1D 배열로 변환
        state = np.concatenate((normalized_features.flatten(), portfolio_info))

        return state
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        done = False
        
        # 매도 (전량)
        if action == 0 and self.shares_held > 0:
            # 매도 금액 계산
            sell_amount = self.shares_held * current_price
            # 거래 수수료 계산
            fee = sell_amount * self.transaction_fee
            # 실제 매도 금액
            self.balance += (sell_amount - fee)

            # 매도 기록 업데이트
            self.total_shares_sold += self.shares_held
            self.total_sales += sell_amount

            # 손익 계산 (매도 금액 - 매수 비용 - 수수료)
            profit = sell_amount - (self.shares_held * (self.total_cost / self.total_shares_bought if
            self.total_shares_bought > 0 else 0)) - fee

            # 보상 설정 (수익률)
            reward = profit / self.initial_balance

            # 주식 보유량 초기화
            self.shares_held = 0

        # 보유 (아무 행동 안함)
        elif action == 1:
            # 작은 음의 보상 (시간 비용)
            reward = -0.0001

        # 매수 (가능한 최대 수량)
        elif action == 2 and self.balance > 0:
            # 최대 구매 가능 주식 수 계산 (수수료 포함)
            max_shares = self.balance / (current_price * (1 + self.transaction_fee))
            max_shares = int(max_shares) # 소수점 이하 버림

            if max_shares > 0:
                # 매수 금액 계산
                buy_amount = max_shares * current_price
                # 거래 수수료 계산
                fee = buy_amount * self.transaction_fee
                # 실제 매수 비용
                cost = buy_amount + fee

                # 잔고 업데이트
                self.balance -= cost
                # 보유 주식 업데이트
                self.shares_held += max_shares

                # 매수 기록 업데이트
                self.total_shares_bought += max_shares
                self.total_cost += cost

                # 작은 음의 보상 (매수 직후에는 수수료만큼 손해)
                reward = -fee / self.initial_balance
            else:
            # 잔고 부족으로 매수 불가
                reward = -0.001

       
        # 다음 스텝으로 이동
        self.current_step += 1

        # 데이터 종료 여부 확인
        if self.current_step >= len(self.data) - 1:
            done = True
            # 마지막 스텝에서 모든 주식 매도하여 최종 성과 계산
            if self.shares_held > 0:
                sell_amount = self.shares_held * current_price
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)
                self.shares_held = 0

            # 최종 수익률 계산
            final_portfolio_value = self.balance
            return_rate = (final_portfolio_value - self.initial_balance) / self.initial_balance

            # 최종 보상에 전체 수익률 반영
            reward += return_rate

        # 다음 상태, 보상, 종료 여부 반환
        next_state = self._get_state()

        return next_state, reward, done, {}
    
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(state_size, 128)
        self.fc2: nn.Linear = nn.Linear(128, 64)
        self.fc3: nn.Linear = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float=0.001,
                 discount_factor: float=0.95,
                 epsilon: float=1.0,
                 epsilon_decay: float=0.995,
                 epsilon_min: float=0.01,
                 batch_size: int=64,
                 memory_size: int=2000):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.memory: deque = deque(maxlen=memory_size)
        self.batch_size: int = batch_size
        self.discount_factor: float = discount_factor # gamma (γ)
        self.epsilon: float = epsilon # epsilon (ε)
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.learning_rate: float = learning_rate

        # main and target network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network: DQNNetwork = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network: DQNNetwork = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() # target network doesn't get trained

        self.optimizer: optim.Adam = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.loss_fn: nn.MSELoss = nn.MSELoss()

        self.update_counter: int = 0
        self.target_update_frequency: int = 100 # standard for how often target network gets updated
        self.step = 0
    
    def remember(self, state, action, reward, next_state, done):
        # store experience in the experience replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        # select an action according to the ε-greedy policy
        if training and np.random.rand() < self.epsilon:
            # Exploration: Select a random action
            return random.randrange(self.action_size)

        # Exploitation: Select the optimal action according to the current policy
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(state)
        self.main_network.train()
        return torch.argmax(q_values, dim=1).item()

    def train(self):
        # Skip training if there is not enough experience accumulated in memory
        if len(self.memory) < self.batch_size:
            return

        # create sample batch
        minibatch = random.sample(self.memory, self.batch_size)

        # ready batch data
        states = torch.FloatTensor(np.array([experience[0] for experience in minibatch])).to(self.device)
        actions = torch.LongTensor([[experience[1]] for experience in minibatch]).to(self.device)
        rewards = torch.FloatTensor([[experience[2]] for experience in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([experience[3] for experience in minibatch])).to(self.device)
        dones = torch.FloatTensor([[experience[4]] for experience in minibatch]).to(self.device)

        # calculate current q values
        q_values = self.main_network(states).gather(1, actions)

        # Calculate the maximum Q-value of the next state using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]

        # calculate target q values using Q-learning formula
        target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # calculate loss and execute backpropagation
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decrease epsilon (decrease exploration probability)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * (self.epsilon_decay ** self.step)

        # periodically update the target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

    def load(self, name):
        self.main_network.load_state_dict(torch.load(name))
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save(self, name):
        torch.save(self.main_network.state_dict(), name)

def train_agent(env: StockTradingEnv, agent: DQNAgent, episodes: int=100):
    scores = []
    balances = []
    
    print('training...') 
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False
        count = 5759
        
        while not done:
            step = 5760 - count
            if step % 1000 == 0:
                print(f'Step: {step}')
            # choose action
            action = agent.act(state)

            # apply action to environment
            next_state, reward, done, _ = env.step(action)

            # save results
            agent.remember(state, action, reward, next_state, done)

            # train agent
            agent.train()

            state = next_state
            score += reward
            count -= 1

        scores.append(score)
        balances.append(env.balance)
        print(f"Episode: {e+1}/{episodes}, Score: {score:.4f}, "f"Balance: {env.balance:.2f}, Epsilon: {agent.epsilon:.4f}")

    return scores, balances

def evaluate_agent(env: StockTradingEnv, agent: DQNAgent, episodes: int=10):
    total_return = 0

    for e in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state, training=False) # evaluation mode
            next_state, reward, done, _ = env.step(action)
            state = next_state

        # calculate total gain
        return_rate = (env.balance - env.initial_balance) / env.initial_balance
        total_return += return_rate

        print(f"Evaluation Episode {e+1}/{episodes}, Return: {return_rate:.4f}, "f"Final Balance: {env.balance:.2f}")

    avg_return = total_return / episodes
    print(f"Average Return: {avg_return:.4f}")

    return avg_return

def load_stock_data(ticker: str) -> pd.DataFrame:
    drop_cols = ['timestamp', 'target']
    file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # ensure datetime format

    # Define your cutoff datetime
    cutoff = pd.Timestamp('2025-04-29 08:00:00', tz='UTC')

    # Filter the DataFrame
    df = df[df['timestamp'] >= cutoff]
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    
    return df

def main():
    data = load_stock_data('AAPL').loc[:,:'vwap']

    # create environment
    env = StockTradingEnv(data, initial_balance=10000, window_size=20)

    # define the size of the state and action space
    state = env.reset()
    state_size = len(state)
    action_size = 3 # sell(0), hold(1), buy(2)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = np.exp(np.log(epsilon_min / epsilon) / len(data))


    # initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        batch_size=1024,
        memory_size=10000
    )
    # train agent
    num_episodes = 100
    scores, balances = train_agent(env, agent, episodes=num_episodes)

    # visualize training result
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Learning Curve')

    plt.subplot(2, 1, 2)
    plt.plot(balances)
    plt.xlabel('Episode')
    plt.ylabel('Final Balance')
    plt.title('Portfolio Value')

    plt.tight_layout()
    plt.savefig('dqn_stock_trading_results.png')
    plt.show()

    # save model
    agent.save("dqn_stock_model.pth")

    # evaluate trained agent
    avg_return = evaluate_agent(env, agent, episodes=10)
    
if __name__ == '__main__':
    main()