import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import numba
from numba import jit
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

@jit(nopython=True)
def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def load_stock_data(file_path):
    print("데이터 로딩 시작...")
    print_memory_usage()
    start_time = time.time()
    
    # pickle 파일 로드
    data = pd.read_pickle(file_path)
    
    # timestamp를 datetime으로 변환
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    # 최근 1년치 데이터만 선택
    end_date = data['timestamp'].max()
    start_date = end_date - timedelta(days=365)  # 1년치 데이터
    data = data[data['timestamp'] >= start_date]
    
    # 10분봉을 1시간봉으로 리샘플링
    data.set_index('timestamp', inplace=True)
    data = data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    data.reset_index(inplace=True)
    
    print(f"데이터 기간: {start_date.date()} ~ {end_date.date()}")
    
    # 필요한 컬럼만 선택
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = data[columns].copy()
    
    # 기술적 지표 계산
    print("기술적 지표 계산 중...")
    
    # 이동평균 (병렬 처리)
    with ThreadPoolExecutor(max_workers=3) as executor:
        ma5_future = executor.submit(lambda: data['close'].rolling(window=5).mean())
        ma10_future = executor.submit(lambda: data['close'].rolling(window=10).mean())
        ma20_future = executor.submit(lambda: data['close'].rolling(window=20).mean())
        
        data['MA5'] = ma5_future.result()
        data['MA10'] = ma10_future.result()
        data['MA20'] = ma20_future.result()
    
    # RSI (Numba 최적화)
    data['RSI'] = calculate_rsi(data['close'].values)
    
    # MACD (병렬 처리)
    with ThreadPoolExecutor(max_workers=2) as executor:
        ema12_future = executor.submit(lambda: data['close'].ewm(span=12, adjust=False).mean())
        ema26_future = executor.submit(lambda: data['close'].ewm(span=26, adjust=False).mean())
        
        data['EMA12'] = ema12_future.result()
        data['EMA26'] = ema26_future.result()
    
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 결측값 제거
    data = data.dropna()
    
    print(f"데이터 로딩 완료 (소요시간: {time.time() - start_time:.2f}초)")
    print(f"데이터 크기: {data.shape}")
    print_memory_usage()
    
    return data

class StockTradingEnv:
    def __init__(self, data, initial_balance=10000, transaction_fee=0.0005, window_size=20):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        
        # 데이터를 numpy 배열로 변환하여 빠른 접근
        self.price_data = data[['close', 'volume', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'Signal']].values
        
        # 특성 캐시 초기화
        self._feature_cache = {}
        
        # 상태 정규화를 위한 스케일러
        self.scaler = StandardScaler()
        features = self._get_features(self.window_size)
        self.scaler.fit(features[:100])
        
        # 거래 기록 초기화
        self.trade_history = []
        
        # 마지막에 reset 호출
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
    
    def _get_features(self, start_idx):
        # 캐시 키 생성
        cache_key = f"{start_idx}_{self.window_size}"
        
        # 캐시된 데이터가 있으면 반환
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # numpy 슬라이싱으로 빠르게 데이터 추출
        features = self.price_data[start_idx - self.window_size:start_idx].copy()
        
        # 가격 변동률 계산 (벡터화)
        price_changes = np.zeros((self.window_size, 1))
        price_changes[1:] = (np.diff(features[:, 0]) / features[:-1, 0]).reshape(-1, 1)
        
        # 모든 특성 결합
        result = np.hstack((features, price_changes))
        
        # 캐시에 저장
        self._feature_cache[cache_key] = result
        
        return result
    
    def _get_state(self):
        features = self._get_features(self.current_step)
        normalized_features = self.scaler.transform(features)
        
        current_price = self.price_data[self.current_step, 0]
        portfolio_value = self.balance + self.shares_held * current_price
        portfolio_info = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,
            self.shares_held > 0,
        ])
        
        return np.concatenate((normalized_features.flatten(), portfolio_info))
    
    def step(self, action):
        current_price = self.price_data[self.current_step, 0]
        reward = 0
        done = False
        trade_info = None
        
        if action == 0 and self.shares_held > 0:  # 매도
            sell_amount = self.shares_held * current_price
            fee = sell_amount * self.transaction_fee
            self.balance += (sell_amount - fee)
            
            self.total_shares_sold += self.shares_held
            self.total_sales += sell_amount
            
            profit = sell_amount - (self.shares_held * (self.total_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0)) - fee
            reward = profit / self.initial_balance
            
            trade_info = {
                'type': 'SELL',
                'shares': self.shares_held,
                'price': current_price,
                'amount': sell_amount,
                'fee': fee,
                'profit': profit
            }
            
            self.shares_held = 0
            
        elif action == 1:  # 보유
            # 보유 중일 때의 보상: 현재 포트폴리오 가치의 변화율
            portfolio_value = self.balance + self.shares_held * current_price
            prev_portfolio_value = self.balance + self.shares_held * self.price_data[self.current_step-1, 0]
            reward = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
        elif action == 2 and self.balance > 0:  # 매수
            max_shares = int(self.balance / (current_price * (1 + self.transaction_fee)))
            
            if max_shares > 0:
                buy_amount = max_shares * current_price
                fee = buy_amount * self.transaction_fee
                cost = buy_amount + fee
                
                self.balance -= cost
                self.shares_held += max_shares
                
                self.total_shares_bought += max_shares
                self.total_cost += cost
                
                reward = -fee / self.initial_balance
                
                trade_info = {
                    'type': 'BUY',
                    'shares': max_shares,
                    'price': current_price,
                    'amount': buy_amount,
                    'fee': fee
                }
            else:
                reward = -0.001
        
        self.current_step += 1
        
        if self.current_step >= len(self.price_data) - 1:
            done = True
            if self.shares_held > 0:
                sell_amount = self.shares_held * current_price
                fee = sell_amount * self.transaction_fee
                self.balance += (sell_amount - fee)
                self.shares_held = 0
            
            final_portfolio_value = self.balance
            return_rate = (final_portfolio_value - self.initial_balance) / self.initial_balance
            reward += return_rate
        
        if trade_info:
            self.trade_history.append(trade_info)
        
        next_state = self._get_state()
        return next_state, reward, done, {'trade_info': trade_info}

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=64, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.main_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.update_counter = 0
        self.target_update_frequency = 100
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.main_network.eval()
        with torch.no_grad():
            q_values = self.main_network(state)
        self.main_network.train()
        return torch.argmax(q_values, dim=1).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 배치 데이터를 한 번에 준비
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([[experience[1]] for experience in minibatch])
        rewards = np.array([[experience[2]] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([[experience[4]] for experience in minibatch])
        
        # numpy 배열을 torch 텐서로 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.main_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
        
        target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))
        
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
        
    
    def save(self, name):
        torch.save(self.main_network.state_dict(), name)
    
    def load(self, name):
        self.main_network.load_state_dict(torch.load(name))
        self.target_network.load_state_dict(self.main_network.state_dict())

def train_agent(env, agent, episodes=100):
    scores = []
    balances = []
    start_time = time.time()
    
    for e in range(episodes):
        episode_start = time.time()
        state = env.reset()
        score = 0
        done = False
        step_count = 0
        trades_this_episode = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            score += reward
            step_count += 1
            
            if info.get('trade_info'):
                trades_this_episode += 1

            # step별 로깅 (100 step마다 출력)
            if step_count % 100 == 0 or done:
                print(f"[에피소드 {e+1}/{episodes} | step {step_count}] "
                      f"액션: {action}, 보상: {reward:.5f}, 잔고: {env.balance:.2f}, "
                      f"보유주식: {env.shares_held}, 총자산: {env.balance + env.shares_held * env.price_data[env.current_step, 0]:.2f}, "
                      f"거래횟수: {trades_this_episode}, "
                      f"Epsilon: {agent.epsilon}")
        
        scores.append(score)
        balances.append(env.balance)
        
        if (e + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            episode_time = time.time() - episode_start
            print(f"\nEpisode: {e+1}/{episodes}, Score: {score:.4f}, "
                  f"Balance: {env.balance:.2f}, Epsilon: {agent.epsilon:.4f}, "
                  f"Trades: {trades_this_episode}, "
                  f"Time: {elapsed_time:.2f}s, Episode Time: {episode_time:.2f}s")
            print_memory_usage()
    
    return scores, balances

def evaluate_agent(env, agent, episodes=10):
    total_return = 0
    start_time = time.time()
    
    for e in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        
        return_rate = (env.balance - env.initial_balance) / env.initial_balance
        total_return += return_rate
        
        print(f"Evaluation Episode {e+1}/{episodes}, Return: {return_rate:.4f}, "
              f"Final Balance: {env.balance:.2f}")
    
    avg_return = total_return / episodes
    elapsed_time = time.time() - start_time
    print(f"Average Return: {avg_return:.4f}, Evaluation Time: {elapsed_time:.2f}s")
    
    return avg_return

if __name__ == "__main__":
    print("프로그램 시작...")
    print_memory_usage()
    
    # 데이터 로드
    file_path = "./data/APPL.pkl"
    data = load_stock_data(file_path)
    
    # 환경 생성
    env = StockTradingEnv(data, initial_balance=10000, window_size=20)
    
    # 상태 및 행동 공간 크기 정의
    state = env.reset()
    state_size = len(state)
    action_size = 3  # 매도(0), 보유(1), 매수(2)
    num_episodes = 30
    epsilon = 1.0
    epsilon_min = 0.01
    total_steps = len(data) * num_episodes
    epsilon_decay = (epsilon_min / epsilon) ** (1 / total_steps)
    
    # 에이전트 생성
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
        batch_size=256,  # 배치 크기 증가
        memory_size=10000
    )
    
    # 에이전트 학습
    print("\n학습 시작...")
    scores, balances = train_agent(env, agent, episodes=num_episodes)
    
    # 학습 결과 시각화
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
    
    # 모델 저장
    agent.save("dqn_stock_model.pth")
    
    # 학습된 에이전트 평가
    print("\n평가 시작...")
    avg_return = evaluate_agent(env, agent, episodes=10)
    
    print("\n프로그램 종료")
    print_memory_usage() 