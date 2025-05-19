import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import torch
from datetime import datetime

from src.config.config import (
    DATA_DIR, 
    MODELS_DIR, 
    RESULTS_DIR, 
    WINDOW_SIZE,
    TRAIN_RATIO, 
    VALID_RATIO,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    NUM_EPISODES,
    EVALUATE_INTERVAL
)
from src.models.mark.dqn.agent.DQNAgent import DQNAgent
from src.models.mark.dqn.env.StockTradingEnv import StockTradingEnv
from src.utils.utils import format_duration
from src.preprocessing.data_processor import RollingWindowFeatureProcessor

pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class DQNTrainer:

    def load_stock_data(self, ticker: str, cutoff: pd.Timestamp | None=None) -> pd.DataFrame:
        """
        Get saved csv data
        """
        print(f'Loading {ticker}...')
        drop_cols: list[str] = ['timestamp']
        file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
        df = pd.read_csv(file_path)

        if cutoff:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= cutoff]
            
        if drop_cols:
            df.drop(drop_cols, axis=1, inplace=True)
        
        print(f'{ticker} loaded succesfully')
        
        return df


    def split_data(self, data: pd.DataFrame, train_ratio: float=TRAIN_RATIO, val_ratio: float=VALID_RATIO):
        """
        Split data chronologically into train, validation, and test sets
        """
        print(f'Splitting data...')
        
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        train_data = data.iloc[:train_end].copy().reset_index(drop=True)
        val_data = data.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_data = data.iloc[val_end:].copy().reset_index(drop=True)
        
        print(f"Data split: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")
        
        return train_data, val_data, test_data
    
    
    def train_agent(self,
                    env: StockTradingEnv, 
                    agent: DQNAgent, 
                    episodes: int=NUM_EPISODES, 
                    validation_env: StockTradingEnv | None=None,
                    validation_frequency: int=EVALUATE_INTERVAL,
                    early_stopping_patience: int=10):
        """
        Train the agent with optional validation and early stopping
        """
        scores = []
        balances = []
        validation_scores = []
        invalid_action_counts = []
        
        best_validation_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        start_time = time.time()
        
        print('Training agent...')
        
        for e in range(episodes):
            agent.current_episode = e
            state = env.reset()
            score = 0
            done = False
            invalid_actions = 0
            trades_this_episode = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                if info.get('invalid_action'):
                    invalid_actions += 1
                if info.get('trade_info'):
                    trades_this_episode += 1
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                score += reward

            agent.scheduler.step(score)
            scores.append(score)
            balances.append(env.balance)
            invalid_action_counts.append(invalid_actions)
            
            if (e + 1) % 5 == 0:
                print(f"Episode: {e+1}/{episodes}, "
                      f"Score: {score:.4f}, "
                      f"Balance: {env.balance:.2f}, "
                      f"Trades: {trades_this_episode}, "
                      f"Invalid Actions: {invalid_actions}, "
                      f"Epsilon: {agent.epsilon:.4f}")
            
            # validation if provided
            if validation_env is not None and (e + 1) % validation_frequency == 0:
                validation_score = self.evaluate_agent(validation_env, agent, episodes=1, verbose=False)
                validation_scores.append(validation_score)
                
                # early stopping logic
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    patience_counter = 0
                    # save best model state
                    best_model_state = {k: v.cpu() for k, v in agent.main_network.state_dict().items()}
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at episode {e+1}. Best validation score: {best_validation_score:.4f}")
                    # restore best model
                    if best_model_state:
                        agent.main_network.load_state_dict(best_model_state)
                        agent.target_network.load_state_dict(best_model_state)
                    break
                
        print(f'Training Time: {format_duration(time.time() - start_time)}')
        
        training_metrics = {
            'scores': scores,
            'balances': balances,
            'validation_scores': validation_scores,
            'loss_history': agent.loss_history,
            'avg_q_values': agent.avg_q_values,
            'invalid_action_counts': invalid_action_counts
        }
        
        return training_metrics


    def evaluate_agent(self,
                       env: StockTradingEnv, 
                       agent: DQNAgent, 
                       episodes: int=10, 
                       verbose: bool=True):
        """
        Evaluate the agent's performance
        """
        total_return_rate = 0
        total_trades = 0
        total_invalid_actions = 0
        start_time = time.time()
        
        for e in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_trades = 0
            episode_invalid_actions = 0
            episode_rewards = []

            while not done:
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                episode_rewards.append(reward)
                episode_reward += reward
                
                if info.get('invalid_action'):
                    episode_invalid_actions += 1
                if info.get('trade_info'):
                    episode_trades += 1
                
            final_portfolio_value = env.balance + env.shares_held * env.data_nparray[env.current_step, env.close_prices_idx]
            return_rate = (final_portfolio_value - env.initial_balance) / env.initial_balance
            total_return_rate += return_rate
            total_trades += episode_trades
            total_invalid_actions += episode_invalid_actions
            
            if verbose:
                print(f"\nEvaluation Episode {e+1}/{episodes}")
                print(f"Final Balance: {env.balance:.2f}")
                print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
                print(f"Return Rate: {return_rate:.4f}")
                print(f"Trades: {episode_trades}")
                print(f"Invalid Actions: {episode_invalid_actions}")
                print(f"Average Reward: {np.mean(episode_rewards):.4f}")
                print(f"Total Shares Bought: {env.total_shares_bought}")
                print(f"Total Shares Sold: {env.total_shares_sold}")
                print(f"Episode Score: {episode_reward:.4f}")
            
        avg_return_rate = total_return_rate / episodes
        if verbose:
            print(f"\nEvaluation Summary:")
            print(f"Final Balance: {env.balance:.2f}")
            print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
            print(f"Average Return: {avg_return_rate:.4f}")
            print(f"Average Trades per Episode: {(total_trades / episodes):.1f}")
            print(f"Average Invalid Actions per Episode: {(total_invalid_actions / episodes):.1f}")
            print(f'Evaluation Time: {format_duration(time.time() - start_time)}')

        return avg_return_rate
    
    
    def plot_training_results(self, 
                              ticker: str,
                              metrics: dict[str, float | int], 
                              model_name: str="DQN", 
                              validation_frequency: int=5,):
        """
        Visualize training and performance metrics
        """
        plt.figure(figsize=(15, 10))
        
        # Plot training scores
        plt.subplot(2, 3, 1)
        plt.plot(metrics['scores'], label='Training Score')
        if metrics['validation_scores']:
            # Plot validation scores at their corresponding episodes
            validation_episodes = [i*validation_frequency for i in range(len(metrics['validation_scores']))]
            plt.plot(validation_episodes, metrics['validation_scores'], 'r-', label='Validation Score')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Score')
        plt.title(f'{model_name} Learning Curve - Cumulative Score')
        plt.legend()
        
        # Plot final balance after each episode
        plt.subplot(2, 3, 2)
        plt.plot(metrics['balances'])
        plt.xlabel('Episode')
        plt.ylabel('Final Balance ($)')
        plt.title('Portfolio Value at End of Episode')
        
        # Plot loss history
        if metrics['loss_history']:
            plt.subplot(2, 3, 3)
            plt.plot(metrics['loss_history'])
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
        
        # Plot average Q-values
        if metrics['avg_q_values']:
            plt.subplot(2, 3, 4)
            plt.plot(metrics['avg_q_values'])
            plt.xlabel('Action Selection')
            plt.ylabel('Average Q-Value')
            plt.title('Average Q-Values During Training')
            
        if metrics['invalid_action_counts']:
            plt.subplot(2, 3, 5)
            plt.plot(metrics['invalid_action_counts'])
            plt.xlabel('Episode')
            plt.ylabel('Invalid Actions')
            plt.title('Average Invalid Actions During Training')
        
        plt.suptitle(f'{ticker} DQN Training Results')
        plt.tight_layout()
        return plt


    def plot_backtest_results(self, 
                              portfolio_values: list[int], 
                              price_history: list[float], 
                              action_history: list[int], 
                              ticker: str):
        """
        Visualize backtesting results including price chart, portfolio value, and buy/sell actions
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
    
    
    def train_and_evaluate(self):
        print("Let's get this bread")
        # parameters
        ticker = 'AAPL'
        
        # load and prepare data
        print('Preparing data...')
        cutoff = pd.Timestamp('2024-04-06 08:00:00', tz='UTC')
        data = self.load_stock_data(ticker, cutoff)
        train_data, val_data, test_data = self.split_data(data)
        
        print('Preprocessing data...')
        window_processsor = RollingWindowFeatureProcessor()
        window_processsor.fit(train_data.iloc[:, :-1], train_data['target'])
        
        train_data = train_data[window_processsor.feature_processor.filtered_feature_names]
        val_data = val_data[window_processsor.feature_processor.filtered_feature_names]
        test_data = test_data[window_processsor.feature_processor.filtered_feature_names]
        
        # create environments
        train_env = StockTradingEnv(
            train_data, 
            window_processsor,
            mode='train'
        )
        
        val_env = StockTradingEnv(
            val_data, 
            window_processsor,
            mode='validation'
        )
        
        test_env = StockTradingEnv(
            test_data, 
            window_processsor,
            mode='test'
        )
        
        # initialize agent
        agent = DQNAgent(
            feature_size=train_env.feature_extracted_column_count * WINDOW_SIZE,
            portfolio_info_size=train_env.portfolio_info_count,
            market_info_size=train_env.market_info_count, 
            constraint_size=train_env.constraint_info_count, 
            action_size=3,
            total_steps=(len(train_data) - WINDOW_SIZE) * NUM_EPISODES,
            decay_rate_multiplier=0.3,
            epsilon_decay_target_pct=0.5,
            update_frequency=4,  
            target_update_frequency=200 
        )
        
        # train agent with validation-based early stopping
        training_metrics = self.train_agent(
            train_env, 
            agent, 
            validation_env=val_env,
            early_stopping_patience=15
        )
        
        # plot training results
        train_date = datetime.today().strftime(r'%Y-%m-%d_%H-%M-%S')
        training_plot = self.plot_training_results(ticker, training_metrics)
        training_plot.savefig(f'{RESULTS_DIR}/dqn_{ticker}_{train_date}_training_results.png')
        
        # evaluate on test set
        print("\nEvaluating on test set...")
        self.evaluate_agent(test_env, agent, episodes=1)
        
        state = test_env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            _, _, done, info = test_env.step(action)
        
        info = info['trade_info']
        
        # Plot backtest results
        if 'portfolio_values' in info:
            backtest_plot = self.plot_backtest_results(
                info['portfolio_values'], 
                info['price_history'], 
                info['action_history'], 
                ticker=ticker
            )
            backtest_plot.savefig(f"{RESULTS_DIR}/dqn_{ticker}_{train_date}_{info['return_rate']*100:.2f}_backtest_results.png")
            
        # Save model
        model_path = f"{MODELS_DIR}/dqn_{ticker}_{info['return_rate']*100:.4f}_model.pth"
        agent.save(model_path)
        
        # Print final metrics
        print(f"\nTest Results for {ticker}:")
        print(f"Final Balance: ${info['final_balance']:.2f}")
        print(f"Return Rate: {info['return_rate']:.4f} ({info['return_rate']*100:.4f}%)")
        print(f"Max Drawdown: {info['max_drawdown']:.4f} ({info['max_drawdown']*100:.2f}%)")
        print(f"Sharpe Ratio: {info['sharpe_ratio']:.4f}")
        
def main():
    trainer = DQNTrainer()
    trainer.train_and_evaluate()
    
if __name__=='__main__':
    main()