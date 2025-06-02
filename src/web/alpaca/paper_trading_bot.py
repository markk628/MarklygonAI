import asyncio
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import logging
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from src.config.apikeys import POLYGON_APIKEY, ALPACA_APIKEY, ALPACA_SECRET_KEY
from src.config.config import DEVICE, WINDOW_SIZE, TRANSACTION_FEE_PERCENT
from src.preprocessing.feature_engineering import FeatureEngineer
from src.web.models import TradingSession, MarklygonModel, BacktestHistory, TradeHistory, TradeType, db, Portfolio
from src.web.extensions import app

logger = logging.getLogger(__name__)


class PaperTradingBot:
    """Paper trading bot using Polygon for data and Alpaca for execution"""
    
    def __init__(self, model_id: int, initial_balance: float, max_position_size: float = 0.7):
        self.model_id = model_id
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.is_running = False
        
        # Load model and preprocessor
        with app.app_context():
            model_record = MarklygonModel.query.get(model_id)
            if not model_record:
                raise ValueError(f"Model {model_id} not found")
            
            self.model_type = model_record.model.value
            self.ticker = model_record.ticker
            self.model_path = model_record.model_path
            
            # Get preprocessor path from latest backtest
            backtest = BacktestHistory.query.filter_by(model_id=model_id).order_by(BacktestHistory.id.desc()).first()
            self.preprocessor_path = backtest.preprocessor_path if backtest else None
        
        # Initialize model and preprocessor
        self._load_model()
        self._load_preprocessor()
        
        # Initialize data buffer
        self.data_buffer = []
        self.feature_engineer = FeatureEngineer()
        
        # Initialize Alpaca client
        self.alpaca_client = TradingClient(ALPACA_APIKEY, ALPACA_SECRET_KEY, paper=True)
        
        # Initialize Polygon websocket
        self.polygon_client = WebSocketClient(
            api_key=POLYGON_APIKEY,
            feed=Feed.RealTime,
            market=Market.Stocks
        )
        
        # Subscribe to minute aggregates for the ticker
        self.polygon_client.subscribe(f"AM.{self.ticker}")
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        # Create trading session in database
        self._create_trading_session()
    
    def _load_model(self):
        """Load the trained model"""
        if self.model_type == "DQN":
            # Import DQN v2 classes
            from src.models.mark.dqn_v2.dqn import DoubleDuelingDQN, TradingConfig
            
            config = TradingConfig()
            self.agent = DoubleDuelingDQN(config, device=DEVICE)
            self.agent.load(self.model_path)
            self.agent.q_network.eval()
            
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported yet")
    
    def _load_preprocessor(self):
        """Load the preprocessor if available"""
        if self.preprocessor_path:
            try:
                # Import the preprocessor class
                from src.models.mark.dqn_v2.data_preprocessor import FinancialDataPreprocessor
                
                # Load using the class method
                self.preprocessor = FinancialDataPreprocessor.load(self.preprocessor_path)
                logger.info(f"Loaded preprocessor from {self.preprocessor_path}")
            except Exception as e:
                logger.error(f"Failed to load preprocessor: {e}")
                logger.warning("Continuing without preprocessor - using raw features")
                self.preprocessor = None
                
                # Try alternative loading method for old format
                try:
                    import joblib
                    state = joblib.load(self.preprocessor_path)
                    if isinstance(state, dict) and 'scaling_method' in state:
                        # It's the raw state, try to reconstruct
                        self.preprocessor = FinancialDataPreprocessor(
                            scaling_method=str(state.get('scaling_method', 'robust')),
                            outlier_method=str(state.get('outlier_method', 'winsorize'))
                        )
                        if 'scalers' in state:
                            self.preprocessor.scalers = state['scalers']
                            self.preprocessor.outlier_bounds = state.get('outlier_bounds', {})
                            self.preprocessor.is_fitted = True
                            logger.info("Reconstructed preprocessor from old format")
                except Exception as e2:
                    logger.error(f"Alternative loading also failed: {e2}")
        else:
            self.preprocessor = None
            logger.warning(f"No preprocessor found for model {self.model_id}")
    
    def _create_trading_session(self):
        """Create a new trading session in the database"""
        with app.app_context():
            self.session = TradingSession(
                model_id=self.model_id,
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance
            )
            db.session.add(self.session)
            db.session.commit()
            self.session_id = self.session.id
    
    def _update_trading_session(self):
        """Update trading session statistics"""
        with app.app_context():
            session = TradingSession.query.get(self.session_id)
            if session:
                session.final_balance = self.balance + (self.position * self.current_price if hasattr(self, 'current_price') else 0)
                session.net_profit = session.final_balance - session.initial_balance
                session.total_trades = self.total_trades
                session.winning_trades = self.winning_trades
                session.losing_trades = self.losing_trades
                session.return_rate = (session.final_balance - session.initial_balance) / session.initial_balance
                session.invalid_actions = self.invalid_actions
                
                # Calculate other metrics
                if self.total_trades > 0:
                    session.win_rate = (self.winning_trades / self.total_trades) * 100
                
                db.session.commit()
    
    def _save_trade(self, trade_type: TradeType, amount: float, price: float, shares: float):
        """Save trade to database"""
        with app.app_context():
            # Get the current user's portfolio
            # For now, we'll use the first portfolio. In production, you'd link this to the logged-in user
            portfolio = Portfolio.query.first()
            
            if not portfolio:
                logger.warning("No portfolio found, creating trade without portfolio link")
                portfolio_id = 1
            else:
                portfolio_id = portfolio.id
            
            trade = TradeHistory(
                portfolio_id=portfolio_id,
                trading_session_id=self.session_id,
                trade_type=trade_type,
                amount=amount,
                price=price,
                shares=shares
            )
            db.session.add(trade)
            db.session.commit()
    
    def handle_message(self, msgs: list[WebSocketMessage]):
        """Handle incoming Polygon websocket messages"""
        for msg in msgs:
            if hasattr(msg, 'symbol') and msg.symbol == self.ticker:
                # Extract OHLCV data from the message
                data_point = {
                    'timestamp': pd.to_datetime(msg.start_timestamp, unit='ms', utc=True),
                    'open': msg.open,
                    'high': msg.high,
                    'low': msg.low,
                    'close': msg.close,
                    'volume': msg.volume,
                    'vwap': msg.vwap if hasattr(msg, 'vwap') else msg.close
                }
                
                self.current_price = msg.close
                self.data_buffer.append(data_point)
                
                # Process data when we have enough
                if len(self.data_buffer) >= WINDOW_SIZE:
                    self._process_data()
    
    def _process_data(self):
        """Process accumulated data and make trading decision"""
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffer[-WINDOW_SIZE:])
            
            # Add technical indicators
            df = self._add_features(df)
            
            # Get the state for the model
            state = self._prepare_state(df)
            
            if state is not None:
                # Get action from model
                action = self._get_action(state)
                
                # Execute action
                self._execute_action(action)
                
                # Update session
                self._update_trading_session()
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features to the data"""
        # Use feature engineer methods
        df = self.feature_engineer._add_technical_indicators(df)
        df = self.feature_engineer._add_temporal_patterns(df)
        df = self.feature_engineer._add_price_differences_and_returns(df)
        
        # Fill any NaN values
        df = df.ffill().fillna(0)
        
        return df
    
    def _prepare_state(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare state tensor for the model"""
        try:
            # Select features based on model type
            if self.model_type == "DQN":
                from src.config.config import STOCK_FEATURES
                
                # Make sure we have all required features
                available_features = [col for col in STOCK_FEATURES if col in df.columns]
                if len(available_features) < len(STOCK_FEATURES) * 0.8:  # Allow some missing features
                    logger.warning(f"Missing features: {set(STOCK_FEATURES) - set(available_features)}")
                    return None
                
                # Get stock data
                stock_data = df[available_features].values
                
                # Apply preprocessing if available
                if self.preprocessor:
                    stock_data = self.preprocessor.transform(df[available_features])
                else:
                    # Basic normalization if no preprocessor
                    logger.warning("No preprocessor available, using raw features")
                    # Just ensure the data has the right shape
                    stock_data = df[available_features].values
                
                # Calculate portfolio features
                portfolio_value = self.balance + (self.position * self.current_price)
                
                portfolio_features = np.array([
                    self.balance / self.initial_balance,
                    (self.position * self.current_price) / self.initial_balance if self.initial_balance > 0 else 0,
                    portfolio_value / self.initial_balance if self.initial_balance > 0 else 0,
                    (self.position * self.current_price) / portfolio_value if portfolio_value > 0 else 0,
                    self.total_trades / 100,  # Normalized
                    self.winning_trades / max(1, self.total_trades),
                    self.invalid_actions / 100,  # Normalized
                    1.0 if self.position > 0 else 0.0
                ])
                
                # Combine features
                portfolio_features_repeated = np.tile(portfolio_features, (WINDOW_SIZE, 1))
                combined_features = np.concatenate([stock_data, portfolio_features_repeated], axis=1)
                
                # Convert to tensor
                state = torch.tensor(combined_features, dtype=torch.float32, device=DEVICE)
                
                return state
            
        except Exception as e:
            logger.error(f"Error preparing state: {e}")
            return None
    
    def _get_action(self, state: torch.Tensor) -> int:
        """Get action from the model"""
        with torch.no_grad():
            if self.model_type == "DQN":
                action = self.agent.select_action(state, epsilon=0.0)  # No exploration
            else:
                raise NotImplementedError(f"Model type {self.model_type} not supported")
        
        return action
    
    def _execute_action(self, action: int):
        """Execute trading action via Alpaca"""
        try:
            # DQN actions: 0=Hold, 1=Buy, 2=Sell
            if action == 0:  # Hold
                return
            
            elif action == 1:  # Buy
                if self.position > 0:  # Already have position
                    self.invalid_actions += 1
                    return
                
                # Calculate position size
                position_value = self.balance * self.max_position_size
                shares_to_buy = int(position_value / self.current_price)
                
                if shares_to_buy <= 0:
                    self.invalid_actions += 1
                    return
                
                # Place buy order
                order_request = MarketOrderRequest(
                    symbol=self.ticker,
                    qty=shares_to_buy,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.alpaca_client.submit_order(order_request)
                
                # Update local state
                cost = shares_to_buy * self.current_price * (1 + TRANSACTION_FEE_PERCENT)
                self.balance -= cost
                self.position = shares_to_buy
                self.entry_price = self.current_price
                
                # Save trade
                self._save_trade(TradeType.BUY, cost, self.current_price, shares_to_buy)
                
                logger.info(f"BUY {shares_to_buy} shares of {self.ticker} at ${self.current_price}")
            
            elif action == 2:  # Sell
                if self.position <= 0:  # No position to sell
                    self.invalid_actions += 1
                    return
                
                # Place sell order
                order_request = MarketOrderRequest(
                    symbol=self.ticker,
                    qty=self.position,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.alpaca_client.submit_order(order_request)
                
                # Calculate profit/loss
                revenue = self.position * self.current_price * (1 - TRANSACTION_FEE_PERCENT)
                cost_basis = self.position * self.entry_price * (1 + TRANSACTION_FEE_PERCENT)
                profit = revenue - cost_basis
                
                # Update statistics
                self.total_trades += 1
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Update local state
                self.balance += revenue
                shares_sold = self.position
                self.position = 0
                
                # Save trade
                self._save_trade(TradeType.SELL, revenue, self.current_price, shares_sold)
                
                logger.info(f"SELL {shares_sold} shares of {self.ticker} at ${self.current_price}, profit: ${profit:.2f}")
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            self.invalid_actions += 1
    
    def start(self):
        """Start the trading bot"""
        self.is_running = True
        logger.info(f"Starting paper trading bot for model {self.model_id} on {self.ticker}")
        
        # Run the websocket client
        self.polygon_client.run(self.handle_message)
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info(f"Stopping paper trading bot for model {self.model_id}...")
        
        # Close any open positions
        if self.position > 0:
            logger.info(f"Closing open position of {self.position} shares...")
            try:
                self._execute_action(2)  # Sell
            except Exception as e:
                logger.error(f"Error closing position: {e}")
        
        # Update final session state
        try:
            with app.app_context():
                session = TradingSession.query.get(self.session_id)
                if session:
                    session.end_time = datetime.now(timezone.utc)
                    # Calculate final balance including any open positions
                    final_balance = self.balance
                    if self.position > 0 and hasattr(self, 'current_price'):
                        final_balance += self.position * self.current_price
                    session.final_balance = final_balance
                    session.net_profit = final_balance - session.initial_balance
                    session.return_rate = (final_balance - session.initial_balance) / session.initial_balance
                    db.session.commit()
                    logger.info(f"Updated trading session {self.session_id} - Final balance: ${final_balance:.2f}")
        except Exception as e:
            logger.error(f"Error updating session: {e}")
        
        # Close websocket connection
        try:
            if hasattr(self, 'polygon_client') and self.polygon_client:
                logger.info("Closing Polygon websocket connection...")
                self.polygon_client.close()
                logger.info("Polygon websocket disconnected successfully")
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")
        
        logger.info(f"Paper trading bot for model {self.model_id} stopped successfully") 