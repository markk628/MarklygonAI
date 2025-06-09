import asyncio
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
import logging
import threading
import time
from decimal import Decimal
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Feed, Market
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from collections import deque

from src.config.apikeys import POLYGON_APIKEY, ALPACA_APIKEY, ALPACA_SECRET_KEY
from src.config.config import DEVICE, WINDOW_SIZE, TRANSACTION_FEE_PERCENT
from src.preprocessing.feature_engineering_2 import FeatureEngineer
from src.web.models import TradingSession, MarklygonModel, BacktestHistory, TradeHistory, TradeType, db, Portfolio
from src.web.extensions import app

# Configure logger for console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Paper Trading Bot module loaded")

class PaperTradingBot:
    """Paper trading bot using Polygon for data and Alpaca for execution"""
    
    # Constants for better maintainability
    MIN_PRICE = 0.01  # Minimum valid price (1 cent)
    DATA_BUFFER_MULTIPLIER = 4  # Buffer size = WINDOW_SIZE * 4 for adequate feature engineering
    GAP_CHECK_INTERVAL = 65  # Seconds between gap checks
    MAX_GAP_FILL_MINUTES = 5  # Maximum minutes to forward fill
    TRADE_EXECUTION_TIMEOUT = 30  # Seconds to wait for trade execution
    
    def __init__(self, model_id: int, initial_balance: float, max_position_size: float = 0.7):
        self.model_id = model_id
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position_size = max_position_size
        self.position = 0
        self.entry_price = 0
        self.current_price = 0
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        # Enhanced tracking for 17-feature state (matching DQN training environment)
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = initial_balance
        self.position_entry_step = -1  # Track when position was entered
        self.consecutive_invalid_actions = 0
        self.last_action = 0  # Track last action (0=hold, 1=buy, 2=sell)
        self.unrealized_pnl = 0.0
        self.step_count = 0  # Track steps for timing calculations
        
        # Trading session tracking
        self.session_start_time = datetime.now(timezone.utc)
        
        # Data buffer for windowed analysis - increased size for better feature engineering
        # Need more data to ensure technical indicators have sufficient history
        buffer_size = WINDOW_SIZE * self.DATA_BUFFER_MULTIPLIER  # 120 minutes for 30-minute window
        self.data_buffer = deque(maxlen=buffer_size)
        logger.info(f"Initialized data buffer with size: {buffer_size} minutes")
        
        self.is_running = False
        self.ticker = None
        self.device = DEVICE
        
        # Load model and preprocessor
        self._load_model_and_config()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Initialize Alpaca client
        self._initialize_alpaca_client()
        
        # Initialize Polygon websocket
        self._initialize_polygon_client()
        
        # Create trading session in database
        self.session_id = self._create_trading_session()
        
        # Gap filling setup
        self._initialize_gap_filling()
    
    def _load_model_and_config(self):
        """Load model configuration and setup"""
        with app.app_context():
            model_record = MarklygonModel.query.get(self.model_id)
            if not model_record:
                raise ValueError(f"Model {self.model_id} not found")
            
            self.model_type = model_record.model.value
            self.ticker = model_record.ticker
            self.model_path = model_record.model_path
            
            # Get preprocessor path from latest backtest
            backtest = BacktestHistory.query.filter_by(model_id=self.model_id).order_by(BacktestHistory.id.desc()).first()
            self.preprocessor_path = backtest.preprocessor_path if backtest else None
        
        # Initialize model and preprocessor
        self._load_model()
        self._load_preprocessor()
        
        # Store config from the model
        if self.model_type == "DQN":
            from src.models.mark.dqn_v2.dqn_v2 import TradingConfig
            self.config = TradingConfig()
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported yet")
    
    def _initialize_alpaca_client(self):
        """Initialize Alpaca trading client"""
        try:
            self.alpaca_client = TradingClient(ALPACA_APIKEY, ALPACA_SECRET_KEY, paper=True)
            logger.info("Alpaca client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            raise
    
    def _initialize_polygon_client(self):
        """Initialize Polygon websocket client"""
        try:
            logger.info("Initializing Polygon websocket client...")
            self.polygon_client = WebSocketClient(
                api_key=POLYGON_APIKEY,
                feed=Feed.RealTime,
                market=Market.Stocks
            )
            logger.info("Polygon websocket client initialized with RealTime feed for Stocks market")
            
            # Subscribe to minute aggregates for the ticker
            subscription = f"AM.{self.ticker}"
            logger.info(f"Subscribing to Polygon websocket for: {subscription}")
            self.polygon_client.subscribe(subscription)
            logger.info(f"Successfully subscribed to {subscription}")
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            raise
    
    def _initialize_gap_filling(self):
        """Initialize gap filling setup"""
        self.last_data_timestamp = None
        self.last_data_point = None
        self.gap_filler_thread = None
        self.gap_filler_stop_event = threading.Event()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if self.model_type == "DQN":
                # Import DQN v2 classes - fixed import path
                from src.models.mark.dqn_v2.dqn_v2 import DoubleDuelingDQN, TradingConfig
                
                config = TradingConfig()
                self.agent = DoubleDuelingDQN(config, device=DEVICE)
                self.agent.load(self.model_path)
                self.agent.q_network.eval()
                logger.info(f"Loaded DQN model from {self.model_path}")
            else:
                raise NotImplementedError(f"Model type {self.model_type} not supported yet")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_preprocessor(self):
        """Load the preprocessor if available"""
        if not self.preprocessor_path:
            self.preprocessor = None
            logger.warning(f"No preprocessor found for model {self.model_id}")
            return
        
        try:
            # Import the preprocessor class - fixed import path
            from src.models.mark.dqn_v2.data_preprocessor_v2 import FinancialDataPreprocessor
            
            # Load using the class method
            self.preprocessor = FinancialDataPreprocessor.load(self.preprocessor_path)
            logger.info(f"Loaded preprocessor from {self.preprocessor_path}")
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            logger.warning("Continuing without preprocessor - using raw features")
            self.preprocessor = None
            
            # Try alternative loading method for backward compatibility
            try:
                import joblib
                state = joblib.load(self.preprocessor_path)
                if isinstance(state, dict) and 'scaling_method' in state:
                    # Reconstruct from old format
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
                self.preprocessor = None
    
    def _create_trading_session(self):
        """Create a new trading session in the database"""
        with app.app_context():
            self.session = TradingSession(
                model_id=self.model_id,
                initial_balance=Decimal(str(self.initial_balance)),
                final_balance=Decimal(str(self.initial_balance))
            )
            db.session.add(self.session)
            db.session.commit()
            return self.session.id
    
    def _update_trading_session(self):
        """Update trading session statistics"""
        with app.app_context():
            session = TradingSession.query.get(self.session_id)
            if session:
                current_price = getattr(self, 'current_price', 0)
                position_value = self.position * current_price if current_price else 0
                final_balance = self.balance + position_value
                
                session.final_balance = Decimal(str(final_balance))
                session.net_profit = Decimal(str(final_balance - self.initial_balance))
                session.total_trades = self.total_trades
                session.winning_trades = self.winning_trades
                session.losing_trades = self.losing_trades
                
                # Calculate return rate with division by zero check
                if self.initial_balance > 0:
                    return_rate = (final_balance - self.initial_balance) / self.initial_balance
                    session.return_rate = Decimal(str(return_rate))
                else:
                    session.return_rate = Decimal('0')
                    
                session.invalid_actions = self.invalid_actions
                
                # Calculate other metrics
                if self.total_trades > 0:
                    win_rate = (self.winning_trades / self.total_trades) * 100
                    session.win_rate = Decimal(str(win_rate)) if hasattr(session, 'win_rate') else None
                
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
                amount=Decimal(str(amount)),
                price=Decimal(str(price)),
                shares=Decimal(str(shares))
            )
            db.session.add(trade)
            db.session.commit()
    
    def _is_market_open(self, check_time: datetime = None) -> bool:
        """Check if US stock market is open at given time"""
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        try:
            # Try to use pytz for proper timezone handling
            import pytz
            et = pytz.timezone('US/Eastern')
            et_time = check_time.astimezone(et)
        except ImportError:
            # Fallback to simple UTC offset (doesn't handle DST properly)
            et_time = check_time.astimezone(timezone(timedelta(hours=-5)))
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        market_open_time = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open_time <= et_time <= market_close_time
    
    def _check_and_fill_gaps(self):
        """Check for gaps in data and forward fill if necessary"""
        while not self.gap_filler_stop_event.is_set():
            try:
                # Wait for specified interval (slightly more than a minute to account for delays)
                self.gap_filler_stop_event.wait(self.GAP_CHECK_INTERVAL)
                
                if self.gap_filler_stop_event.is_set():
                    break
                
                # Check if we have received any data
                if self.last_data_timestamp and self.last_data_point:
                    current_time = datetime.now(timezone.utc)
                    time_since_last_data = (current_time - self.last_data_timestamp).total_seconds()
                    
                    # Only check for gaps during market hours
                    if not self._is_market_open(current_time):
                        logger.debug("Market is closed, skipping gap check")
                        continue
                    
                    # If more than 70 seconds have passed since last data, forward fill
                    if time_since_last_data > 70:
                        logger.warning(f"⚠️ No data received for {time_since_last_data:.0f} seconds, forward filling...")
                        
                        # Calculate how many minutes we need to fill
                        minutes_to_fill = int(time_since_last_data // 60)
                        
                        # Fill each missing minute (cap at MAX_GAP_FILL_MINUTES to avoid too many fills)
                        for i in range(1, min(minutes_to_fill + 1, self.MAX_GAP_FILL_MINUTES + 1)):
                            fill_timestamp = self.last_data_timestamp + timedelta(minutes=i)
                            
                            # Skip if market would be closed at this timestamp
                            if not self._is_market_open(fill_timestamp):
                                continue
                            
                            # Create forward filled data point
                            filled_data_point = self.last_data_point.copy()
                            filled_data_point['timestamp'] = fill_timestamp
                            
                            # Add to buffer
                            self.current_price = filled_data_point['close']
                            self.data_buffer.append(filled_data_point)
                            
                            logger.info(f"📋 Forward filled minute {i}/{min(minutes_to_fill, self.MAX_GAP_FILL_MINUTES)}:")
                            logger.info(f"   Timestamp: {fill_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                            logger.info(f"   Price: ${filled_data_point['close']:.2f}")
                        
                        logger.info(f"✅ Forward fill complete. Buffer size: {len(self.data_buffer)}/{self.data_buffer.maxlen}")
                        
                        # Update last data timestamp to current time
                        self.last_data_timestamp = current_time
                        
                        # Process data if buffer has enough data AND market is open
                        min_buffer_size = WINDOW_SIZE * 2  # Need at least 2x window size for feature engineering
                        if len(self.data_buffer) >= min_buffer_size and self._is_market_open(current_time):
                            logger.info(f"Buffer adequate after forward fill and market open - processing data for trading decision...")
                            self._process_data()
                        elif len(self.data_buffer) >= min_buffer_size:
                            logger.info(f"Buffer adequate after forward fill but market closed - data collected for buffer maintenance only")
                
            except Exception as e:
                logger.error(f"Error in gap filler: {e}", exc_info=True)
    
    def handle_message(self, msgs: list[WebSocketMessage]):
        """Handle incoming Polygon websocket messages"""
        # Check if we should stop processing
        if not self.is_running:
            logger.info("Bot is stopping, ignoring incoming messages")
            # Try to close the websocket
            try:
                self.polygon_client.close()
            except:
                pass
            return
            
        logger.info(f"Received {len(msgs)} messages from Polygon websocket")
        
        for msg in msgs:
            # Log the raw message type and attributes
            logger.info(f"Message type: {type(msg).__name__}")
            
            # Try to log common attributes
            if hasattr(msg, 'event_type'):
                logger.info(f"Event type: {msg.event_type}")
            if hasattr(msg, 'status'):
                logger.info(f"Status: {msg.status}")
            if hasattr(msg, 'message'):
                logger.info(f"Message: {msg.message}")
                
            # Check if it's a data message with our ticker
            if hasattr(msg, 'symbol'):
                logger.info(f"Symbol: {msg.symbol}")
                
                if msg.symbol == self.ticker:
                    current_time = datetime.now(timezone.utc)
                    is_market_open = self._is_market_open(current_time)
                    
                    logger.info(f"Processing {self.ticker} data:")
                    if is_market_open:
                        logger.info(f"📊 REAL DATA RECEIVED (MARKET OPEN)")
                    else:
                        logger.info(f"📊 REAL DATA RECEIVED (MARKET CLOSED - DATA ONLY)")
                    
                    # Log all available price/volume data
                    if hasattr(msg, 'open'):
                        logger.info(f"  Open: ${msg.open}")
                    if hasattr(msg, 'high'):
                        logger.info(f"  High: ${msg.high}")
                    if hasattr(msg, 'low'):
                        logger.info(f"  Low: ${msg.low}")
                    if hasattr(msg, 'close'):
                        logger.info(f"  Close: ${msg.close}")
                    if hasattr(msg, 'volume'):
                        logger.info(f"  Volume: {msg.volume}")
                    if hasattr(msg, 'vwap'):
                        logger.info(f"  VWAP: ${msg.vwap}")
                    if hasattr(msg, 'start_timestamp'):
                        logger.info(f"  Timestamp: {pd.to_datetime(msg.start_timestamp, unit='ms', utc=True)}")
                    
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
                    
                    # Update last data tracking for gap filling
                    self.last_data_timestamp = datetime.now(timezone.utc)
                    self.last_data_point = data_point.copy()
                    
                    logger.info(f"Added data point to buffer. Buffer size: {len(self.data_buffer)}/{self.data_buffer.maxlen}")
                    
                    # Process trading decisions if we have enough data
                    min_buffer_size = WINDOW_SIZE * 2  # Need sufficient data for feature engineering
                    if len(self.data_buffer) >= min_buffer_size:
                        if is_market_open:
                            logger.info(f"Buffer adequate and market open - processing data for trading decision...")
                            self._process_data()
                        else:
                            logger.info(f"Buffer adequate but market closed - data collected for buffer maintenance only")
                else:
                    logger.info(f"Ignoring message for symbol {msg.symbol} (not {self.ticker})")
            else:
                # Log non-data messages
                logger.info(f"Non-data message received: {msg}")
    
    def _process_data(self):
        """Process accumulated data and make trading decision"""
        try:
            # Double-check market hours before processing
            current_time = datetime.now(timezone.utc)
            if not self._is_market_open(current_time):
                logger.warning("_process_data called outside market hours - aborting trading decision")
                return
            
            logger.info("="*60)
            logger.info("PROCESSING DATA FOR TRADING DECISION (MARKET HOURS)")
            
            # Convert full buffer to DataFrame
            full_df = pd.DataFrame(list(self.data_buffer))
            logger.info(f"Full buffer DataFrame: {len(full_df)} rows")
            
            # Apply feature engineering following the exact sequence from feature_engineering_2.py
            # Note: Data gaps should already be handled by the gap filler thread
            # Main focus here is ensuring proper technical indicator calculation
            
            # Basic data validation (gaps should already be filled)
            # full_df = self._validate_and_clean_data(full_df)
            
            # Apply feature engineering in the correct order
            full_df = self.feature_engineer._add_price_features(full_df)
            full_df = self.feature_engineer._add_volume_features(full_df)
            full_df = self.feature_engineer._add_technical_indicators(full_df)
            full_df = self.feature_engineer._add_temporal_patterns(full_df)
            full_df = self.feature_engineer._add_market_context(full_df)
            full_df = self.feature_engineer._add_portfolio_features(full_df)
            full_df = self.feature_engineer._add_derived_features(full_df)
            
            # Apply comprehensive NaN and infinite value handling
            # full_df = self._handle_infinite_values(full_df)
            
            # Extract last WINDOW_SIZE minutes for model analysis
            df = full_df.tail(WINDOW_SIZE).copy()
            logger.info(f"Analysis window: {len(df)} rows (last {WINDOW_SIZE} minutes)")
            
            # Final validation
            nan_count = df.isnull().sum().sum()
            inf_count = np.isinf(df.select_dtypes(include=[np.number]).values).sum()
            
            if nan_count > 0:
                logger.warning(f"⚠️ Analysis window contains {nan_count} NaN values")
            if inf_count > 0:
                logger.warning(f"⚠️ Analysis window contains {inf_count} infinite values")
            
            if nan_count == 0 and inf_count == 0:
                logger.info("✅ Analysis window clean - no NaN or infinite values")
            
            # Get the state for the model
            state = self._prepare_state(df)
            
            if state is not None:
                logger.info(f"State prepared successfully. Shape: {state.shape}")
                
                # Log current market conditions
                self._log_market_conditions()
                
                # Get action from model
                action = self._get_action(state)
                
                # Execute action (only during market hours)
                self._execute_action(action)
                
                # Update session
                self._update_trading_session()
                
                logger.info("Trading decision completed")
            else:
                logger.warning("Failed to prepare state - skipping trading decision")
                
            logger.info("="*60)
                
        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data - gaps should already be handled by gap filler thread"""
        try:
            # Ensure timestamp is properly formatted and sorted
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic data quality checks - gaps should already be filled by gap filler
            price_cols = ['open', 'high', 'low', 'close', 'vwap']
            volume_cols = ['volume']
            
            # Ensure no negative prices or volumes (sanity check)
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].clip(lower=self.MIN_PRICE)  # Use constant for minimum price
            
            for col in volume_cols:
                if col in df.columns:
                    df[col] = df[col].clip(lower=0)  # No negative volume
            
            # Log data quality
            if df.isnull().any().any():
                null_counts = df.isnull().sum()
                logger.warning(f"Data contains nulls: {null_counts[null_counts > 0].to_dict()}")
            
            logger.info(f"Data validation complete: {len(df)} rows processed")
            return df
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            return df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle infinite and NaN values comprehensively (adapted from feature_engineering_2.py)
        """
        try:
            # Get numeric columns only (exclude timestamp)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Replace infinite values with NaN first
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # Handle NaN values with appropriate strategies (adapted from feature_engineering_2.py)
            for col in numeric_cols:
                if df[col].isnull().any():
                    # For ratio features, use 1.0 (neutral ratio)
                    if any(keyword in col.lower() for keyword in ['ratio', '_to_', 'normalized']):
                        df[col] = df[col].fillna(1.0)
                    # For percentage features, use 0.0 (neutral percentage)
                    elif any(keyword in col.lower() for keyword in ['percent', 'pct', '_b']):
                        df[col] = df[col].fillna(0.0)
                    # For oscillators and indicators, use median
                    elif any(keyword in col.lower() for keyword in ['rsi', 'stoch', 'williams', 'cci', 'mfi', 'macd']):
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                    # For volume features, use 0.0
                    elif 'volume' in col.lower():
                        df[col] = df[col].fillna(0.0)
                    # For volatility features, use median
                    elif any(keyword in col.lower() for keyword in ['vol', 'atr', 'std']):
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                    # For return features, use 0.0 (no return)
                    elif 'return' in col.lower() or 'roc' in col.lower():
                        df[col] = df[col].fillna(0.0)
                    # For momentum features, use 0.0 (no momentum)
                    elif 'momentum' in col.lower():
                        df[col] = df[col].fillna(0.0)
                    # For all other features, use forward fill then 0
                    else:
                        df[col] = df[col].fillna(method='ffill').fillna(0.0)
            
            # Final replacement of any remaining problematic values
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf, np.nan], 0.0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling infinite values: {e}")
            return df
    
    def _log_market_conditions(self):
        """Log current market conditions (extracted for code cleanup)"""
        latest_data = self.data_buffer[-1]
        portfolio_value = self.balance + (self.position * self.current_price)
        
        logger.info("Current Market Conditions:")
        logger.info(f"  Price: ${self.current_price:.2f}")
        logger.info(f"  Latest OHLC: O=${latest_data['open']:.2f}, H=${latest_data['high']:.2f}, L=${latest_data['low']:.2f}, C=${latest_data['close']:.2f}")
        logger.info(f"  Volume: {latest_data['volume']:,}")
        
        logger.info("Current Portfolio Status:")
        logger.info(f"  Cash Balance: ${self.balance:.2f}")
        logger.info(f"  Position: {self.position} shares")
        logger.info(f"  Portfolio Value: ${portfolio_value:.2f}")
        logger.info(f"  P&L: ${portfolio_value - self.initial_balance:.2f} ({((portfolio_value/self.initial_balance - 1) * 100):.2f}%)")
    
    def _prepare_state(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare state tensor for the model"""
        try:
            logger.info("Preparing state for model inference...")
            
            # Select features based on model type
            if self.model_type == "DQN":
                from src.config.config import STOCK_FEATURES_V2
                
                logger.info(f"DataFrame shape: {df.shape}")
                logger.info(f"DataFrame columns ({len(df.columns)}): {list(df.columns)[:10]}...")  # First 10 columns
                
                # Make sure we have all required features
                available_features = [col for col in STOCK_FEATURES_V2 if col in df.columns]
                logger.info(f"Available stock features: {len(available_features)}/{len(STOCK_FEATURES_V2)}")
                
                if len(available_features) < len(STOCK_FEATURES_V2) * 0.8:  # Allow some missing features
                    logger.warning(f"Missing features: {set(STOCK_FEATURES_V2) - set(available_features)}")
                    return None
                
                # Get stock data - select only the features expected by the model
                if self.preprocessor and hasattr(self.preprocessor, 'feature_names'):
                    # Use the same features the preprocessor was trained on
                    features_to_use = self.preprocessor.feature_names
                    stock_data = df[features_to_use].values
                    logger.info(f"Using {len(features_to_use)} features from preprocessor")
                else:
                    # Use the available STOCK_FEATURES_V2
                    stock_data = df[available_features].values
                    logger.info(f"Using {len(available_features)} stock features (no preprocessor feature names)")
                
                # Apply preprocessing if available
                if self.preprocessor:
                    try:
                        stock_data = self.preprocessor.transform(df[available_features if not hasattr(self.preprocessor, 'feature_names') else features_to_use])
                        logger.info(f"Preprocessor transformed data shape: {stock_data.shape}")
                    except Exception as e:
                        logger.error(f"Preprocessor transform failed: {e}")
                        logger.warning("Using raw features instead")
                        stock_data = df[available_features].values
                
                # Verify the shape matches what the model expects
                expected_stock_features = self.config.num_stock_features if hasattr(self, 'config') else 39
                if stock_data.shape[1] != expected_stock_features:
                    logger.error(f"Feature count mismatch! Got {stock_data.shape[1]} features, expected {expected_stock_features}")
                    logger.info(f"Available columns in df: {list(df.columns)}")
                    # Try to fix by padding or truncating
                    if stock_data.shape[1] < expected_stock_features:
                        # Pad with zeros
                        padding = np.zeros((stock_data.shape[0], expected_stock_features - stock_data.shape[1]))
                        stock_data = np.concatenate([stock_data, padding], axis=1)
                        logger.warning(f"Padded features from {stock_data.shape[1]} to {expected_stock_features}")
                    else:
                        # Truncate
                        stock_data = stock_data[:, :expected_stock_features]
                        logger.warning(f"Truncated features from {stock_data.shape[1]} to {expected_stock_features}")
                
                # Calculate portfolio features
                portfolio_value = self.balance + (self.position * self.current_price)
                
                # Update unrealized P&L if holding position
                if self.position > 0:
                    cost_basis = self.position * self.entry_price * (1 + TRANSACTION_FEE_PERCENT)
                    market_value = self.position * self.current_price
                    self.unrealized_pnl = (market_value - cost_basis) / cost_basis
                else:
                    self.unrealized_pnl = 0.0
                
                # Update max portfolio value for drawdown calculation
                self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
                
                # Calculate time and session features
                current_time = datetime.now(timezone.utc)
                
                # For live trading, we'll approximate the time within trading day
                # Convert to Eastern time to match market hours
                eastern_time = current_time.astimezone(timezone(timedelta(hours=-5)))  # EST approximation
                market_open_time = eastern_time.replace(hour=9, minute=30, second=0, microsecond=0)
                
                # Calculate minutes into trading day (regular market hours: 390 minutes)
                if eastern_time >= market_open_time:
                    minutes_into_day = (eastern_time - market_open_time).total_seconds() / 60
                    minutes_into_day = max(0, min(minutes_into_day, 390))  # Cap at 390 minutes
                else:
                    minutes_into_day = 0
                
                time_of_day_normalized = minutes_into_day / 390  # Normalize to 0-1
                
                # Market session features (matching DQN training)
                morning_session = 1.0 if minutes_into_day < 120 else 0.0  # First 2 hours (9:30-11:30)
                midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0  # Middle 2.5 hours (11:30-2:00)
                afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0  # Last 2 hours (2:00-4:00)
                
                # Position timing features
                position_holding_time = self.step_count - self.position_entry_step if self.position_entry_step >= 0 else 0
                normalized_holding_time = min(position_holding_time / 60, 1.0)  # Normalize to 1 hour max
                
                # Enhanced portfolio features matching DQN training environment (12 features)
                normalized_balance = self.balance / self.initial_balance if self.initial_balance > 0 else 0
                normalized_position = self.position * self.current_price / self.initial_balance if self.initial_balance > 0 else 0
                normalized_portfolio_value = portfolio_value / self.initial_balance if self.initial_balance > 0 else 0
                position_ratio = self.position * self.current_price / portfolio_value if portfolio_value > 0 else 0
                
                # Action validity flags (match execution logic exactly)
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = int(position_value / self.current_price) if self.current_price > 0 else 0
                total_cost = shares_to_buy * self.current_price * (1 + TRANSACTION_FEE_PERCENT)

                can_buy = 1.0 if (self.position == 0 and 
                                shares_to_buy > 0 and 
                                total_cost <= self.balance) else 0.0
                can_sell = 1.0 if self.position > 0 else 0.0
                
                # ✅ Portfolio features: Current situation only (12 features)
                portfolio_features = np.array([
                    # Core current metrics (4)
                    normalized_balance,           # Current cash available
                    normalized_position,          # Current stock holdings
                    normalized_portfolio_value,   # Current total value
                    position_ratio,              # Current position size ratio
                    
                    # Current position status (2)
                    self.unrealized_pnl,         # Current position P&L
                    normalized_holding_time,     # How long holding current position
                    
                    # Current timing context (4)
                    time_of_day_normalized,      # Where in trading day
                    morning_session,             # Current market session
                    midday_session,
                    afternoon_session,
                    
                    # Current action validity (2)
                    can_buy,                     # Can execute buy now
                    can_sell                     # Can execute sell now
                ])
                
                # Ensure all values are finite and increment step count
                portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=1.0, neginf=-1.0)
                self.step_count += 1
                
                # Combine features
                portfolio_features_repeated = np.tile(portfolio_features, (WINDOW_SIZE, 1))
                combined_features = np.concatenate([stock_data, portfolio_features_repeated], axis=1)
                
                # Convert to tensor
                state = torch.tensor(combined_features, dtype=torch.float32, device=DEVICE)
                
                # Log final state details
                logger.info(f"Final state tensor created:")
                logger.info(f"  Shape: {state.shape} (expected: [{WINDOW_SIZE}, {expected_stock_features + 12}])")
                logger.info(f"  Stock features: {expected_stock_features}")
                logger.info(f"  Portfolio features: 12")
                logger.info(f"  Total features per timestep: {state.shape[1]}")
                
                # Check for any data quality issues
                if torch.isnan(state).any():
                    logger.warning("State contains NaN values!")
                if torch.isinf(state).any():
                    logger.warning("State contains Inf values!")
                
                return state
            
        except Exception as e:
            logger.error(f"Error preparing state: {e}")
            return None
    
    def _get_action(self, state: torch.Tensor) -> int:
        """Get action from the model"""
        logger.info("Model evaluating state...")
        
        # Log state statistics for debugging
        logger.info(f"State shape: {state.shape}")
        logger.info(f"State device: {state.device}")
        logger.info(f"State dtype: {state.dtype}")
        
        # Log statistical summary of the state
        state_np = state.cpu().numpy()
        logger.info(f"State statistics:")
        logger.info(f"  Min: {state_np.min():.6f}")
        logger.info(f"  Max: {state_np.max():.6f}")
        logger.info(f"  Mean: {state_np.mean():.6f}")
        logger.info(f"  Std: {state_np.std():.6f}")
        logger.info(f"  Contains NaN: {np.isnan(state_np).any()}")
        logger.info(f"  Contains Inf: {np.isinf(state_np).any()}")
        
        # Log sample of state values (first and last timestep)
        logger.info("Sample state values:")
        logger.info(f"  First timestep stock features (first 6): {state_np[0, :6]}")
        logger.info(f"  Last timestep stock features (first 6): {state_np[-1, :6]}")
        logger.info(f"  Portfolio features: {state_np[0, -12:]}")  # Last 12 features are portfolio
        
        with torch.no_grad():
            if self.model_type == "DQN":
                # Add state dimension for batch
                state_batch = state.unsqueeze(0).to(self.device)
                logger.info(f"State batch shape for model: {state_batch.shape}")
                
                # Get Q-values
                q_values = self.agent.q_network(state_batch)
                logger.info(f"Q-values: {q_values.cpu().numpy()}")
                
                # Get action
                action = q_values.max(1)[1].item()
                
                # Log Q-value statistics
                q_np = q_values.cpu().numpy()[0]
                logger.info(f"Q-value analysis:")
                logger.info(f"  Q(Hold): {q_np[0]:.6f}")
                logger.info(f"  Q(Buy): {q_np[1]:.6f}")
                logger.info(f"  Q(Sell): {q_np[2]:.6f}")
                logger.info(f"  Best action: {['HOLD', 'BUY', 'SELL'][action]} (Q={q_np[action]:.6f})")
                
            else:
                raise NotImplementedError(f"Model type {self.model_type} not supported")
        
        return action
    
    def _execute_action(self, action: int):
        """Execute trading action via Alpaca"""
        try:
            # Final safety check - don't execute trades outside market hours
            current_time = datetime.now(timezone.utc)
            if not self._is_market_open(current_time):
                logger.warning(f"⚠️ Trade execution blocked - market is closed")
                logger.info(f"Action {action} would have been executed but market hours restriction prevented it")
                return
            
            # DQN actions: 0=Hold, 1=Buy, 2=Sell
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            logger.info(f"Model decision: {action_names.get(action, 'UNKNOWN')} (action={action})")
            
            if action == 0:  # Hold
                # Reset consecutive invalid actions on valid action
                self.consecutive_invalid_actions = 0
                self.last_action = 0  # Track last action as HOLD
                logger.info(f"HOLD - Current position: {self.position} shares, Balance: ${self.balance:.2f}")
            
            elif action == 1:  # Buy
                if self.position > 0:  # Already have position
                    self.invalid_actions += 1
                    self.consecutive_invalid_actions += 1
                    logger.warning(f"Invalid BUY action - Already holding {self.position} shares")
                    return
                
                # Calculate position size
                position_value = self.balance * self.max_position_size
                shares_to_buy = int(position_value / self.current_price)
                
                if shares_to_buy <= 0:
                    self.invalid_actions += 1
                    self.consecutive_invalid_actions += 1
                    logger.warning(f"Invalid BUY action - Insufficient balance (${self.balance:.2f}) to buy at ${self.current_price}")
                    return
                
                # Reset consecutive invalid actions on valid action
                self.consecutive_invalid_actions = 0
                
                # Place buy order with proper error handling
                try:
                    order_request = MarketOrderRequest(
                        symbol=self.ticker,
                        qty=shares_to_buy,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    logger.info(f"Submitting BUY order to Alpaca for {shares_to_buy} shares...")
                    order = self.alpaca_client.submit_order(order_request)
                    logger.info(f"Order submitted successfully. Order ID: {order.id}")
                    
                    # Update local state
                    cost = shares_to_buy * self.current_price * (1 + TRANSACTION_FEE_PERCENT)
                    self.balance -= cost
                    self.position = shares_to_buy
                    self.entry_price = self.current_price
                    self.position_entry_step = self.step_count  # Track when position was entered
                    self.last_action = 1  # Track last action as BUY
                    
                    # Save trade
                    self._save_trade(TradeType.BUY, cost, self.current_price, shares_to_buy)
                    
                    logger.info(f"✅ BUY executed: {shares_to_buy} shares of {self.ticker} at ${self.current_price:.2f}")
                    logger.info(f"   Cost: ${cost:.2f} (including fees), New balance: ${self.balance:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to submit BUY order: {e}")
                    self.invalid_actions += 1
                    return
            
            elif action == 2:  # Sell
                if self.position <= 0:  # No position to sell
                    self.invalid_actions += 1
                    self.consecutive_invalid_actions += 1
                    logger.warning(f"Invalid SELL action - No position to sell (position={self.position})")
                    return
                
                # Reset consecutive invalid actions on valid action
                self.consecutive_invalid_actions = 0
                
                # Place sell order with proper error handling
                try:
                    order_request = MarketOrderRequest(
                        symbol=self.ticker,
                        qty=self.position,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    logger.info(f"Submitting SELL order to Alpaca for {self.position} shares...")
                    order = self.alpaca_client.submit_order(order_request)
                    logger.info(f"Order submitted successfully. Order ID: {order.id}")
                    
                    # Calculate profit/loss
                    revenue = self.position * self.current_price * (1 - TRANSACTION_FEE_PERCENT)
                    cost_basis = self.position * self.entry_price * (1 + TRANSACTION_FEE_PERCENT)
                    profit = revenue - cost_basis
                    profit_percent = (profit / cost_basis) * 100 if cost_basis > 0 else 0
                    
                    # Update statistics
                    self.total_trades += 1
                    if profit > 0:
                        self.winning_trades += 1
                        self.total_profit += profit
                    else:
                        self.losing_trades += 1
                        self.total_loss += abs(profit)
                    
                    # Update local state
                    self.balance += revenue
                    shares_sold = self.position
                    self.position = 0
                    self.position_entry_step = -1  # Reset position entry tracking
                    self.last_action = 2  # Track last action as SELL
                    
                    # Save trade
                    self._save_trade(TradeType.SELL, revenue, self.current_price, shares_sold)
                    
                    logger.info(f"✅ SELL executed: {shares_sold} shares of {self.ticker} at ${self.current_price:.2f}")
                    logger.info(f"   Entry: ${self.entry_price:.2f}, Exit: ${self.current_price:.2f}")
                    logger.info(f"   Profit: ${profit:.2f} ({profit_percent:.2f}%), New balance: ${self.balance:.2f}")
                    
                    # Calculate and log win rate
                    win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                    logger.info(f"   Stats: {self.winning_trades}W/{self.losing_trades}L (Win rate: {win_rate:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to submit SELL order: {e}")
                    self.invalid_actions += 1
                    return
                
        except Exception as e:
            logger.error(f"❌ Error executing action {action}: {e}")
            self.invalid_actions += 1
    
    def start(self):
        """Start the trading bot"""
        self.is_running = True
        logger.info(f"Starting paper trading bot for model {self.model_id} on {self.ticker}")
        logger.info(f"Initial balance: ${self.initial_balance}")
        logger.info(f"Max position size: {self.max_position_size * 100}%")
        logger.info(f"Window size for analysis: {WINDOW_SIZE}")
        
        # Log market hours information
        current_time = datetime.now(timezone.utc)
        logger.info(f"🕐 TRADING HOURS RESTRICTION:")
        logger.info(f"   Trading decisions will ONLY be made during regular market hours (9:30 AM - 4:00 PM EST)")
        logger.info(f"   Data collection will continue 24/7 for buffer maintenance")
        logger.info(f"   Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"   Market status: {'🟢 OPEN' if self._is_market_open(current_time) else '🔴 CLOSED'}")
        
        # Start gap filler thread
        logger.info("Starting gap filler thread...")
        self.gap_filler_thread = threading.Thread(target=self._check_and_fill_gaps, daemon=True)
        self.gap_filler_thread.start()
        
        logger.info(f"Starting Polygon websocket connection...")
        
        # Run the websocket client
        self.polygon_client.run(self.handle_message)
        
        logger.info("Polygon websocket client started and running")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info(f"Stopping paper trading bot for model {self.model_id}...")
        
        # Stop gap filler thread
        if self.gap_filler_thread:
            logger.info("Stopping gap filler thread...")
            self.gap_filler_stop_event.set()
            self.gap_filler_thread.join(timeout=2)
            if self.gap_filler_thread.is_alive():
                logger.warning("Gap filler thread did not stop cleanly")
            else:
                logger.info("Gap filler thread stopped successfully")
        
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
                    
                    session.final_balance = Decimal(str(final_balance))
                    session.net_profit = Decimal(str(final_balance - self.initial_balance))
                    
                    # Calculate return rate with division by zero check
                    if self.initial_balance > 0:
                        return_rate = (final_balance - self.initial_balance) / self.initial_balance
                        session.return_rate = Decimal(str(return_rate))
                    else:
                        session.return_rate = Decimal('0')
                        
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