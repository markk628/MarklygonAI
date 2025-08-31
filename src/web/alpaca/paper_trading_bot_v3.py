import asyncio
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List
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

logger.info("Enhanced Paper Trading Bot v3 (DQN v7 Compatible) module loaded")

class EnhancedPaperTradingBot:
    """
    Enhanced Paper Trading Bot for DQN v7 models
    
    Features:
    - 7-action space support (HOLD, BUY_S/M/L, SELL_S/M/L)
    - 30 portfolio features state preparation
    - Action masking integration
    - Real-time data streaming from Polygon WebSocket
    - Automatic position management and risk controls
    - Market hours enforcement (9:30 AM - 4:00 PM EST)
    - Automatic position closing before market close
    - Daily portfolio snapshots
    - Comprehensive logging and trade tracking
    """
    
    # Constants for better maintainability
    MIN_PRICE = 0.01  # Minimum valid price (1 cent)
    DATA_BUFFER_MULTIPLIER = 4  # Buffer size = WINDOW_SIZE * 4 for adequate feature engineering
    GAP_CHECK_INTERVAL = 65  # Seconds between gap checks
    MAX_GAP_FILL_MINUTES = 5  # Maximum minutes to forward fill
    TRADE_EXECUTION_TIMEOUT = 30  # Seconds to wait for trade execution
    MARKET_CLOSE_WARNING_MINUTES = 15  # Minutes before market close to start warning
    FORCE_CLOSE_MINUTES = 5  # Minutes before market close to force close positions
    
    # Action mappings for DQN v7
    ACTION_NAMES = {
        0: "HOLD",
        1: "BUY_SMALL",
        2: "BUY_MEDIUM", 
        3: "BUY_LARGE",
        4: "SELL_SMALL",
        5: "SELL_MEDIUM",
        6: "SELL_LARGE"
    }
    
    def __init__(self, model_id: int, initial_balance: float, max_position_size: float = 0.7, portfolio_id: int = None):
        self.model_id = model_id
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position_size = max_position_size
        self.portfolio_id = portfolio_id
        self.position = 0
        self.entry_price = 0
        self.current_price = 0
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        # Enhanced tracking for DQN v7 (30-feature state)
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = initial_balance
        self.position_entry_step = -1  # Track when position was entered
        self.consecutive_invalid_actions = 0
        self.consecutive_holds = 0
        self.last_action = 0  # Track last action
        self.unrealized_pnl = 0.0
        self.step_count = 0  # Track steps for timing calculations
        
        # Enhanced tracking for 7-action space
        self.action_counts = [0] * 7  # Track usage of each action
        self.successful_trades = [0] * 7  # Track successful trades per action
        self.last_position_change_step = -1
        self.last_trade_step = -100  # For trade cooldowns
        
        # Trading session tracking
        self.session_start_time = datetime.now(timezone.utc)
        
        # Data buffer for windowed analysis - increased size for better feature engineering
        buffer_size = WINDOW_SIZE * self.DATA_BUFFER_MULTIPLIER  # 120 minutes for 30-minute window
        self.data_buffer = deque(maxlen=buffer_size)
        logger.info(f"Initialized data buffer with size: {buffer_size} minutes")
        
        self.is_running = False
        self.ticker = None
        self.device = DEVICE
        
        # Load model and preprocessor for DQN v7
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
        
        # Initialize daily snapshot tracking
        self.last_snapshot_date = None  # Track the last day we captured a snapshot
        
        logger.info(f"✅ Enhanced Paper Trading Bot v3 initialized for DQN v7")
        logger.info(f"   Action space: 7 actions (HOLD, BUY_S/M/L, SELL_S/M/L)")
        logger.info(f"   Portfolio features: 30 enhanced features")
        logger.info(f"   Action masking: Enabled")
        logger.info(f"   Portfolio ID: {self.portfolio_id}")
        if self.portfolio_normalizer and self.portfolio_normalizer.is_fitted:
            logger.info(f"   Portfolio normalizer: ✅ LOADED (features: {list(self.portfolio_normalizer.feature_stats.keys())})")
        else:
            logger.info(f"   Portfolio normalizer: ❌ NOT AVAILABLE")
    
    def _initialize_portfolio_history(self):
        """Initialize portfolio history buffer with initial state"""
        initial_portfolio_state = self._calculate_current_portfolio_state()
        
        # Fill portfolio history with initial state
        for _ in range(WINDOW_SIZE):
            self.portfolio_history.append(initial_portfolio_state)
    
    def _calculate_current_portfolio_state(self) -> np.ndarray:
        """Calculate current portfolio state features (same as DQN environment)"""
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
        if hasattr(self, 'config') and self.config:
            max_position_size = self.config.max_position_size
        else:
            max_position_size = self.max_position_size
            
        position_value = self.balance * max_position_size
        shares_to_buy = int(position_value / self.current_price) if self.current_price > 0 else 0
        total_cost = shares_to_buy * self.current_price * (1 + TRANSACTION_FEE_PERCENT)

        can_buy = 1.0 if (self.position == 0 and 
                        shares_to_buy > 0 and 
                        total_cost <= self.balance) else 0.0
        can_sell = 1.0 if self.position > 0 else 0.0
        
        # Portfolio features (12 features)
        portfolio_features = [
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
        ]
        
        # Ensure all values are finite
        portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
        
        return np.array(portfolio_features, dtype=np.float32)
    
    def _load_model_and_config(self):
        """Load DQN v7 model configuration and setup"""
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
        
        # Initialize model and preprocessor for DQN v7
        self._load_model()
        self._load_preprocessor()
        # CRITICAL: Load portfolio normalizer for DQN v7 consistency
        self._load_portfolio_normalizer()
        
        # Initialize Alpaca client
        self._initialize_alpaca_client()
        
        # Initialize Polygon websocket
        self._initialize_polygon_client()
        
        # Create trading session in database
        self.session_id = self._create_trading_session()
        
        # Gap filling setup
        self._initialize_gap_filling()
        
        # Initialize daily snapshot tracking
        self.last_snapshot_date = None  # Track the last day we captured a snapshot
        
        logger.info(f"✅ Enhanced Paper Trading Bot v3 initialized for DQN v7")
        logger.info(f"   Action space: 7 actions (HOLD, BUY_S/M/L, SELL_S/M/L)")
        logger.info(f"   Portfolio features: 30 enhanced features")
        logger.info(f"   Action masking: Enabled")
        logger.info(f"   Portfolio ID: {self.portfolio_id}")
        if self.portfolio_normalizer and self.portfolio_normalizer.is_fitted:
            logger.info(f"   Portfolio normalizer: ✅ LOADED (features: {list(self.portfolio_normalizer.feature_stats.keys())})")
        else:
            logger.info(f"   Portfolio normalizer: ❌ NOT AVAILABLE")
    
    def _load_portfolio_normalizer(self):
        """Load the portfolio normalizer if available"""
        self.portfolio_normalizer = None
        
        if not self.model_path:
            logger.warning("No model path available - cannot load portfolio normalizer")
            return
        
        try:
            # Import the normalizer class
            from src.models.mark.dqn_v2.normalization import PortfolioStateNormalizer
            
            # Try both naming conventions (DQN v7 uses _normalizer.pkl, DQN v4 uses _portfolio_normalizer.pkl)
            normalizer_paths = [
                self.model_path.replace('.pt', '_normalizer.pkl'),           # DQN v7 format
                self.model_path.replace('.pt', '_portfolio_normalizer.pkl')  # DQN v4 format
            ]
            
            normalizer_loaded = False
            for normalizer_path in normalizer_paths:
                try:
                    # Create normalizer instance and try to load
                    self.portfolio_normalizer = PortfolioStateNormalizer()
                    self.portfolio_normalizer.load(normalizer_path)
                    
                    if self.portfolio_normalizer.is_fitted:
                        logger.info(f"✅ Loaded portfolio normalizer from {normalizer_path}")
                        logger.info(f"   Normalizer is fitted and ready for inference")
                        logger.info(f"   Normalized features: {list(self.portfolio_normalizer.feature_stats.keys())}")
                        normalizer_loaded = True
                        break
                    else:
                        logger.warning(f"Portfolio normalizer at {normalizer_path} loaded but not fitted")
                        
                except FileNotFoundError:
                    continue  # Try next path
                except Exception as e:
                    logger.warning(f"Error loading normalizer from {normalizer_path}: {e}")
                    continue
            
            if not normalizer_loaded:
                logger.info(f"No portfolio normalizer found - model may not use portfolio normalization")
                logger.info(f"   Tried paths: {normalizer_paths}")
                self.portfolio_normalizer = None
                
        except Exception as e:
            logger.error(f"Failed to load portfolio normalizer: {e}")
            logger.warning("Continuing without portfolio normalizer")
            self.portfolio_normalizer = None
    
    def _load_model(self):
        """Load the trained DQN v7 model"""
        try:
            if self.model_type == "DQN":
                try:
                    # Try to load DQN v7 model first
                    from src.models.mark.dqn_v2.dqn_v7 import EnhancedDoubleDuelingDQN, EnhancedTradingConfig
                    
                    self.config = EnhancedTradingConfig()
                    self.agent = EnhancedDoubleDuelingDQN(self.config, device=DEVICE)
                    # Load model weights (portfolio normalizer is handled separately by paper trading bot)
                    self.agent.load(self.model_path)
                    self.agent.q_network.eval()
                    logger.info(f"✅ Loaded Enhanced DQN v7 model from {self.model_path}")
                    self.is_enhanced_dqn = True
                    
                except ImportError:
                    # Fallback to regular DQN v5
                    logger.warning("DQN v7 not available, falling back to DQN v5")
                    from src.models.mark.dqn_v2.dqn_v5 import DoubleDuelingDQN, TradingConfig
                
                    self.config = TradingConfig()
                    self.agent = DoubleDuelingDQN(self.config, device=DEVICE)
                    # Load model weights (portfolio normalizer is handled separately by paper trading bot)
                    self.agent.load(self.model_path)
                    self.agent.q_network.eval()
                    logger.info(f"✅ Loaded DQN v5 model from {self.model_path}")
                    self.is_enhanced_dqn = False
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
            # Import the preprocessor class
            from src.models.mark.dqn_v2.data_preprocessor_v2 import FinancialDataPreprocessor
            
            # Load using the class method
            self.preprocessor = FinancialDataPreprocessor.load(self.preprocessor_path)
            logger.info(f"✅ Loaded preprocessor from {self.preprocessor_path}")
            
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
                        logger.info("✅ Reconstructed preprocessor from old format")
            except Exception as e2:
                logger.error(f"Alternative loading also failed: {e2}")
                self.preprocessor = None
    
    def _initialize_alpaca_client(self):
        """Initialize Alpaca trading client"""
        try:
            self.alpaca_client = TradingClient(ALPACA_APIKEY, ALPACA_SECRET_KEY, paper=True)
            logger.info("✅ Alpaca client initialized successfully")
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
            logger.info("✅ Polygon websocket client initialized with RealTime feed for Stocks market")
            
            # Subscribe to minute aggregates for the ticker
            subscription = f"AM.{self.ticker}"
            logger.info(f"Subscribing to Polygon websocket for: {subscription}")
            self.polygon_client.subscribe(subscription)
            logger.info(f"✅ Successfully subscribed to {subscription}")
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            raise
    
    def _initialize_gap_filling(self):
        """Initialize gap filling setup"""
        self.last_data_timestamp = None
        self.last_data_point = None
        self.gap_filler_thread = None
        self.gap_filler_stop_event = threading.Event()
    
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
        """Save trade to database with proper error handling"""
        try:
            with app.app_context():
                # Use the specific portfolio ID passed to the constructor
                if not self.portfolio_id:
                    logger.error("No portfolio ID provided to trading bot! Cannot save trade.")
                    return
                
                # Verify the portfolio exists
                portfolio = Portfolio.query.get(self.portfolio_id)
                if not portfolio:
                    logger.error(f"Portfolio {self.portfolio_id} not found! Cannot save trade.")
                    return
                
                portfolio_id = portfolio.id
                
                # Verify trading session exists
                session = TradingSession.query.get(self.session_id)
                if not session:
                    logger.error(f"Trading session {self.session_id} not found! Cannot save trade.")
                    return
                
                # Create trade record
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
                
                logger.info(f"✅ Trade saved to database:")
                logger.info(f"   Trade ID: {trade.id}")
                logger.info(f"   Type: {trade_type.value}")
                logger.info(f"   Amount: ${amount:.2f}")
                logger.info(f"   Price: ${price:.2f}")
                logger.info(f"   Shares: {shares}")
                logger.info(f"   Portfolio ID: {portfolio_id}")
                logger.info(f"   Session ID: {self.session_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save trade to database: {e}")
            logger.error(f"   Trade details: {trade_type.value}, ${amount:.2f}, ${price:.2f}, {shares} shares")
            # Rollback any partial transaction
            try:
                db.session.rollback()
            except:
                pass    
    
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
    
    def _is_approaching_market_close(self, check_time: datetime = None, warning_minutes: int = None) -> tuple[bool, int]:
        """
        Check if we're approaching market close
        Returns: (is_approaching, minutes_until_close)
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        if warning_minutes is None:
            warning_minutes = self.MARKET_CLOSE_WARNING_MINUTES
        
        try:
            # Try to use pytz for proper timezone handling
            import pytz
            et = pytz.timezone('US/Eastern')
            et_time = check_time.astimezone(et)
        except ImportError:
            # Fallback to simple UTC offset (doesn't handle DST properly)
            et_time = check_time.astimezone(timezone(timedelta(hours=-5)))
        
        # Skip if not a trading day
        if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, 0
        
        market_close_time = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        time_until_close = (market_close_time - et_time).total_seconds() / 60  # Minutes
        
        # If market is already closed or will close soon
        if time_until_close <= 0:
            return False, 0
        
        is_approaching = time_until_close <= warning_minutes
        return is_approaching, int(time_until_close)
    
    def _force_close_position(self, reason: str = "Market closing soon"):
        """Force close any open position"""
        if self.position <= 0:
            logger.info(f"No position to close ({reason})")
            return True
        
        logger.warning(f"🚨 FORCE CLOSING POSITION: {reason}")
        logger.info(f"   Current position: {self.position} shares at ${self.current_price:.2f}")
        
        try:
            # Force sell using enhanced action 6 (SELL_LARGE) for DQN v7 or action 2 (SELL) for standard
            if self.is_enhanced_dqn:
                logger.info(f"Executing forced SELL_LARGE for market close protection...")
                self._execute_enhanced_sell(6)  # SELL_LARGE = 100% of position
            else:
                logger.info(f"Executing forced SELL for market close protection...")
                self._execute_standard_sell()
            
            # Verify position was closed
            if self.position <= 0:
                logger.info(f"✅ Position successfully closed due to: {reason}")
                return True
            else:
                logger.error(f"❌ Failed to close position - still holding {self.position} shares")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during forced position close: {e}")
            return False
            
    def _capture_daily_snapshot(self):
        """Capture a daily portfolio snapshot at the end of trading day"""
        if not self.portfolio_id:
            logger.warning("No portfolio ID provided to trading bot - cannot capture snapshot")
            return False
            
        try:
            with app.app_context():
                # Import the capture function from app.py
                from src.web.app import capture_portfolio_snapshot
                
                # Check if we already captured a snapshot today
                current_date = datetime.now(timezone.utc).date()
                
                if self.last_snapshot_date == current_date:
                    logger.info(f"Portfolio snapshot already captured today ({current_date})")
                    return True
                
                # Capture the snapshot
                logger.info(f"📸 Capturing daily portfolio snapshot for portfolio {self.portfolio_id}...")
                success = capture_portfolio_snapshot(self.portfolio_id)
                
                if success:
                    self.last_snapshot_date = current_date
                    logger.info(f"✅ Daily portfolio snapshot captured successfully for {current_date}")
                    return True
                else:
                    logger.error(f"❌ Failed to capture daily portfolio snapshot")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error capturing daily portfolio snapshot: {e}")
            return False
    
    def _check_and_fill_gaps(self):
        """Check for gaps in data and forward fill if necessary, also monitor for market close"""
        market_close_warning_logged = False
        last_close_check_minute = -1
        
        while not self.gap_filler_stop_event.is_set():
            try:
                # Wait for specified interval (slightly more than a minute to account for delays)
                self.gap_filler_stop_event.wait(self.GAP_CHECK_INTERVAL)
                
                if self.gap_filler_stop_event.is_set():
                    break
                
                current_time = datetime.now(timezone.utc)
                
                # 🚨 PRIORITY 1: Check for approaching market close
                is_approaching, minutes_until_close = self._is_approaching_market_close(current_time)
                current_minute = int(minutes_until_close)
                
                if is_approaching and self.position > 0:
                    # Log warning once per minute to avoid spam
                    if current_minute != last_close_check_minute:
                        if minutes_until_close <= self.FORCE_CLOSE_MINUTES:
                            logger.warning(f"🚨 CRITICAL: {minutes_until_close} minutes until market close - FORCE CLOSING POSITION!")
                            success = self._force_close_position(f"Force close - {minutes_until_close} minutes until market close")
                            if success:
                                logger.info(f"✅ Position closed successfully with {minutes_until_close} minutes to spare")
                                # Capture daily portfolio snapshot after position is closed
                                self._capture_daily_snapshot()
                            else:
                                logger.error(f"❌ Failed to close position - {minutes_until_close} minutes until close!")
                        elif not market_close_warning_logged or current_minute != last_close_check_minute:
                            logger.warning(f"⚠️ WARNING: {minutes_until_close} minutes until market close - holding position")
                            logger.info(f"   Position will be force-closed in {minutes_until_close - self.FORCE_CLOSE_MINUTES} minutes if not closed by model")
                            market_close_warning_logged = True
                        
                        last_close_check_minute = current_minute
                
                # Check if market is closing soon but no position to close
                elif is_approaching and self.position <= 0 and minutes_until_close <= self.FORCE_CLOSE_MINUTES:
                    # Capture daily snapshot even when no position is held
                    if current_minute != last_close_check_minute:
                        logger.info(f"📸 Market closing in {minutes_until_close} minutes - capturing daily snapshot (no position held)")
                        self._capture_daily_snapshot()
                        last_close_check_minute = current_minute
                
                # Reset warning flag when not approaching close
                if not is_approaching:
                    market_close_warning_logged = False
                    last_close_check_minute = -1
                
                # Continue with existing gap filling logic
                # Check if we have received any data
                if self.last_data_timestamp and self.last_data_point:
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
            if self.is_enhanced_dqn:
                logger.info("PROCESSING DATA FOR ENHANCED TRADING DECISION (DQN v7 - 7 ACTIONS)")
            else:
                logger.info("PROCESSING DATA FOR STANDARD TRADING DECISION (DQN v5 - 3 ACTIONS)")
            
            # Convert full buffer to DataFrame
            full_df = pd.DataFrame(list(self.data_buffer))
            logger.info(f"Full buffer DataFrame: {len(full_df)} rows")
            
            # Apply feature engineering in the correct order
            full_df = self.feature_engineer._add_price_features(full_df)
            full_df = self.feature_engineer._add_volume_features(full_df)
            full_df = self.feature_engineer._add_technical_indicators(full_df)
            full_df = self.feature_engineer._add_temporal_patterns(full_df)
            full_df = self.feature_engineer._add_market_context(full_df)
            full_df = self.feature_engineer._add_portfolio_features(full_df)
            full_df = self.feature_engineer._add_derived_features(full_df)
            
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
                
                # Get action from model with masking
                action = self._get_action_with_masking(state)
                
                # Execute action (only during market hours)
                self._execute_enhanced_action(action)
                
                # Update session
                self._update_trading_session()
                
                logger.info("Trading decision completed")
            else:
                logger.warning("Failed to prepare state - skipping trading decision")
                
            logger.info("="*60)
                
        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)
    
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
        
        if self.is_enhanced_dqn:
            logger.info("Enhanced Features:")
            logger.info(f"  Action counts: {dict(zip(self.ACTION_NAMES.values(), self.action_counts))}")
            logger.info(f"  Valid actions: {[self.ACTION_NAMES[a] for a in self.get_valid_actions()]}")
    
    def start(self):
        """Start the enhanced trading bot"""
        self.is_running = True
        logger.info(f"Starting Enhanced Paper Trading Bot v3 for model {self.model_id} on {self.ticker}")
        logger.info(f"Initial balance: ${self.initial_balance}")
        logger.info(f"Max position size: {self.max_position_size * 100}%")
        logger.info(f"Window size for analysis: {WINDOW_SIZE}")
        
        # Log enhanced features
        if self.is_enhanced_dqn:
            logger.info(f"🚀 ENHANCED DQN v7 FEATURES ACTIVE:")
            logger.info(f"   Action space: 7 actions (HOLD, BUY_S/M/L, SELL_S/M/L)")
            logger.info(f"   Portfolio features: 30 enhanced features")
            logger.info(f"   Action masking: Enabled")
            logger.info(f"   Trade cooldowns: Enabled")
        else:
            logger.info(f"📊 STANDARD DQN v5 COMPATIBILITY MODE:")
            logger.info(f"   Action space: 3 actions (HOLD, BUY, SELL)")
            logger.info(f"   Portfolio features: 13 standard features")
            logger.info(f"   Action masking: Disabled")
        
        # Log market hours information
        current_time = datetime.now(timezone.utc)
        logger.info(f"🕐 TRADING HOURS & RISK MANAGEMENT:")
        logger.info(f"   Trading decisions will ONLY be made during regular market hours (9:30 AM - 4:00 PM EST)")
        logger.info(f"   Data collection will continue 24/7 for buffer maintenance")
        logger.info(f"   📢 AUTOMATIC POSITION CLOSING:")
        logger.info(f"      - Warning starts {self.MARKET_CLOSE_WARNING_MINUTES} minutes before market close")
        logger.info(f"      - Positions will be FORCE-CLOSED {self.FORCE_CLOSE_MINUTES} minutes before market close")
        logger.info(f"      - This protects against overnight risk and ensures position closure")
        logger.info(f"   Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"   Market status: {'🟢 OPEN' if self._is_market_open(current_time) else '🔴 CLOSED'}")
        
        # Check if we're starting close to market close
        is_approaching, minutes_until_close = self._is_approaching_market_close(current_time)
        if is_approaching:
            logger.warning(f"⚠️ STARTING CLOSE TO MARKET CLOSE: {minutes_until_close} minutes remaining")
            if minutes_until_close <= self.FORCE_CLOSE_MINUTES:
                logger.warning(f"🚨 CRITICAL: Bot starting within force-close window!")
                logger.warning(f"   Any positions opened may be immediately closed!")
        elif self._is_market_open(current_time):
            is_approaching_warning, minutes_until_warning = self._is_approaching_market_close(current_time, self.MARKET_CLOSE_WARNING_MINUTES)
            if minutes_until_warning > 0:
                logger.info(f"✅ Safe trading window: {minutes_until_warning} minutes until close warnings begin")
        
        # Start gap filler thread
        logger.info("Starting gap filler thread...")
        self.gap_filler_thread = threading.Thread(target=self._check_and_fill_gaps, daemon=True)
        self.gap_filler_thread.start()
        
        logger.info(f"Starting Polygon websocket connection...")
        
        # Run the websocket client
        self.polygon_client.run(self.handle_message)
        
        logger.info("Polygon websocket client started and running")
    
    def stop(self):
        """Stop the enhanced trading bot"""
        self.is_running = False
        logger.info(f"Stopping Enhanced Paper Trading Bot v3 for model {self.model_id}...")
        
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
            logger.info(f"Closing open position of {self.position} shares during bot shutdown...")
            try:
                # Use force close method for more robust handling during shutdown
                success = self._force_close_position("Bot shutdown")
                if not success:
                    logger.warning("Force close failed, attempting regular sell action...")
                    if self.is_enhanced_dqn:
                        self._execute_enhanced_sell(6)  # SELL_LARGE
                    else:
                        self._execute_standard_sell()
            except Exception as e:
                logger.error(f"Error closing position during shutdown: {e}")
        
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
        
        # Log final statistics
        if self.is_enhanced_dqn:
            logger.info(f"📊 ENHANCED BOT STATISTICS:")
            logger.info(f"   Action usage: {dict(zip(self.ACTION_NAMES.values(), self.action_counts))}")
            logger.info(f"   Successful trades: {dict(zip(self.ACTION_NAMES.values(), self.successful_trades))}")
        
        logger.info(f"✅ Enhanced Paper Trading Bot v3 for model {self.model_id} stopped successfully")

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions for current state - KEY FEATURE FOR ACTION MASKING (DQN v7)"""
        if not self.is_enhanced_dqn:
            # For standard DQN, return all 3 actions
            return [0, 1, 2]
        
        valid_actions = [0]  # HOLD is always valid
        
        current_price = self.current_price if self.current_price > 0 else 100  # Fallback price
        
        # Check BUY actions (1, 2, 3) - with stricter requirements for enhanced trading
        available_cash = self.balance
        min_investment = self.config.initial_balance * getattr(self.config, 'min_trade_ratio', 0.05)
        
        # Only allow buying if we have sufficient cash and haven't traded recently
        steps_since_last_trade = self.step_count - self.last_trade_step
        min_steps_between_trades = getattr(self.config, 'min_steps_between_trades', 10)
        can_trade = steps_since_last_trade >= min_steps_between_trades
        
        if can_trade and available_cash > min_investment and self.position == 0:  # Only buy when no position
            
            # Get buy ratios from config
            buy_small_ratio = getattr(self.config, 'buy_small_ratio', 0.25)
            buy_medium_ratio = getattr(self.config, 'buy_medium_ratio', 0.50)
            buy_large_ratio = getattr(self.config, 'buy_large_ratio', 0.75)
            
            # BUY_SMALL (25% of available cash)
            small_investment = available_cash * buy_small_ratio
            if small_investment >= min_investment:
                shares_small = int(small_investment / current_price)
                cost_small = shares_small * current_price * (1 + TRANSACTION_FEE_PERCENT)
                if cost_small <= available_cash and shares_small > 0:
                    valid_actions.append(1)
            
            # BUY_MEDIUM (50% of available cash)
            medium_investment = available_cash * buy_medium_ratio
            if medium_investment >= min_investment:
                shares_medium = int(medium_investment / current_price)
                cost_medium = shares_medium * current_price * (1 + TRANSACTION_FEE_PERCENT)
                if cost_medium <= available_cash and shares_medium > 0:
                    valid_actions.append(2)
            
            # BUY_LARGE (75% of available cash)
            large_investment = available_cash * buy_large_ratio
            if large_investment >= min_investment:
                shares_large = int(large_investment / current_price)
                cost_large = shares_large * current_price * (1 + TRANSACTION_FEE_PERCENT)
                if cost_large <= available_cash and shares_large > 0:
                    valid_actions.append(3)
        
        # Check SELL actions (4, 5, 6) - with stricter requirements
        if self.position > 0 and can_trade:
            position_value = self.position * current_price
            min_sell_value = self.config.initial_balance * getattr(self.config, 'min_trade_ratio', 0.05)
            
            # Get sell ratios from config
            sell_small_ratio = getattr(self.config, 'sell_small_ratio', 0.25)
            sell_medium_ratio = getattr(self.config, 'sell_medium_ratio', 0.50)
            sell_large_ratio = getattr(self.config, 'sell_large_ratio', 1.00)
            
            # SELL_SMALL (25% of position)
            small_sell_shares = int(self.position * sell_small_ratio)
            small_sell_value = small_sell_shares * current_price
            if small_sell_value >= min_sell_value and small_sell_shares > 0:
                valid_actions.append(4)
            
            # SELL_MEDIUM (50% of position)
            medium_sell_shares = int(self.position * sell_medium_ratio)
            medium_sell_value = medium_sell_shares * current_price
            if medium_sell_value >= min_sell_value and medium_sell_shares > 0:
                valid_actions.append(5)
            
            # SELL_LARGE (100% of position) - always valid if we have position
            valid_actions.append(6)
        
        return valid_actions
    
    def _prepare_state(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare state tensor - route to appropriate method based on model type"""
        if self.is_enhanced_dqn:
            return self._prepare_enhanced_state(df)
        else:
            # Use standard state preparation for DQN v5 compatibility
            return self._prepare_standard_state(df)
    
    def _prepare_enhanced_state(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare enhanced state tensor with 30 portfolio features for DQN v7"""
        try:
            logger.info("Preparing enhanced state for DQN v7 model inference...")
            
            from src.config.config import STOCK_FEATURES_V2
            
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns ({len(df.columns)}): {list(df.columns)[:10]}...")
            
            # Make sure we have all required features
            available_features = [col for col in STOCK_FEATURES_V2 if col in df.columns]
            logger.info(f"Available stock features: {len(available_features)}/{len(STOCK_FEATURES_V2)}")
            
            if len(available_features) < len(STOCK_FEATURES_V2) * 0.8:
                logger.warning(f"Missing features: {set(STOCK_FEATURES_V2) - set(available_features)}")
                return None
            
            # Get stock data
            if self.preprocessor and hasattr(self.preprocessor, 'feature_names'):
                features_to_use = self.preprocessor.feature_names
                stock_data = df[features_to_use].values
                logger.info(f"Using {len(features_to_use)} features from preprocessor")
            else:
                stock_data = df[available_features].values
                logger.info(f"Using {len(available_features)} stock features")
            
            # Apply preprocessing if available
            if self.preprocessor:
                try:
                    stock_data = self.preprocessor.transform(df[available_features if not hasattr(self.preprocessor, 'feature_names') else features_to_use])
                    # Ensure stock_data is a numpy array (preprocessor might return DataFrame)
                    if hasattr(stock_data, 'values'):
                        stock_data = stock_data.values
                    logger.info(f"Preprocessor transformed data shape: {stock_data.shape}")
                except Exception as e:
                    logger.error(f"Preprocessor transform failed: {e}")
                    logger.warning("Using raw features instead")
                    stock_data = df[available_features].values
            
            # Verify the shape matches what the model expects
            expected_stock_features = getattr(self.config, 'num_stock_features', 39)
            if stock_data.shape[1] != expected_stock_features:
                logger.error(f"Feature count mismatch! Got {stock_data.shape[1]} features, expected {expected_stock_features}")
                # Try to fix by padding or truncating
                if stock_data.shape[1] < expected_stock_features:
                    padding = np.zeros((stock_data.shape[0], expected_stock_features - stock_data.shape[1]))
                    stock_data = np.concatenate([stock_data, padding], axis=1)
                    logger.warning(f"Padded features from {stock_data.shape[1]} to {expected_stock_features}")
                else:
                    stock_data = stock_data[:, :expected_stock_features]
                    logger.warning(f"Truncated features from {stock_data.shape[1]} to {expected_stock_features}")
            
            # Calculate enhanced portfolio features (30 features for DQN v7)
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
            eastern_time = current_time.astimezone(timezone(timedelta(hours=-5)))
            market_open_time = eastern_time.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Calculate minutes into trading day
            if eastern_time >= market_open_time:
                minutes_into_day = (eastern_time - market_open_time).total_seconds() / 60
                minutes_into_day = max(0, min(minutes_into_day, 390))
            else:
                minutes_into_day = 0
            
            time_of_day_normalized = minutes_into_day / 390
            
            # Market session features
            morning_session = 1.0 if minutes_into_day < 120 else 0.0
            midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0
            afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0
            
            # Position timing features
            position_holding_time = (self.step_count - self.position_entry_step) if self.position_entry_step >= 0 else 0
            position_ratio = self.position * self.current_price / portfolio_value if portfolio_value > 0 else 0
            
            # ENHANCED STATE FEATURES FOR DQN v7 (30 features total)
            # Calculate available action ratios for better decision making
            available_cash = self.balance
            position_value = self.position * self.current_price if self.position > 0 else 0
            
            # Cash utilization ratios
            buy_small_ratio = getattr(self.config, 'buy_small_ratio', 0.25)
            buy_medium_ratio = getattr(self.config, 'buy_medium_ratio', 0.50)
            buy_large_ratio = getattr(self.config, 'buy_large_ratio', 0.75)
            
            cash_ratio_small = min(1.0, (available_cash * buy_small_ratio) / max(available_cash, 1)) if available_cash > 0 else 0
            cash_ratio_medium = min(1.0, (available_cash * buy_medium_ratio) / max(available_cash, 1)) if available_cash > 0 else 0
            cash_ratio_large = min(1.0, (available_cash * buy_large_ratio) / max(available_cash, 1)) if available_cash > 0 else 0
            
            # Position utilization ratios
            sell_small_ratio = getattr(self.config, 'sell_small_ratio', 0.25)
            sell_medium_ratio = getattr(self.config, 'sell_medium_ratio', 0.50)
            sell_large_ratio = getattr(self.config, 'sell_large_ratio', 1.00)
            
            pos_ratio_small = sell_small_ratio if self.position > 0 else 0
            pos_ratio_medium = sell_medium_ratio if self.position > 0 else 0
            pos_ratio_large = sell_large_ratio if self.position > 0 else 0
            
            # Action opportunity indicators
            valid_actions = self.get_valid_actions()
            can_buy_small = 1.0 if 1 in valid_actions else 0.0
            can_buy_medium = 1.0 if 2 in valid_actions else 0.0
            can_buy_large = 1.0 if 3 in valid_actions else 0.0
            can_sell_small = 1.0 if 4 in valid_actions else 0.0
            can_sell_medium = 1.0 if 5 in valid_actions else 0.0
            can_sell_large = 1.0 if 6 in valid_actions else 0.0
            
            # Trading activity features
            steps_since_last_position_change = (self.step_count - self.last_position_change_step) if self.last_position_change_step >= 0 else 0
            
            # ENHANCED PORTFOLIO FEATURES (30 features total for DQN v7)
            portfolio_features = [
                # Basic portfolio features (0-5)
                self.balance / self.initial_balance if self.initial_balance > 0 else 0,  # 0: Normalized balance
                position_value / self.initial_balance if self.initial_balance > 0 else 0,  # 1: Normalized position value
                portfolio_value / self.initial_balance if self.initial_balance > 0 else 0,  # 2: Normalized portfolio value
                position_ratio,                  # 3: Position ratio (0-1)
                self.unrealized_pnl,            # 4: Unrealized P&L
                float(position_holding_time) / 60.0,  # 5: Normalized holding time (hours)
                
                # Time features (6-9)
                time_of_day_normalized,          # 6: Time of day (0-1)
                morning_session,                 # 7: Morning session binary
                midday_session,                  # 8: Midday session binary
                afternoon_session,               # 9: Afternoon session binary
                
                # Cash utilization features (10-12)
                cash_ratio_small,                # 10: Small buy ratio
                cash_ratio_medium,               # 11: Medium buy ratio
                cash_ratio_large,                # 12: Large buy ratio
                
                # Position utilization features (13-15)
                pos_ratio_small,                 # 13: Small sell ratio
                pos_ratio_medium,                # 14: Medium sell ratio
                pos_ratio_large,                 # 15: Large sell ratio
                
                # Action opportunity flags (16-21)
                can_buy_small,                   # 16: Can execute buy small
                can_buy_medium,                  # 17: Can execute buy medium
                can_buy_large,                   # 18: Can execute buy large
                can_sell_small,                  # 19: Can execute sell small
                can_sell_medium,                 # 20: Can execute sell medium
                can_sell_large,                  # 21: Can execute sell large
                
                # Trading activity features (22-29)
                float(steps_since_last_position_change) / 60.0, # 22: Steps since position change (hours)
                float(self.total_trades),        # 23: Total trades this episode
                float(self.winning_trades),      # 24: Winning trades
                float(self.losing_trades),       # 25: Losing trades
                float(len(valid_actions)),       # 26: Number of valid actions available
                float(self.last_action),         # 27: Last action taken
                float(self.consecutive_holds),   # 28: Consecutive holds
                float(self.invalid_actions),     # 29: Invalid actions count
            ]
            
            # Replace any non-finite values with 0
            portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
            
            # Ensure we have exactly 30 features
            if len(portfolio_features) != 30:
                logger.warning(f"Portfolio features count mismatch: got {len(portfolio_features)}, expected 30")
                # Pad or truncate to 30
                if len(portfolio_features) < 30:
                    portfolio_features.extend([0.0] * (30 - len(portfolio_features)))
                else:
                    portfolio_features = portfolio_features[:30]
            
            # Convert to numpy array
            portfolio_state = np.array(portfolio_features, dtype=np.float32)
            
            # CRITICAL: Apply portfolio normalization if available (for train/inference consistency)
            if self.portfolio_normalizer and self.portfolio_normalizer.is_fitted:
                portfolio_state = self.portfolio_normalizer.normalize_state(portfolio_state)
                logger.debug("✅ Portfolio normalization applied for inference consistency")
            else:
                logger.debug("⚠️ No portfolio normalization - using raw features (may cause train/inference mismatch)")
            
            # Convert to tensor
            portfolio_state = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device)
            
            # Convert stock data to tensor
            stock_data_state = torch.tensor(stock_data.astype(np.float32), dtype=torch.float32, device=self.device)
            
            # Repeat portfolio state for each timestep and concatenate
            portfolio_state_repeated = portfolio_state.unsqueeze(0).repeat(WINDOW_SIZE, 1)
            
            # Concatenate stock data and portfolio features
            combined_state = torch.cat([stock_data_state, portfolio_state_repeated], dim=1)
            
            # Increment step count
            self.step_count += 1
            
            logger.info(f"✅ Enhanced state tensor created:")
            logger.info(f"  Shape: {combined_state.shape} (expected: [{WINDOW_SIZE}, {expected_stock_features + 30}])")
            logger.info(f"  Stock features: {expected_stock_features}")
            logger.info(f"  Portfolio features: 30 (enhanced for DQN v7)")
            logger.info(f"  Valid actions: {valid_actions}")
            
            return combined_state
            
        except Exception as e:
            logger.error(f"Error preparing enhanced state: {e}")
            return None
    
    def _prepare_standard_state(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare standard state tensor with 13 portfolio features for DQN v5 compatibility"""
        try:
            logger.info("Preparing standard state for DQN v5 model inference...")
            
            from src.config.config import STOCK_FEATURES_V2
            
            # Select features based on model type
            available_features = [col for col in STOCK_FEATURES_V2 if col in df.columns]
            logger.info(f"Available stock features: {len(available_features)}/{len(STOCK_FEATURES_V2)}")
            
            if len(available_features) < len(STOCK_FEATURES_V2) * 0.8:
                logger.warning(f"Missing features: {set(STOCK_FEATURES_V2) - set(available_features)}")
                return None
            
            # Get stock data
            if self.preprocessor and hasattr(self.preprocessor, 'feature_names'):
                features_to_use = self.preprocessor.feature_names
                stock_data = df[features_to_use].values
            else:
                stock_data = df[available_features].values
            
            # Apply preprocessing if available
            if self.preprocessor:
                try:
                    stock_data = self.preprocessor.transform(df[available_features if not hasattr(self.preprocessor, 'feature_names') else features_to_use])
                    # Ensure stock_data is a numpy array (preprocessor might return DataFrame)
                    if hasattr(stock_data, 'values'):
                        stock_data = stock_data.values
                except Exception as e:
                    logger.error(f"Preprocessor transform failed: {e}")
                    stock_data = df[available_features].values
            
            # Calculate portfolio features (13 features for DQN v5)
            portfolio_value = self.balance + (self.position * self.current_price)
            
            # Update unrealized P&L
            if self.position > 0:
                cost_basis = self.position * self.entry_price * (1 + TRANSACTION_FEE_PERCENT)
                market_value = self.position * self.current_price
                self.unrealized_pnl = (market_value - cost_basis) / cost_basis
            else:
                self.unrealized_pnl = 0.0
            
            # Standard portfolio features (13 features)
            portfolio_features = np.array([
                self.balance / self.initial_balance if self.initial_balance > 0 else 0,
                self.position * self.current_price / self.initial_balance if self.initial_balance > 0 else 0,
                portfolio_value / self.initial_balance if self.initial_balance > 0 else 0,
                self.position * self.current_price / portfolio_value if portfolio_value > 0 else 0,
                self.unrealized_pnl,
                float(self.step_count - self.position_entry_step) / 60.0 if self.position_entry_step >= 0 else 0,
                0.5,  # Time of day (placeholder)
                1.0,  # Morning session (placeholder)
                0.0,  # Midday session (placeholder)
                0.0,  # Afternoon session (placeholder)
                1.0 if self.position == 0 else 0.0,  # Can buy
                1.0 if self.position > 0 else 0.0,   # Can sell
                min(self.invalid_actions / max(self.step_count, 1), 1.0)  # Invalid action rate
            ])
            
            # Replace any non-finite values
            portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=1.0, neginf=-1.0)
            self.step_count += 1
            
            # CRITICAL: Apply portfolio normalization if available (for train/inference consistency)
            if self.portfolio_normalizer and self.portfolio_normalizer.is_fitted:
                portfolio_features = self.portfolio_normalizer.normalize_state(portfolio_features)
                logger.debug("✅ Portfolio normalization applied for standard DQN inference consistency")
            else:
                logger.debug("⚠️ No portfolio normalization - using raw features (may cause train/inference mismatch)")
            
            # Combine features
            portfolio_features_repeated = np.tile(portfolio_features, (WINDOW_SIZE, 1))
            combined_features = np.concatenate([stock_data, portfolio_features_repeated], axis=1)
            
            # Convert to tensor
            state = torch.tensor(combined_features, dtype=torch.float32, device=self.device)
            
            logger.info(f"✅ Standard state tensor created: shape {state.shape}")
            return state
            
        except Exception as e:
            logger.error(f"Error preparing standard state: {e}")
            return None
    
    def _get_action_with_masking(self, state: torch.Tensor) -> int:
        """Get action from the model with action masking for DQN v7"""
        logger.info("Model evaluating state with action masking...")
        
        # Log state statistics for debugging
        logger.info(f"State shape: {state.shape}")
        logger.info(f"State device: {state.device}")
        logger.info(f"State dtype: {state.dtype}")
        
        with torch.no_grad():
            if self.model_type == "DQN":
                # Add state dimension for batch
                state_batch = state.unsqueeze(0).to(self.device)
                logger.info(f"State batch shape for model: {state_batch.shape}")
                
                # Get Q-values
                q_values = self.agent.q_network(state_batch)[0]  # Shape: [num_actions]
                logger.info(f"Raw Q-values: {q_values.cpu().numpy()}")
                
                # Get valid actions for masking
                valid_actions = self.get_valid_actions()
                logger.info(f"Valid actions: {valid_actions}")
                
                if self.is_enhanced_dqn and len(valid_actions) > 0:
                    # ACTION MASKING: Only consider valid actions
                    action_mask = torch.full_like(q_values, float('-inf'))
                    action_mask[valid_actions] = 0  # Set valid actions to 0 (no penalty)
                    
                    # Apply mask to Q-values
                    masked_q_values = q_values + action_mask
                    
                    # Select best valid action
                    action = torch.argmax(masked_q_values).item()
                    
                    logger.info(f"✅ Action masking applied - selected action {action} from valid: {valid_actions}")
                else:
                    # Standard action selection (for DQN v5 or when no valid actions)
                    action = q_values.max(0)[1].item()
                    logger.info(f"Standard action selection - selected action {action}")
                
                # Log Q-value analysis
                q_np = q_values.cpu().numpy()
                if self.is_enhanced_dqn and len(q_np) >= 7:
                    logger.info(f"Q-value analysis (DQN v7):")
                    for i, action_name in self.ACTION_NAMES.items():
                        logger.info(f"  Q({action_name}): {q_np[i]:.6f}")
                    logger.info(f"  Selected: {self.ACTION_NAMES.get(action, f'Action {action}')} (Q={q_np[action]:.6f})")
                else:
                    logger.info(f"Q-value analysis (Standard DQN):")
                    action_names = ['HOLD', 'BUY', 'SELL']
                    for i, name in enumerate(action_names[:len(q_np)]):
                        logger.info(f"  Q({name}): {q_np[i]:.6f}")
                
            else:
                raise NotImplementedError(f"Model type {self.model_type} not supported")
        
        return action
    
    def _execute_enhanced_action(self, action: int):
        """Execute trading action for DQN v7 (7 actions) or fallback to standard (3 actions)"""
        try:
            # Final safety check - don't execute trades outside market hours
            current_time = datetime.now(timezone.utc)
            if not self._is_market_open(current_time):
                logger.warning(f"⚠️ Trade execution blocked - market is closed")
                logger.info(f"Action {action} would have been executed but market hours restriction prevented it")
                return
            
            # Additional safety check - don't open new positions too close to market close
            is_approaching, minutes_until_close = self._is_approaching_market_close(current_time)
            if action in [1, 2, 3] and is_approaching:  # Any BUY action when approaching close
                logger.warning(f"⚠️ BUY action {action} blocked - {minutes_until_close} minutes until market close")
                logger.info(f"   Preventing new position opening close to market close (safety measure)")
                self.invalid_actions += 1
                return
            
            action_name = self.ACTION_NAMES.get(action, f"UNKNOWN_{action}")
            logger.info(f"Model decision: {action_name} (action={action})")
            
            # Execute based on action type
            if action == 0:  # HOLD
                self._execute_hold()
                
            elif self.is_enhanced_dqn and action in [1, 2, 3]:  # Enhanced BUY actions
                self._execute_enhanced_buy(action)
                
            elif self.is_enhanced_dqn and action in [4, 5, 6]:  # Enhanced SELL actions
                self._execute_enhanced_sell(action)
                
            elif not self.is_enhanced_dqn and action == 1:  # Standard BUY
                self._execute_standard_buy()
                
            elif not self.is_enhanced_dqn and action == 2:  # Standard SELL
                self._execute_standard_sell()
                
            else:
                logger.error(f"❌ Invalid action {action} for model type {self.model_type} (enhanced: {self.is_enhanced_dqn})")
                self.invalid_actions += 1
                return
                
        except Exception as e:
            logger.error(f"❌ Error executing action {action}: {e}")
            self.invalid_actions += 1
    
    def _execute_hold(self):
        """Execute HOLD action"""
        self.consecutive_holds += 1
        self.last_action = 0
        logger.info(f"HOLD - Current position: {self.position} shares, Balance: ${self.balance:.2f}")
        logger.info(f"  Consecutive holds: {self.consecutive_holds}")
    
    def _execute_enhanced_buy(self, action: int):
        """Execute enhanced BUY actions (1=SMALL, 2=MEDIUM, 3=LARGE)"""
        if self.position > 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid {self.ACTION_NAMES[action]} action - Already holding {self.position} shares")
            return
        
        # Get buy ratio based on action
        ratios = {1: getattr(self.config, 'buy_small_ratio', 0.25),
                  2: getattr(self.config, 'buy_medium_ratio', 0.50), 
                  3: getattr(self.config, 'buy_large_ratio', 0.75)}
        
        buy_ratio = ratios[action]
        position_value = self.balance * buy_ratio
        shares_to_buy = int(position_value / self.current_price)
        
        if shares_to_buy <= 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid {self.ACTION_NAMES[action]} action - Insufficient balance")
            return
        
        # Place buy order
        try:
            order_request = MarketOrderRequest(
                symbol=self.ticker,
                qty=shares_to_buy,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Submitting {self.ACTION_NAMES[action]} order to Alpaca for {shares_to_buy} shares...")
            order = self.alpaca_client.submit_order(order_request)
            logger.info(f"Order submitted successfully. Order ID: {order.id}")
            
            # Update local state
            cost = shares_to_buy * self.current_price * (1 + TRANSACTION_FEE_PERCENT)
            self.balance -= cost
            self.position = shares_to_buy
            self.entry_price = self.current_price
            self.position_entry_step = self.step_count
            self.last_action = action
            self.last_position_change_step = self.step_count
            self.last_trade_step = self.step_count
            self.consecutive_holds = 0
            
            # Update action tracking
            self.action_counts[action] += 1
            self.successful_trades[action] += 1
            
            # Save trade
            self._save_trade(TradeType.BUY, cost, self.current_price, shares_to_buy)
            
            logger.info(f"✅ {self.ACTION_NAMES[action]} executed: {shares_to_buy} shares of {self.ticker} at ${self.current_price:.2f}")
            logger.info(f"   Cost: ${cost:.2f} (including fees), New balance: ${self.balance:.2f}")
            logger.info(f"   Buy ratio: {buy_ratio*100:.0f}% of available cash")
            
        except Exception as e:
            logger.error(f"❌ Failed to submit {self.ACTION_NAMES[action]} order: {e}")
            self.invalid_actions += 1
    
    def _execute_enhanced_sell(self, action: int):
        """Execute enhanced SELL actions (4=SMALL, 5=MEDIUM, 6=LARGE)"""
        if self.position <= 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid {self.ACTION_NAMES[action]} action - No position to sell")
            return
        
        # Get sell ratio based on action
        ratios = {4: getattr(self.config, 'sell_small_ratio', 0.25),
                  5: getattr(self.config, 'sell_medium_ratio', 0.50),
                  6: getattr(self.config, 'sell_large_ratio', 1.00)}
        
        sell_ratio = ratios[action]
        shares_to_sell = int(self.position * sell_ratio)
        
        if shares_to_sell <= 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid {self.ACTION_NAMES[action]} action - No shares to sell")
            return
        
        # Place sell order
        try:
            order_request = MarketOrderRequest(
                symbol=self.ticker,
                qty=shares_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Submitting {self.ACTION_NAMES[action]} order to Alpaca for {shares_to_sell} shares...")
            order = self.alpaca_client.submit_order(order_request)
            logger.info(f"Order submitted successfully. Order ID: {order.id}")
            
            # Calculate profit/loss for the shares being sold
            revenue = shares_to_sell * self.current_price * (1 - TRANSACTION_FEE_PERCENT)
            cost_basis = shares_to_sell * self.entry_price * (1 + TRANSACTION_FEE_PERCENT)
            profit = revenue - cost_basis
            profit_percent = (profit / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Update statistics (count all sells as trades)
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
            else:
                self.losing_trades += 1
                self.total_loss += abs(profit)
            
            # Update local state
            self.balance += revenue
            self.position -= shares_to_sell
            self.last_action = action
            self.last_position_change_step = self.step_count
            self.last_trade_step = self.step_count
            self.consecutive_holds = 0
            
            # Reset position tracking if fully sold
            if self.position <= 0:
                self.position_entry_step = -1
                self.position = 0  # Ensure exact zero
            
            # Update action tracking
            self.action_counts[action] += 1
            self.successful_trades[action] += 1
            
            # Save trade
            self._save_trade(TradeType.SELL, revenue, self.current_price, shares_to_sell)
            
            logger.info(f"✅ {self.ACTION_NAMES[action]} executed: {shares_to_sell} shares of {self.ticker} at ${self.current_price:.2f}")
            logger.info(f"   Entry: ${self.entry_price:.2f}, Exit: ${self.current_price:.2f}")
            logger.info(f"   Profit: ${profit:.2f} ({profit_percent:.2f}%), New balance: ${self.balance:.2f}")
            logger.info(f"   Remaining position: {self.position} shares")
            logger.info(f"   Sell ratio: {sell_ratio*100:.0f}% of position")
            
            # Calculate and log win rate
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            logger.info(f"   Stats: {self.winning_trades}W/{self.losing_trades}L (Win rate: {win_rate:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Failed to submit {self.ACTION_NAMES[action]} order: {e}")
            self.invalid_actions += 1
    
    def _execute_standard_buy(self):
        """Execute standard BUY action for DQN v5 compatibility"""
        if self.position > 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid BUY action - Already holding {self.position} shares")
            return
        
        # Use max_position_size for standard buy
        position_value = self.balance * self.max_position_size
        shares_to_buy = int(position_value / self.current_price)
        
        if shares_to_buy <= 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid BUY action - Insufficient balance")
            return
        
        # Place standard buy order
        try:
            order_request = MarketOrderRequest(
                symbol=self.ticker,
                qty=shares_to_buy,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Submitting standard BUY order to Alpaca for {shares_to_buy} shares...")
            order = self.alpaca_client.submit_order(order_request)
            logger.info(f"Order submitted successfully. Order ID: {order.id}")
            
            # Update local state
            cost = shares_to_buy * self.current_price * (1 + TRANSACTION_FEE_PERCENT)
            self.balance -= cost
            self.position = shares_to_buy
            self.entry_price = self.current_price
            self.position_entry_step = self.step_count
            self.last_action = 1
            self.consecutive_holds = 0
            
            # Save trade
            self._save_trade(TradeType.BUY, cost, self.current_price, shares_to_buy)
            
            logger.info(f"✅ Standard BUY executed: {shares_to_buy} shares of {self.ticker} at ${self.current_price:.2f}")
            logger.info(f"   Cost: ${cost:.2f} (including fees), New balance: ${self.balance:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to submit standard BUY order: {e}")
            self.invalid_actions += 1
    
    def _execute_standard_sell(self):
        """Execute standard SELL action for DQN v5 compatibility"""
        if self.position <= 0:
            self.invalid_actions += 1
            logger.warning(f"Invalid SELL action - No position to sell")
            return
        
        # Sell entire position
        shares_to_sell = self.position
        
        # Place standard sell order
        try:
            order_request = MarketOrderRequest(
                symbol=self.ticker,
                qty=shares_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Submitting standard SELL order to Alpaca for {shares_to_sell} shares...")
            order = self.alpaca_client.submit_order(order_request)
            logger.info(f"Order submitted successfully. Order ID: {order.id}")
            
            # Calculate profit/loss
            revenue = shares_to_sell * self.current_price * (1 - TRANSACTION_FEE_PERCENT)
            cost_basis = shares_to_sell * self.entry_price * (1 + TRANSACTION_FEE_PERCENT)
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
            self.position = 0
            self.position_entry_step = -1
            self.last_action = 2
            self.consecutive_holds = 0
            
            # Save trade
            self._save_trade(TradeType.SELL, revenue, self.current_price, shares_to_sell)
            
            logger.info(f"✅ Standard SELL executed: {shares_to_sell} shares of {self.ticker} at ${self.current_price:.2f}")
            logger.info(f"   Entry: ${self.entry_price:.2f}, Exit: ${self.current_price:.2f}")
            logger.info(f"   Profit: ${profit:.2f} ({profit_percent:.2f}%), New balance: ${self.balance:.2f}")
            
            # Calculate and log win rate
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            logger.info(f"   Stats: {self.winning_trades}W/{self.losing_trades}L (Win rate: {win_rate:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Failed to submit standard SELL order: {e}")
            self.invalid_actions += 1

# Create an alias for backward compatibility
PaperTradingBot = EnhancedPaperTradingBot 