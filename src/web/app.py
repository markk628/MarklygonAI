from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from sqlalchemy import desc
import json
import random
import os

from src.web.models import *
from src.config.config import DATABASE_WEB
from src.utils.database import DatabaseManager
from src.web.extensions import *

# Initialize the database
# db.init_app(app)

# Initialize extensions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return Profile.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        profile = Profile.query.filter_by(username=username).first()
        
        if profile and check_password_hash(profile._password, password):
            login_user(profile)
            flash('로그인에 성공했습니다!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호가 잘못되었습니다.', 'error')
    
    return render_template('signin.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if Profile.query.filter_by(username=username).first():
            flash('이미 사용 중인 아이디입니다.', 'error')
            return render_template('signup.html')
        
        if Profile.query.filter_by(email=email).first():
            flash('이미 등록된 이메일입니다.', 'error')
            return render_template('signup.html')
        
        # Create new user using the property setter for password hashing
        profile = Profile(
            email=email,
            username=username,
        )
        profile.password = password  
        
        db.session.add(profile)
        db.session.commit()
        
        login_user(profile)
        flash('회원가입이 완료되었습니다!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch real backtest data from database
    latest_backtests = BacktestHistory.query.order_by(desc(BacktestHistory.backtest_date)).limit(10).all()
    
    # Calculate portfolio summary from real data
    if latest_backtests:
        total_value = sum(bt.final_balance for bt in latest_backtests) / len(latest_backtests)
        avg_return = sum(bt.return_rate for bt in latest_backtests) / len(latest_backtests)
        total_trades = sum(bt.total_trades for bt in latest_backtests)
        winning_trades = sum(bt.winning_trades for bt in latest_backtests)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        active_models = len(set(bt.model_id for bt in latest_backtests))
    else:
        # Default values if no data
        total_value = 100000
        avg_return = 0
        win_rate = 0
        active_models = 0
    
    portfolio_data = {
        'portfolio_summary': {
            'total_value': total_value,
            'daily_change': random.uniform(-2.5, 3.5),  # This could be calculated from real data
            'daily_change_amount': random.uniform(-500, 800),
            'total_return': avg_return,
            'active_positions': active_models,
            'win_rate': win_rate,
            'new_positions_today': random.randint(0, 5)
        },
        'model_performance': []
    }
    
    # Get unique models with their latest backtest results
    models = MarklygonModel.query.all()
    for model in models:
        latest_backtest = BacktestHistory.query.filter_by(model_id=model.id).order_by(desc(BacktestHistory.backtest_date)).first()
        if latest_backtest:
            portfolio_data['model_performance'].append({
                'model_id': f'model_{model.id}',
                'model_name': f'{model.model.value} - {model.ticker}',
                'model_type': model.model.value,
                'status': 'completed',
                'current_value': float(latest_backtest.final_balance),
                'allocation_percentage': random.uniform(15, 25),  # Could be calculated based on total portfolio
                'total_return': float(latest_backtest.return_rate),
                'daily_change': random.uniform(-2, 4)  # Could be calculated from recent backtests
            })
    
    return render_template('dashboard.html', portfolio=portfolio_data)

@app.route('/portfolio')
@login_required
def portfolio():
    # 포트폴리오 페이지는 대시보드로 통합됨
    return redirect(url_for('dashboard'))

@app.route('/backtest')
@login_required
def backtest():
    return render_template('backtest.html')

@app.route('/mypage')
@login_required
def mypage():
    return render_template('mypage.html')

@app.route('/trading')
@login_required
def trading():
    return render_template('trading.html')

# Global dictionary to track active trading bots
active_trading_bots = {}

# API Routes
@app.route('/api/models')
def api_models():
    models = MarklygonModel.query.all()
    models_data = {
        'models': [
            {
                'id': f'model_{model.id}',
                'model_name': f'{model.model.value} - {model.ticker}',
                'model_type': model.model.value,
                'ticker': model.ticker,
                'description': f'{model.model.value} model trained on {model.ticker} stock data',
                'training_status': 'completed',
                'backtest_count': len(model.backtests)
            } for model in models
        ]
    }
    return jsonify(models_data)

@app.route('/api/tickers')
def api_tickers():
    # Get unique tickers from models
    tickers = db.session.query(MarklygonModel.ticker).distinct().all()
    
    # For now, we'll return basic ticker info
    # In production, you might want to fetch real-time prices from an API
    tickers_data = {
        'tickers': [
            {
                'symbol': ticker[0],
                'company_name': get_company_name(ticker[0]),  # Helper function to get company names
                'current_price': random.uniform(50, 1000),  # Replace with real price data
                'change_percent': random.uniform(-5, 5)
            } for ticker in tickers
        ]
    }
    return jsonify(tickers_data)

def get_company_name(ticker):
    """Helper function to get company names from ticker symbols"""
    ticker_names = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'BAC': 'Bank of America Corp.',
        'GS': 'Goldman Sachs Group Inc.'
    }
    return ticker_names.get(ticker, ticker + ' Inc.')

@app.route('/api/portfolio')
@login_required
def api_portfolio():
    # Fetch real backtest data
    latest_backtests = BacktestHistory.query.order_by(desc(BacktestHistory.backtest_date)).limit(10).all()
    
    # Calculate portfolio metrics
    if latest_backtests:
        total_value = sum(bt.final_balance for bt in latest_backtests) / len(latest_backtests)
        avg_return = sum(bt.return_rate for bt in latest_backtests) / len(latest_backtests)
        total_trades = sum(bt.total_trades for bt in latest_backtests)
        winning_trades = sum(bt.winning_trades for bt in latest_backtests)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        active_models = len(set(bt.model_id for bt in latest_backtests))
    else:
        total_value = 100000
        avg_return = 0
        win_rate = 0
        active_models = 0
    
    portfolio_data = {
        'portfolio_summary': {
            'total_value': float(total_value),
            'daily_change': random.uniform(-2.5, 3.5),
            'daily_change_amount': random.uniform(-500, 800),
            'total_return': float(avg_return),
            'active_positions': active_models,
            'win_rate': float(win_rate)
        },
        'model_performance': []
    }
    
    # Get model performance data
    models = MarklygonModel.query.all()
    for model in models:
        latest_backtest = BacktestHistory.query.filter_by(model_id=model.id).order_by(desc(BacktestHistory.backtest_date)).first()
        if latest_backtest:
            portfolio_data['model_performance'].append({
                'model_id': f'model_{model.id}',
                'model_name': f'{model.model.value} - {model.ticker}',
                'model_type': model.model.value,
                'status': 'completed',
                'current_value': float(latest_backtest.final_balance),
                'allocation_percentage': random.uniform(15, 25),
                'total_return': float(latest_backtest.return_rate),
                'daily_change': random.uniform(-2, 4)
            })
    
    return jsonify(portfolio_data)

@app.route('/api/backtest-results')
@login_required
def api_backtest_results():
    # Fetch all backtest results with model information
    backtests = db.session.query(BacktestHistory, MarklygonModel)\
        .join(MarklygonModel, BacktestHistory.model_id == MarklygonModel.id)\
        .order_by(desc(BacktestHistory.backtest_date))\
        .all()
    
    backtest_data = {
        'backtest_results': []
    }
    
    for backtest, model in backtests:
        result = {
            'backtest_id': backtest.id,
            'model_id': f'model_{model.id}',
            'model_name': f'{model.model.value} - {model.ticker}',
            'model_type': model.model.value,
            'ticker': model.ticker,
            'company_name': get_company_name(model.ticker),
            'backtest_date': backtest.backtest_date.isoformat(),
            'start_date': backtest.start_date.isoformat(),
            'end_date': backtest.end_date.isoformat(),
            'initial_balance': float(backtest.initial_balance),
            'final_balance': float(backtest.final_balance),
            'net_profit': float(backtest.net_profit),
            'return_rate': float(backtest.return_rate),
            'max_drawdown': float(backtest.max_drawdown),
            'sharpe_ratio': float(backtest.sharpe_ratio),
            'total_trades': backtest.total_trades,
            'winning_trades': backtest.winning_trades,
            'losing_trades': backtest.losing_trades,
            'win_rate': (backtest.winning_trades / backtest.total_trades * 100) if backtest.total_trades > 0 else 0,
            'invalid_actions': backtest.invalid_actions
        }
        backtest_data['backtest_results'].append(result)
    
    return jsonify(backtest_data)

@app.route('/api/realtime/portfolio')
@login_required
def api_realtime_portfolio():
    # For real-time data, you would fetch from your trading system
    # For now, we'll use the latest backtest data
    latest_backtest = BacktestHistory.query.order_by(desc(BacktestHistory.backtest_date)).first()
    
    if latest_backtest:
        total_value = float(latest_backtest.final_balance)
        daily_change = random.uniform(-2.5, 3.5)  # This would be calculated from real data
    else:
        total_value = 100000
        daily_change = 0
    
    realtime_data = {
        'total_value': total_value,
        'daily_change': daily_change,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(realtime_data)

@app.route('/api/user/profile', methods=['GET', 'POST'])
@login_required
def api_user_profile():
    if request.method == 'GET':
        profile_data = {
            'full_name': current_user.username,
            'email': current_user.email,
            'trading_level': 'intermediate',
            'risk_tolerance': 'moderate',
            'preferred_models': [f'model_{m.id}' for m in MarklygonModel.query.limit(3).all()],
            'notification_settings': {
                'email': True,
                'push': True,
                'trading': True,
                'reports': True
            }
        }
        return jsonify(profile_data)
    
    elif request.method == 'POST':
        return jsonify({'status': 'success', 'message': 'Profile updated successfully'})

@app.route('/api/models-with-backtests')
@login_required
def api_models_with_backtests():
    """Get models with their backtest results for trading selection"""
    models = MarklygonModel.query.all()
    models_data = {'models': []}
    
    for model in models:
        # Get latest backtest for this model
        latest_backtest = BacktestHistory.query.filter_by(model_id=model.id)\
            .order_by(desc(BacktestHistory.backtest_date)).first()
        
        if latest_backtest:
            model_data = {
                'id': model.id,
                'model_name': f'{model.model.value} - {model.ticker}',
                'model_type': model.model.value,
                'ticker': model.ticker,
                'return_rate': float(latest_backtest.return_rate),
                'sharpe_ratio': float(latest_backtest.sharpe_ratio),
                'max_drawdown': float(latest_backtest.max_drawdown),
                'win_rate': (latest_backtest.winning_trades / latest_backtest.total_trades * 100) 
                           if latest_backtest.total_trades > 0 else 0,
                'total_trades': latest_backtest.total_trades,
                'backtest_date': latest_backtest.backtest_date.isoformat()
            }
            models_data['models'].append(model_data)
    
    return jsonify(models_data)

@app.route('/api/start-trading', methods=['POST'])
@login_required
def api_start_trading():
    """Start paper trading with a selected model"""
    import threading
    import traceback
    
    try:
        # Import at module level to ensure all dependencies are loaded
        from src.web.alpaca.paper_trading_bot import PaperTradingBot
        
        data = request.json
        model_id = data.get('model_id')
        initial_balance = data.get('initial_balance', 10000)
        max_position_size = data.get('max_position_size', 0.7)
        
        # Check if model exists
        model = MarklygonModel.query.get(model_id)
        if not model:
            return jsonify({'success': False, 'error': 'Model not found'})
        
        # Check if already trading
        if model_id in active_trading_bots:
            return jsonify({'success': False, 'error': 'Trading already active for this model'})
        
        # Create and start trading bot
        bot = PaperTradingBot(model_id, initial_balance, max_position_size)
        
        # Run bot in a separate thread
        thread = threading.Thread(target=bot.start, daemon=True)
        thread.start()
        
        # Store bot reference
        active_trading_bots[model_id] = {
            'bot': bot,
            'thread': thread,
            'start_time': datetime.now(timezone.utc)
        }
        
        return jsonify({
            'success': True, 
            'message': f'Started trading with model {model_id}',
            'session_id': bot.session_id
        })
        
    except ImportError as e:
        error_msg = f"Import error: {str(e)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/stop-trading/<int:model_id>', methods=['POST'])
@login_required
def api_stop_trading(model_id):
    """Stop paper trading for a model"""
    try:
        if model_id not in active_trading_bots:
            return jsonify({'success': False, 'error': 'No active trading for this model'})
        
        # Stop the bot
        bot_info = active_trading_bots[model_id]
        bot_info['bot'].stop()
        
        # Remove from active bots
        del active_trading_bots[model_id]
        
        return jsonify({'success': True, 'message': f'Stopped trading for model {model_id}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/active-trading-sessions')
@login_required
def api_active_trading_sessions():
    """Get all active trading sessions"""
    sessions_data = {'sessions': []}
    
    for model_id, bot_info in active_trading_bots.items():
        bot = bot_info['bot']
        session_id = bot.session_id
        
        # Get session from database
        session = TradingSession.query.get(session_id)
        if session:
            model = MarklygonModel.query.get(model_id)
            
            # Calculate current stats
            current_balance = bot.balance + (bot.position * bot.current_price if hasattr(bot, 'current_price') else 0)
            win_rate = (bot.winning_trades / bot.total_trades * 100) if bot.total_trades > 0 else 0
            
            session_data = {
                'session_id': session_id,
                'model_id': model_id,
                'model_name': f'{model.model.value} - {model.ticker}',
                'ticker': model.ticker,
                'start_time': session.start_time.isoformat(),
                'initial_balance': float(session.initial_balance),
                'current_balance': float(current_balance),
                'position': bot.position,
                'total_trades': bot.total_trades,
                'win_rate': win_rate,
                'status': 'active'
            }
            sessions_data['sessions'].append(session_data)
    
    return jsonify(sessions_data)

@app.route('/api/test-alpaca')
def api_test_alpaca():
    """Simple test endpoint for Alpaca connection"""
    from alpaca.trading.client import TradingClient
    from src.config.apikeys import ALPACA_APIKEY, ALPACA_SECRET_KEY
    
    try:
        alpaca = TradingClient(ALPACA_APIKEY, ALPACA_SECRET_KEY, paper=True)
        account = alpaca.get_account()
        
        return jsonify({
            'success': True,
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'cash': float(account.cash)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/alpaca/portfolio')
@login_required
def api_alpaca_portfolio():
    """Get portfolio data from Alpaca"""
    from alpaca.trading.client import TradingClient
    from src.config.apikeys import ALPACA_APIKEY, ALPACA_SECRET_KEY
    
    try:
        print("Fetching Alpaca portfolio data...")
        
        # Initialize Alpaca client
        alpaca = TradingClient(ALPACA_APIKEY, ALPACA_SECRET_KEY, paper=True)
        
        # Get account info
        account = alpaca.get_account()
        print(f"Account fetched - Portfolio Value: ${account.portfolio_value}")
        
        # Get positions
        positions = alpaca.get_all_positions()
        print(f"Found {len(positions)} positions")
        
        # Get recent orders
        try:
            # Get all orders without any filters
            orders = alpaca.get_orders()
            # Take only the most recent 10 orders
            orders = orders[:10] if len(orders) > 10 else orders
            print(f"Found {len(orders)} recent orders")
        except Exception as e:
            print(f"Error fetching orders: {e}")
            orders = []  # Fall back to empty list if orders fail
        
        portfolio_data = {
            'account': {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': int(account.daytrade_count),
                'pattern_day_trader': account.pattern_day_trader
            },
            'positions': [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price),
                'lastday_price': float(pos.lastday_price),
                'change_today': float(pos.change_today)
            } for pos in positions],
            'recent_orders': [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty) if order.qty else 0,
                'side': order.side.value,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force.value,
                'status': order.status.value,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0
            } for order in orders]
        }
        
        print("Successfully created portfolio data response")
        return jsonify(portfolio_data)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error fetching Alpaca portfolio: {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg})

if __name__ == '__main__':
    print("MarklygonAI Flask Application Starting...")

    with app.app_context():
        db.create_all()
        print("Database initialized")

        # Create default user if not exists
        if not Profile.query.filter_by(username='demo').first():
            demo_user = Profile(
                username='demo',
                email='demo@marklygon.ai',
            )
            demo_user.password = 'demo123'  # Triggers password hash via property setter
            db.session.add(demo_user)
            db.session.commit()
            print("Demo user created: demo/demo123")

        # Display available models and backtests
        models_count = MarklygonModel.query.count()
        backtests_count = BacktestHistory.query.count()
        print(f"Found {models_count} models and {backtests_count} backtest results in database")

    print("Application running at: http://localhost:5000")
    print("Default login: demo/demo123 or create a new account at /register")
    app.run(debug=True, host='0.0.0.0', port=5000)
