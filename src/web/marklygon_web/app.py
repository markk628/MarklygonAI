from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from src.web.marklygon_web.models import db, Profile, Portfolio, TradeHistory, TradingSession, TradeType
from datetime import datetime

from src.config.config import WEB_DATABASE_URI, DATABASE_WEB
from src.utils.database import DatabaseManager

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = WEB_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Both username and password are required.', 'error')
            return render_template('login.html')

        user = Profile.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.first_name}!', 'success')
            return redirect(url_for('portfolio'))
        else:
            flash('Invalid credentials. Please try again.', 'error')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Sign up page"""
    if request.method == 'POST':
        username = request.form.get('username')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        password = request.form.get('password')

        if not all([username, first_name, last_name, password]):
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        existing_user = Profile.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('signup.html')

        try:
            new_user = Profile(
                username=username,
                first_name=first_name,
                last_name=last_name,
                password=password  # Triggers hashing via setter
            )
            db.session.add(new_user)
            db.session.commit()

            # default_portfolio = Portfolio(
            #     name=f"{first_name}'s Portfolio",
            #     current_balance=10000.0,
            #     initial_balance=10000.0,
            #     is_live_trading=False,
            #     profile_id=new_user.id
            # )
            # db.session.add(default_portfolio)
            # db.session.commit()

            flash('Account created successfully! You can now log in.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            flash('An error occurred while creating your account. Please try again.', 'error')
            app.logger.error(f"Signup error: {e}")

    return render_template('signup.html')


@app.route('/portfolio')
@login_required
def portfolio():
    """Portfolio page - shows user's portfolios and trading history"""
    user_id = session.get('user_id')
    user = session.get(Profile, user_id)
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    
    # Get user's portfolios
    portfolios = Portfolio.query.filter_by(profile_id=user_id).all()
    
    # Get recent trades across all portfolios
    recent_trades = []
    for portfolio in portfolios:
        trades = TradeHistory.query.filter_by(portfolio_id=portfolio.id)\
                                 .order_by(TradeHistory.timestamp.desc())\
                                 .limit(10).all()
        recent_trades.extend(trades)
    
    # Sort all trades by timestamp
    recent_trades.sort(key=lambda x: x.timestamp, reverse=True)
    recent_trades = recent_trades[:20]  # Show last 20 trades
    
    return render_template('portfolio.html', 
                         user=user, 
                         portfolios=portfolios, 
                         recent_trades=recent_trades)

@app.route('/create_portfolio', methods=['POST'])
@login_required
def create_portfolio():
    """Create a new portfolio"""
    user_id = session.get('user_id')
    portfolio_name = request.form.get('portfolio_name')
    initial_balance = float(request.form.get('initial_balance', 0))
    
    if not portfolio_name:
        flash('Portfolio name is required.', 'error')
        return redirect(url_for('portfolio'))
    
    try:
        new_portfolio = Portfolio(
            name=portfolio_name,
            current_balance=initial_balance,
            initial_balance=initial_balance,
            is_live_trading=False,
            profile_id=user_id
        )
        db.session.add(new_portfolio)
        db.session.commit()
        flash(f'Portfolio "{portfolio_name}" created successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error creating portfolio. Please try again.', 'error')
        app.logger.error(f"Portfolio creation error: {e}")
    
    return redirect(url_for('portfolio'))

@app.route('/api/portfolio/<int:portfolio_id>/trades')
@login_required
def get_portfolio_trades(portfolio_id):
    """API endpoint to get trades for a specific portfolio"""
    user_id = session.get('user_id')
    
    # Verify the portfolio belongs to the current user
    portfolio = Portfolio.query.filter_by(id=portfolio_id, profile_id=user_id).first()
    if not portfolio:
        return jsonify({'error': 'Portfolio not found'}), 404
    
    trades = TradeHistory.query.filter_by(portfolio_id=portfolio_id)\
                              .order_by(TradeHistory.timestamp.desc())\
                              .limit(50).all()
    
    trades_data = [trade.to_dict() for trade in trades]
    return jsonify(trades_data)

@app.route('/logout')
def logout():
    """Logout - clear session"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500


if __name__ == '__main__':
    db_manager = DatabaseManager(database=DATABASE_WEB)
    with app.app_context():
        # Create tables
        db.create_all()
        db_manager.execute("SELECT create_hypertable('trade_history', 'timestamp', if_not_exists => TRUE);")
        
        # Create a sample user if none exists (for testing)
        if not Profile.query.first():
            sample_user = Profile(
                username='demo_user',
                first_name='Demo',
                last_name='User',
                password='password123'
            )
            db.session.add(sample_user)
            db.session.commit()
            
            # Create sample portfolio
            sample_portfolio = Portfolio(
                name='Demo Portfolio',
                current_balance=15000.0,
                initial_balance=10000.0,
                is_live_trading=False,
                profile_id=sample_user.id
            )
            db.session.add(sample_portfolio)
            db.session.commit()
            
            print("Created demo user: username='demo_user'")
    
    app.run(debug=True)