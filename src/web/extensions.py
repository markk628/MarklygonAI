import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from src.config.config import PRODUCT_SECRET_KEY, DEV_SECRET_KEY, WEB_DATABASE_URI
# from src.web.marklygon_web.models import Profile, Portfolio, TradingSession, TradeHistory, MarklygonModel, BacktestHistory

def create_app():
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
    
    if os.environ.get("FLASK_ENV") == "production":
        app.config['SECRET_KEY'] = PRODUCT_SECRET_KEY
    else:
        app.config['SECRET_KEY'] = DEV_SECRET_KEY

    app.config['SQLALCHEMY_DATABASE_URI'] = WEB_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    return app
def initialize_db(app):
    return SQLAlchemy(app)
    
app = create_app()
db = initialize_db(app)