from typing import List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import Integer, String, DateTime, Numeric, ForeignKey, func, Index, UniqueConstraint, Enum as SQLEnum
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from src.web.extensions import app, db


def add_to_dict_method(cls):
    def to_dict(self, include_relationships: bool = False) -> Dict[str, Any]:
        data = {}
        mapper = inspect(self.__class__)
        for column in mapper.columns:
            value = getattr(self, column.key)
            if isinstance(value, datetime):
                data[column.key] = value.isoformat()
            elif isinstance(value, Enum):
                data[column.key] = value.value
            else:
                data[column.key] = value

        if include_relationships:
            for rel in mapper.relationships:
                related_obj = getattr(self, rel.key)
                if related_obj is not None:
                    if rel.uselist:
                        data[rel.key] = [obj.id for obj in related_obj if hasattr(obj, 'id')]
                    else:
                        data[rel.key] = getattr(related_obj, 'id', str(related_obj))
                else:
                    data[rel.key] = None
        return data

    cls.to_dict = to_dict
    return cls

@add_to_dict_method
class Profile(UserMixin, db.Model):
    __tablename__ = 'profiles'
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    _password: Mapped[str] = mapped_column("password", String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    portfolios: Mapped[List["Portfolio"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def __repr__(self):
        return f'<Profile {self.username}>'
    
    @property
    def password(self):
        raise AttributeError("Password is write-only.")

    @password.setter
    def password(self, plaintext_password):
        self._password = generate_password_hash(plaintext_password)

    def check_password(self, plaintext_password):
        return check_password_hash(self._password, plaintext_password)

@add_to_dict_method
class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    current_balance: Mapped[float] = mapped_column(default=0.0)
    initial_balance: Mapped[float] = mapped_column(default=0.0)
    is_live_trading: Mapped[bool] = mapped_column(default=False)

    trade_history_entries: Mapped[List["TradeHistory"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"), nullable=False)
    owner: Mapped["Profile"] = relationship(back_populates="portfolios")

    def __repr__(self):
        return f'<Portfolio {self.name} (Owner ID: {self.profile_id})>'

@add_to_dict_method
class TradingSession(db.Model):
    __tablename__ = 'trading_sessions'
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    model: Mapped["MarklygonModel"] = relationship(back_populates="trading_sessions")

    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    initial_balance: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    final_balance: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    net_profit: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    return_rate: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    max_drawdown: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    sharpe_ratio: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    calmar_ratio: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    invalid_actions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    trades: Mapped[List["TradeHistory"]] = relationship(
        back_populates="trading_session",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def __repr__(self):
        return (
            f"<TradingSession {self.id} | Model: {self.model_id} | "
            f"Start: {self.start_time} | End: {self.end_time}>"
        )


class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeHistory(db.Model):
    __tablename__ = 'trade_history'
    id: Mapped[int] = mapped_column()
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    trade_type: Mapped[TradeType] = mapped_column(SQLEnum(TradeType), nullable=False, default=TradeType.HOLD)
    amount: Mapped[float] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    shares: Mapped[float] = mapped_column(Numeric(precision=18, scale=4), nullable=False)

    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    portfolio: Mapped["Portfolio"] = relationship(back_populates="trade_history_entries")

    trading_session_id: Mapped[int] = mapped_column(ForeignKey("trading_sessions.id"), nullable=False)
    trading_session: Mapped["TradingSession"] = relationship(back_populates="trades")

    __table_args__ = (
        Index('idx_trade_history_timestamp', "timestamp"),
        db.PrimaryKeyConstraint('id', 'timestamp', name='pk_trade_history_id_timestamp'),
    )

    def __repr__(self):
        return (
            f'<Trade {self.trade_type} at {self.timestamp} '
            f'(Portfolio ID: {self.portfolio_id}, Session ID: {self.trading_session_id})>'
        )

class ModelType(Enum):
    DQN = "DQN"
    SAC = "SAC"
 
@add_to_dict_method    
class MarklygonModel(db.Model):
    __tablename__ = 'models'
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[ModelType] = mapped_column(SQLEnum(ModelType), nullable=False)
    ticker: Mapped[str] = mapped_column(String(5), nullable=False)
    model_path: Mapped[str] = mapped_column(String, nullable=True)
    
    backtests: Mapped[List["BacktestHistory"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    trading_sessions: Mapped[List["TradingSession"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
@add_to_dict_method
class BacktestHistory(db.Model):
    __tablename__ = 'backtests'
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    backtest_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_balance: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    final_balance: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    net_profit: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    return_rate: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    max_drawdown: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    sharpe_ratio: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=0)
    invalid_actions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    preprocessor_path: Mapped[str] = mapped_column(String, nullable=True)
    model: Mapped["MarklygonModel"] = relationship(back_populates="backtests")