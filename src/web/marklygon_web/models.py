from typing import List, Dict, Any
from datetime import datetime
from enum import Enum
from sqlalchemy import String, DateTime, Numeric, ForeignKey, func, Index, UniqueConstraint, Enum as SQLEnum
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Mapped, mapped_column, relationship
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

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
class Profile(db.Model):
    __tablename__ = 'profiles'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    first_name = mapped_column(String(30), nullable=False)
    last_name = mapped_column(String(30), nullable=False)
    _password = mapped_column("password", String(255), nullable=False)

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
    user_id: Mapped[int] = mapped_column(nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    trades: Mapped[List["TradeHistory"]] = relationship(
        back_populates="trading_session",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def __repr__(self):
        return f"<TradingSession {self.id} | User: {self.user_id} | Start: {self.start_time} | End: {self.end_time}>"

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeHistory(db.Model):
    __tablename__ = 'trade_history'
    id: Mapped[int] = mapped_column()
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    trade_type: Mapped[TradeType] = mapped_column(SQLEnum(TradeType), nullable=False, default=TradeType.HOLD)
    amount: Mapped[Numeric] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    price: Mapped[Numeric] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    shares: Mapped[Numeric] = mapped_column(Numeric(precision=18, scale=4), nullable=False)

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
