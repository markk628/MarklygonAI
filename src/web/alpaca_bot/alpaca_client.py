import asyncio
import logging
from typing import List, Dict, Any, Optional
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.entity import Position
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AlpacaClient:
    
    
    def __init__(self, api_key: str, secret_key: str, base_url: str, data_feed: str = 'iex'):
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_feed = data_feed
        self.paper = 'paper' in base_url.lower()
        
        self.rest = REST(api_key, secret_key, base_url)
        
        self.stream = Stream(
            api_key, 
            secret_key, 
            base_url, 
            data_feed=data_feed,
            raw_data=True
        )
        
        self._handlers_registered = False

    async def start_websocket(self, symbols: List[str] = None):
        
        if symbols is None:
            symbols = ["SPY"]  # 기본값으로 S&P 500 ETF
        
        print(f"웹소켓 연결 시작 - 데이터 피드: {self.data_feed}")
        
        if not self._handlers_registered:
            print("이벤트 핸들러 등록 중...")
            
            async def on_trade(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    price = data.price if hasattr(data, 'price') else 0.0
                    size = data.size if hasattr(data, 'size') else 0
                    timestamp = data.timestamp if hasattr(data, 'timestamp') else datetime.now()
                    print(f"[거래] {symbol}: 가격=${price:.2f}, 수량={size}, 시간={timestamp}")
                except Exception as e:
                    logger.error(f"거래 데이터 처리 오류: {e}")
            
            async def on_quote(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    bid = data.bid_price if hasattr(data, 'bid_price') else 0.0
                    ask = data.ask_price if hasattr(data, 'ask_price') else 0.0
                    print(f"[호가] {symbol}: 매수=${bid:.2f}, 매도=${ask:.2f}, 스프레드=${ask-bid:.2f}")
                except Exception as e:
                    logger.error(f"호가 데이터 처리 오류: {e}")
            
            async def on_bar(data):
                try:
                    symbol = data.symbol if hasattr(data, 'symbol') else "Unknown"
                    close = data.close if hasattr(data, 'close') else 0.0
                    volume = data.volume if hasattr(data, 'volume') else 0
                    print(f"[바] {symbol}: 종가=${close:.2f}, 거래량={volume}")
                except Exception as e:
                    logger.error(f"바 데이터 처리 오류: {e}")
            
            @self.stream.on_status("*")
            async def on_status(status):
                print(f"[상태] 웹소켓 상태 변경: {status}")
                if status.get('status') == 'connected' or status.get('status') == 'auth_success':
                    print(f"[연결 성공] 웹소켓 연결이 설정되었습니다: {status}")
            
            print(f"심볼 구독 등록 중: {', '.join(symbols)}")
            for symbol in symbols:
                self.stream.subscribe_trades(on_trade, symbol)
                print(f"- {symbol} 거래 데이터 구독")
                
                self.stream.subscribe_quotes(on_quote, symbol)
                print(f"- {symbol} 호가 데이터 구독")
                
                # 1분봉 구독
                @self.stream.on_bar(symbol)
                async def _(bar):
                    await on_bar(bar)
                print(f"- {symbol} 바 데이터 구독")
            
            self._handlers_registered = True
            print("모든 이벤트 핸들러 등록 완료")
        
        try:
            
            print("웹소켓 데이터 스트림 시작...")
            await self.stream._run_forever()
        except Exception as e:
            logger.error(f"웹소켓 오류: {e}")
            raise

    def place_order(self, symbol: str, qty: float, side: str, 
                    order_type: str = "market", time_in_force: str = "day",
                    limit_price: float = None, stop_price: float = None) -> Dict[str, Any]:
       
        try:
            params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }
            
            if limit_price is not None and order_type in ["limit", "stop_limit"]:
                params["limit_price"] = str(limit_price)
            
            if stop_price is not None and order_type in ["stop", "stop_limit"]:
                params["stop_price"] = str(stop_price)
            
            order = self.rest.submit_order(**params)
            
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.order_type,
                "status": order.status,
                "created_at": order.created_at
            }
        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            raise

    def get_portfolio(self) -> List[Dict[str, Any]]:
        
        try:
            positions = self.rest.list_positions()
            
            result = []
            for p in positions:
                position_data = {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "change_today": float(p.change_today)
                }
                result.append(position_data)
                
                logger.info(f"{p.symbol}: {p.qty} 주 @ ${p.current_price} "
                           f"(평균가: ${p.avg_entry_price}, 손익: ${p.unrealized_pl})")
            
            return result
        except Exception as e:
            logger.error(f"포트폴리오 조회 실패: {e}")
            return []

    def get_account(self) -> Dict[str, Any]:
        
        try:
            account = self.rest.get_account()
            
            return {
                "id": account.id,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "status": account.status
            }
        except Exception as e:
            logger.error(f"계정 정보 조회 실패: {e}")
            return {"error": str(e)}
