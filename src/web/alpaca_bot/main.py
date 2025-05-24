import asyncio
import time
import json
from datetime import datetime
from src.web.alpaca_bot.alpaca_client import AlpacaClient
from src.config.apikeys import API_KEY, API_SECRET
from src.web.alpaca_bot.config import BASE_URL, DATA_FEED, DEBUG

async def run_client(client, symbols):
    
    try:
        await client.start_websocket(symbols)
    except Exception as e:
        print(f"웹소켓 실행 중 오류: {e}")
        raise

def print_json(obj):
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'created_at' and hasattr(value, 'isoformat'):
                obj[key] = value.isoformat()
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)

def place_market_order(client):
    
    print("\n===== 시장가 주문 =====")
    
    symbol = input("주식 심볼(예: AAPL): ").strip().upper()
    if not symbol:
        print("주식 심볼을 입력해야 합니다.")
        return
    
    try:
        qty = float(input("수량(예: 1): ").strip())
        if qty <= 0:
            print("수량은 0보다 커야 합니다.")
            return
    except ValueError:
        print("유효한 숫자를 입력해야 합니다.")
        return
    
    side = input("매수/매도 (buy/sell): ").strip().lower()
    if side not in ['buy', 'sell']:
        print("'buy' 또는 'sell'을 입력해야 합니다.")
        return
    
    try:
        order = client.place_order(symbol, qty=qty, side=side)
        print("\n주문이 성공적으로 전송되었습니다!")
        print(f"주문 ID: {order.get('id', 'N/A')}")
        print(f"심볼: {order.get('symbol', 'N/A')}")
        print(f"수량: {order.get('qty', 'N/A')}")
        print(f"주문 방향: {order.get('side', 'N/A')}")
        print(f"주문 타입: {order.get('type', 'N/A')}")
        print(f"주문 상태: {order.get('status', 'N/A')}")
        print(f"생성 시간: {order.get('created_at', 'N/A')}")
        print(f"전체 주문 정보: {print_json(order)}")
    except Exception as e:
        print(f"주문 실행 중 오류 발생: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

def place_limit_order(client):
    
    print("\n===== 지정가 주문 =====")
    
    symbol = input("주식 심볼(예: AAPL): ").strip().upper()
    if not symbol:
        print("주식 심볼을 입력해야 합니다.")
        return
    
    try:
        qty = float(input("수량(예: 1): ").strip())
        if qty <= 0:
            print("수량은 0보다 커야 합니다.")
            return
    except ValueError:
        print("유효한 숫자를 입력해야 합니다.")
        return
    
    side = input("매수/매도 (buy/sell): ").strip().lower()
    if side not in ['buy', 'sell']:
        print("'buy' 또는 'sell'을 입력해야 합니다.")
        return
    
    try:
        limit_price = float(input("지정가(예: 150.00): ").strip())
        if limit_price <= 0:
            print("지정가는 0보다 커야 합니다.")
            return
    except ValueError:
        print("유효한 숫자를 입력해야 합니다.")
        return
    
    time_in_force = input("주문 유효 기간(day/gtc/ioc/fok, 기본값: day): ").strip().lower() or "day"
    if time_in_force not in ["day", "gtc", "ioc", "fok"]:
        print("유효하지 않은 주문 유효 기간입니다. 'day'로 설정합니다.")
        time_in_force = "day"
    
    try:
        order = client.place_order(
            symbol, 
            qty=qty, 
            side=side, 
            order_type="limit", 
            time_in_force=time_in_force,
            limit_price=limit_price
        )
        print("\n주문이 성공적으로 전송되었습니다!")
        print(f"주문 ID: {order.get('id', 'N/A')}")
        print(f"심볼: {order.get('symbol', 'N/A')}")
        print(f"수량: {order.get('qty', 'N/A')}")
        print(f"주문 방향: {order.get('side', 'N/A')}")
        print(f"주문 타입: {order.get('type', 'N/A')}")
        print(f"지정가: ${limit_price}")
        print(f"주문 상태: {order.get('status', 'N/A')}")
        print(f"생성 시간: {order.get('created_at', 'N/A')}")
        print(f"전체 주문 정보: {print_json(order)}")
    except Exception as e:
        print(f"주문 실행 중 오류 발생: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

def view_portfolio(client):
    """
    포트폴리오 조회
    """
    print("\n===== 포트폴리오 정보 =====")
    try:
        positions = client.get_portfolio()
        if positions:
            print(f"보유 포지션: {len(positions)}개")
            for pos in positions:
                print(f"- {pos['symbol']}: {pos['qty']} 주, 현재가: ${pos['current_price']}, 손익: ${pos['unrealized_pl']}")
        else:
            print("보유 중인 포지션이 없습니다.")
    except Exception as e:
        print(f"포트폴리오 조회 중 오류 발생: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

def view_account(client):
    
    print("\n===== 계정 정보 =====")
    try:
        account = client.get_account()
        print(f"현금 잔고: ${account.get('cash', 'N/A')}")
        print(f"포트폴리오 가치: ${account.get('portfolio_value', 'N/A')}")
        print(f"매수 가능 금액: ${account.get('buying_power', 'N/A')}")
        print(f"계정 상태: {account.get('status', 'N/A')}")
    except Exception as e:
        print(f"계정 정보 조회 중 오류 발생: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

def run_websocket(client):
    
    print("\n===== 실시간 시세 데이터 수신 =====")
    print("(Ctrl+C를 눌러 종료)")
    
    default_symbols = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]
    symbols_input = input(f"구독할 심볼(쉼표로 구분, 기본값: {', '.join(default_symbols)}): ").strip()
    
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
    else:
        symbols = default_symbols
    
    print(f"구독 심볼: {', '.join(symbols)}")
    print("참고: 무료 플랜('iex')은 최대 30개 심볼까지 구독 가능합니다.")
    
    try:
        try:
            asyncio.run(run_client(client, symbols))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_client(client, symbols))
    except KeyboardInterrupt:
        print("\n웹소켓 연결 종료...")
    except Exception as e:
        print(f"\n웹소켓 연결 중 오류 발생: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()

def show_menu():
    
    print("\n===== Alpaca 트레이딩 메뉴 =====")
    print("1: 시장가 주문")
    print("2: 지정가 주문")
    print("3: 포트폴리오 조회")
    print("4: 계정 정보 조회")
    print("5: 실시간 시세 데이터 수신 (웹소켓)")
    print("0: 종료")
    return input("\n원하는 작업을 선택하세요: ").strip()

def main():
    print("Alpaca API 클라이언트 초기화 중...")
    client = AlpacaClient(API_KEY, API_SECRET, BASE_URL, data_feed=DATA_FEED)
    print(f"데이터 피드: {DATA_FEED} (무료 플랜, 최대 30개 심볼 제한)")
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            place_market_order(client)
        elif choice == "2":
            place_limit_order(client)
        elif choice == "3":
            view_portfolio(client)
        elif choice == "4":
            view_account(client)
        elif choice == "5":
            run_websocket(client)
        elif choice == "0":
            print("\n프로그램을 종료합니다.")
            break
        else:
            print("\n잘못된 선택입니다. 다시 시도하세요.")
        
        if choice in ["1", "2"]:
            print("\n주문 처리 중... 3초 대기")
            time.sleep(3)

if __name__ == "__main__":
    main()