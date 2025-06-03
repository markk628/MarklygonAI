# Alpaca API 트레이딩 프로그램

이 프로젝트는 Alpaca API를 사용하여 주식 거래를 위한 Python 기반 트레이딩 프로그램입니다.

## 주요 기능

- **실시간 시세 데이터 수신**: 웹소켓을 통한 거래, 호가, 바 데이터 스트리밍
- **주문 처리**: 시장가, 지정가 주문 실행
- **포트폴리오 관리**: 보유 포지션 정보 조회
- **계정 정보 조회**: 현금 잔고, 포트폴리오 가치, 매수 가능 금액 확인

## 파일 구조

- **alpaca_bot/alpaca_client.py**: Alpaca API와 통신하는 AlpacaClient 클래스
- **alpaca_bot/config.py**: API 키, 시크릿, 기본 URL 등의 설정값
- **alpaca_bot/main.py**: 사용자 인터페이스 및 메뉴 시스템

## 설치 방법

1. 저장소 클론:
```
git clone <저장소 URL>
cd alphaca_api
```

2. 필요한 패키지 설치:
```
pip install -r requirements.txt
```

## 설정 방법

`config.py` 파일에서 Alpaca API 키와 시크릿을 설정합니다:

```python
API_KEY = "YOUR_API_KEY_HERE"
API_SECRET = "YOUR_API_SECRET_HERE"
```

또는 환경 변수를 통해 설정할 수 있습니다:
```
export APCA_API_KEY_ID="YOUR_API_KEY_HERE"
export APCA_API_SECRET_KEY="YOUR_API_SECRET_HERE"
```

## 실행 방법

```
python alpaca_bot/main.py
```

## 메뉴 항목

1. 시장가 주문
2. 지정가 주문
3. 포트폴리오 조회
4. 계정 정보 조회
5. 실시간 시세 데이터 수신
0. 종료

## 주의사항

- 무료 'iex' 데이터 피드는 15분 지연 데이터를 제공하며, 최대 30개 심볼까지 구독 가능합니다.
- 실시간 데이터는 미국 시장 거래 시간(미 동부 시간 9:30-16:00)에만 수신됩니다.
- 이 프로그램은 Alpaca의 종이 거래(Paper Trading) 계정으로 설정되어 있어 실제 자금을 사용하지 않습니다.

## API 문서

Alpaca API에 대한 자세한 내용은 [공식 문서](https://alpaca.markets/docs/api-documentation/)를 참조하세요. 