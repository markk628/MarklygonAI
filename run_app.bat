@echo off
echo MarklygonAI Flask 앱 실행 중...
echo.

REM 의존성 설치 확인
python -c "import flask_login" 2>nul
if errorlevel 1 (
    echo Flask-Login이 설치되지 않았습니다. 설치 중...
    pip install Flask-Login==0.6.3
)

python -c "import flask_sqlalchemy" 2>nul
if errorlevel 1 (
    echo Flask-SQLAlchemy가 설치되지 않았습니다. 설치 중...
    pip install Flask-SQLAlchemy==3.1.1
)

python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask가 설치되지 않았습니다. 설치 중...
    pip install Flask==3.1.1
)

echo.
echo Flask 앱 시작...
echo 브라우저에서 http://localhost:5000 으로 접속하세요
echo 데모 계정: demo / demo123
echo.
python app.py 