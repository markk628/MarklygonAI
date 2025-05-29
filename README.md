# MarklygonAI - Deep Learning Algorithmic Trading Platform

[![Project Status: 90% Complete](https://img.shields.io/badge/Status-90%25%20Complete-brightgreen)](https://github.com/your-repo/MarklygonAI)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 프로젝트 개요

MarklygonAI는 Eddie 시스템을 중심으로 한 고성능 딥러닝 알고리즘 트레이딩 플랫폼입니다. 40개 US 주식의 1분봉 데이터를 기반으로 실시간 매매 신호를 생성합니다.

### 🏆 주요 성과
- ✅ **40개 주식 데이터 완전 처리** (Feature Engineering 완료)
- ✅ **Eddie 시스템 구현** (Signal Generator + Pattern Analyzer)
- ✅ **메모리 최적화 시스템** (지능형 메모리 관리)
- ✅ **실시간 모니터링 대시보드**
- 🔄 **Phase 1 훈련 진행 중** (10개 종목)

## 📊 시스템 아키텍처

### Eddie 시스템 구성
```
📈 Signal Generator (TCN + Multi-head Attention)
├── 파라미터: ~3.3M
├── 출력: sell_intensity, uncertainty
└── 실시간 신호 생성

🔍 Pattern Analyzer (TimesNet + Wavelet)
├── 파라미터: ~2-3M
├── 출력: volatility, market_regime, trend_strength
└── 시장 패턴 분석
```

### 데이터 파이프라인
```
Raw Data (1분봉) → Feature Engineering (91개 지표) → PCA (20개 컴포넌트) → Sequences → Training
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# 의존성 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn ta-lib matplotlib seaborn

# 프로젝트 클론
git clone https://github.com/your-repo/MarklygonAI.git
cd MarklygonAI
```

### 2. 주요 실행 스크립트

#### 📊 통합 모니터링 (권장)
```bash
python unified_monitor.py          # 대화형 대시보드
python unified_monitor.py summary  # 프로젝트 요약
python unified_monitor.py cleanup  # 정리 도구
```

#### 🎯 Phase별 훈련
```bash
# Phase 1: 10개 종목 (진행 중)
python start_eddie_training.py

# Phase 2: 20개 종목 
python prepare_phase2_training.py

# Phase 3: 40개 종목 (최종)
python prepare_phase2_training.py  # 옵션 3 선택
```

#### 🔍 실시간 모니터링
```bash
python monitor_training_dashboard.py  # 고급 대시보드
python simple_monitor.py             # 간단한 상태 확인
```

## 📁 디렉토리 구조

```
MarklygonAI/
├── 📂 src/                    # 핵심 소스 코드
│   ├── models/eddie/          # Eddie 시스템
│   ├── preprocessing/         # 데이터 전처리
│   ├── utils/                 # 유틸리티 함수
│   └── config/               # 설정 파일
├── 📂 data/                   # 데이터 디렉토리
│   ├── raw/                  # 원본 JSON 데이터
│   └── feature_engineered/   # 처리된 CSV 데이터
├── 📂 outputs/               # 훈련 결과
├── 📂 memory-bank/           # 프로젝트 컨텍스트
├── 📂 documentation/         # 문서화
├── 🐍 unified_monitor.py     # 통합 모니터링 시스템
├── 🐍 prepare_phase2_training.py  # Phase 관리
└── 📋 README.md             # 이 파일
```

## 🎯 현재 진행 상황

### ✅ 완료된 작업 (90%)
1. **데이터 인프라**: 40개 주식 완전 처리
2. **Eddie 아키텍처**: Signal Generator + Pattern Analyzer
3. **메모리 시스템**: 지능형 메모리 관리자
4. **모니터링 도구**: 실시간 대시보드

### 🔄 진행 중인 작업
- **Phase 1 훈련**: 10개 종목 (NVDA, AAPL, MSFT, AMZN, JPM, BAC, GS, JNJ, MCD, KO)

### 📋 예정된 작업
- **Phase 2**: 20개 종목 확장
- **Phase 3**: 40개 종목 최종 훈련
- **백테스팅 시스템**: 성능 검증
- **실시간 API**: 웹 인터페이스

## 💻 시스템 요구사항

- **GPU**: NVIDIA RTX 4060 Ti 16GB (권장)
- **RAM**: 32GB 이상
- **Python**: 3.8+
- **CUDA**: 11.8+

## 📈 성능 지표

### 데이터 처리 성능
- **Feature Engineering**: 40/40 종목 완료
- **PCA 분산 설명률**: 97.5%
- **메모리 효율성**: 동적 청크 관리

### 모델 성능 (목표)
- **신호 정확도**: > 65%
- **추론 속도**: < 100ms
- **샤프 비율**: > 1.2

## 🔧 트러블슈팅

### 메모리 부족 시
```bash
python unified_monitor.py cleanup  # 임시 파일 정리
```

### 훈련 상태 확인
```bash
python unified_monitor.py summary  # 프로젝트 요약
python simple_monitor.py          # 빠른 상태 확인
```

## 📞 문의사항

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**MarklygonAI Team** | 2025 © All Rights Reserved