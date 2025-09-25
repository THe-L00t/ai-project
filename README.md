# 🤖 코인 자동매매 AI 시스템

자동화된 암호화폐 거래를 통한 안정적 수익 창출을 목표로 하는 AI 기반 트레이딩 시스템입니다.

## 📁 프로젝트 구조

```
CoinTradingAI/
├── src/                    # 소스 코드
│   ├── data_collector/     # 데이터 수집
│   ├── preprocessor/       # 데이터 전처리
│   ├── models/            # AI 모델
│   ├── strategies/        # 거래 전략
│   ├── trader/            # 실거래 실행
│   ├── risk_manager/      # 리스크 관리
│   └── utils/             # 유틸리티 함수
├── config/                # 설정 파일
├── data/                  # 데이터 저장소
├── logs/                  # 로그 파일
├── models/                # 학습된 모델
├── strategies/            # 전략 정의
├── backtests/            # 백테스팅 결과
├── tests/                # 테스트 코드
└── docs/                 # 문서
```

## 🚀 시작하기

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정 파일 준비
```bash
# 환경 변수 파일 복사 및 설정
cp .env.example .env
# .env 파일을 열어서 실제 API 키와 설정값 입력
```

### 3. 프로젝트 설정
- `project_config.md` 파일을 열어서 모든 [ ] 항목을 채워주세요
- 거래소 API 키, 투자 목표, 리스크 설정 등 필수 정보 입력

## ⚠️ 보안 주의사항

- **절대로** API 키나 시크릿 키를 코드에 직접 입력하지 마세요
- `.env` 파일은 Git에 커밋되지 않도록 주의하세요
- 처음에는 반드시 **페이퍼 트레이딩**으로 시작하세요
- 실거래 전 충분한 백테스팅과 검증을 진행하세요

## 📊 개발 단계

1. **Phase 1**: 기반 구축 (데이터 수집, API 연동)
2. **Phase 2**: AI 모델 개발 (예측 모델, 전략)
3. **Phase 3**: 고도화 (리스크 관리, 모니터링)
4. **Phase 4**: 배포 및 운영 (실거래, 성과 분석)

## 🎯 주요 기능 (예정)

- [ ] 다중 거래소 데이터 수집
- [ ] AI 기반 가격 예측
- [ ] 자동 거래 실행
- [ ] 리스크 관리 시스템
- [ ] 실시간 모니터링 대시보드
- [ ] 백테스팅 및 성과 분석
- [ ] 텔레그램 알림 시스템

## 📈 사용법

### 백테스팅 실행
```bash
python -m src.backtester --strategy=ma_crossover --period=1y
```

### 페이퍼 트레이딩 시작
```bash
python -m src.trader --mode=paper
```

### 실거래 시작 (충분한 테스트 후)
```bash
python -m src.trader --mode=live
```

## 🔧 개발 환경

- **Python**: 3.9+
- **주요 라이브러리**: pandas, scikit-learn, ccxt, backtrader
- **데이터베이스**: SQLite (기본) / PostgreSQL (옵션)
- **모니터링**: Dash + Plotly

## 📞 문의 및 지원

프로젝트 관련 문의사항이나 버그 리포트는 이슈를 통해 제보해주세요.

---

**⚠️ 면책 조항**: 이 프로젝트는 교육 및 학습 목적으로 제작되었습니다. 암호화폐 거래는 높은 위험을 수반하며, 투자 손실에 대한 책임은 사용자에게 있습니다.