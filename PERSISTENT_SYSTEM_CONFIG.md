# 🔄 CoinTradingAI 지속적 시스템 설정

## 📋 중요한 사용자 설정 (절대 변경 금지)

### ⚡ 단타 매매 설정
```
사이클 간격: 10초 (단타 최적화)
API 호출: 배치 + 캐싱으로 최적화
목표: 빠른 반응속도 + API 안정성
```

### 🎯 운영 모드 분리
```
1. 학습 모드: run_learning_mode.py
   - 매매 없이 순수 학습만
   - 데이터 수집 및 모델 훈련
   - 백테스팅 및 성능 검증

2. 매매 모드: run_trading_mode.py
   - 10초 사이클 단타 매매
   - 실제 자금으로 거래
   - 리스크 관리 자동화
```

### 🖥️ 데스크톱 단축키
```
start_ai_learning.command: 학습 전용 모드
start_ai_trading.command: 실제 매매 모드
view_ai_logs.command: 로그 모니터링
```

### 📊 포지션 관리
```
이전 포지션 자동 로딩: 활성화
대상 코인: KRW-BTC, KRW-ETH, KRW-ADA, KRW-DOT + 보유 코인
최소 포지션 크기: 1000원 이상
```

---

## ⚠️ 시스템 복구 지침

### 문제 발생 시 체크리스트

#### 1. API 429 에러 발생 시
- [ ] AI_SETTINGS.md에서 TRADING_CYCLE_SECONDS = 10 확인
- [ ] 배치 요청 시스템 작동 확인
- [ ] API 캐싱 (5초 TTL) 정상 작동 확인

#### 2. 포지션 로딩 실패 시
- [ ] 계정 정보 접근 권한 확인
- [ ] API 키 유효성 확인
- [ ] 대상 코인 목록 업데이트

#### 3. Timestamp 에러 발생 시
- [ ] get_position_entry_time() 함수 존재 확인
- [ ] normalize_position_fields() 호출 확인
- [ ] entry_time 필드 표준화 확인

#### 4. 설정 반영 안됨
- [ ] config_loader.py 정상 로딩 확인
- [ ] AI_SETTINGS.md 문법 오류 확인
- [ ] 설정 파일 권한 확인

---

## 🛠️ 핵심 기능 보존 설정

### API 최적화 (절대 변경 금지)
```python
# 배치 요청 시스템
all_tickers = self.get_cached_ticker(self.target_coins)

# 5초 TTL 캐싱
self.cache_ttl = 5

# 개별 호출 금지 - 반드시 배치 사용
```

### 단타 사이클 (사용자 지정)
```python
# AI_SETTINGS.md에서 관리
cycle_interval = self.config.get('TRADING_CYCLE_SECONDS', 10)

# 기본값: 10초 (단타 최적화)
# 사용자가 원하면 변경 가능
```

### 포지션 필드 표준화 (자동화)
```python
# 매 사이클마다 자동 실행
self.normalize_position_fields()

# entry_time 필드로 통일
# timestamp -> entry_time 자동 변환
```

### 가중치 기반 신호 통합
```python
# AI_SETTINGS.md 연동
self.news_sentiment_weight = self.config.get('NEWS_SENTIMENT_WEIGHT', 0.8)
self.aggressive_pattern_weight = self.config.get('AGGRESSIVE_PATTERN_WEIGHT', 0.7)
self.adaptive_learning_weight = self.config.get('ADAPTIVE_LEARNING_WEIGHT', 0.6)
self.pattern_model_weight = self.config.get('PATTERN_MODEL_WEIGHT', 0.6)
```

---

## 📚 모드별 실행 방법

### 학습 모드 실행
```bash
# 방법 1: 데스크톱 단축키
더블클릭: start_ai_learning.command

# 방법 2: 직접 실행
python3 run_learning_mode.py

# 특징: 매매 없이 순수 학습만
```

### 매매 모드 실행
```bash
# 방법 1: 데스크톱 단축키
더블클릭: start_ai_trading.command

# 방법 2: 직접 실행
python3 run_trading_mode.py

# 특징: 10초 사이클 실제 거래
```

### 통합 모드 실행 (학습+매매)
```bash
# 통합 실행
python3 main.py

# 또는
python3 smart_hybrid_ai.py
```

---

## 🔧 문제 해결 스크립트

### 시스템 상태 점검
```bash
# 전체 시스템 점검
python3 -c "
import smart_hybrid_ai
ai = smart_hybrid_ai.SmartHybridAI()
print('✅ 시스템 정상 초기화')
print(f'사이클: {ai.config.get(\"TRADING_CYCLE_SECONDS\")}초')
print(f'포지션 크기: {ai.max_position_size*100}%')
print(f'손절: {ai.stop_loss_percentage}%')
print(f'익절: {ai.take_profit_percentage}%')
"
```

### 설정 파일 검증
```bash
# AI_SETTINGS.md 검증
python3 -c "
from config_loader import get_config
config = get_config()
print('✅ 설정 파일 로딩 성공')
for key, value in config.items():
    print(f'{key}: {value}')
"
```

### API 연결 테스트
```bash
# API 연결 상태 확인
python3 -c "
from src.exchange.UpbitAPI import UpbitAPI
upbit = UpbitAPI()
accounts = upbit.GetAccountInfo()
print(f'✅ API 연결 성공: {len(accounts)}개 계정')
"
```

---

## 💾 백업 및 복구

### 중요 파일 백업 목록
```
필수 백업 파일:
- AI_SETTINGS.md (사용자 설정)
- smart_hybrid_ai.py (메인 로직)
- config_loader.py (설정 로더)
- run_learning_mode.py (학습 모드)
- run_trading_mode.py (매매 모드)
- models/ (학습된 모델들)
- logs/ (거래 기록)

데스크톱 단축키:
- start_ai_learning.command
- start_ai_trading.command
- view_ai_logs.command
```

### 복구 순서
1. 백업 파일 복원
2. 권한 설정: `chmod +x *.command`
3. 의존성 설치: `pip3 install -r requirements.txt`
4. 설정 검증: 위의 검증 스크립트 실행
5. 테스트 실행: 학습 모드로 먼저 테스트

---

**마지막 업데이트**: 2025-09-27
**설정 담당자**: 사용자 + Claude Code AI
**중요도**: 🔴 최고 (절대 임의 변경 금지)**