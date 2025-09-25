# 업비트 연동 및 지갑 연결 가이드

## 개요
이 가이드는 코인 자동매매 AI에서 업비트 거래소와 코인 지갑을 연결하는 방법을 안내합니다.

## 📋 사전 준비사항

### 1. 업비트 계정 및 API 키 준비
1. [업비트 사이트](https://upbit.com)에서 계정 생성 및 본인인증 완료
2. 투자자보호 의무사항 이수 완료
3. OTP(2차 인증) 설정 완료

### 2. API 키 발급
1. 업비트 로그인 → **마이페이지** → **Open API 관리**
2. **새 API 키 발급** 클릭
3. 필요한 권한 선택:
   - ✅ **자산 조회** (필수 - 잔고 확인)
   - ✅ **주문 조회** (필수 - 주문 내역 확인)
   - ✅ **주문하기** (필수 - 자동매매 실행)
   - ✅ **출금하기** (선택 - 지갑 연결시 필요)

4. IP 제한 설정 (보안 강화를 위해 권장)
5. 발급된 **Access Key**와 **Secret Key** 저장

## 🔧 환경 설정

### 1. 의존성 패키지 설치
```bash
cd Project/CoinTradingAI
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일에서 다음 항목들을 설정:

```env
# 업비트 API 키 설정
UPBIT_ACCESS_KEY=3BIjkcIGlczJTYUbasvfpX4xqnDbja7CHSPxxiSc
UPBIT_SECRET_KEY=NT4kID2GBely8fMDCx2EAID7oXU4XIIBeWoP8T7f

# 거래소 설정
EXCHANGE_NAME=upbit
SANDBOX_MODE=false

# 거래 모드 설정
TRADING_MODE=paper  # 모의투자로 시작
# TRADING_MODE=live  # 실거래 (충분한 테스트 후 변경)
```

### 3. 연결 테스트 실행
```bash
# 기본 API 연결 테스트
python test_upbit_connection.py

# 지갑 연결 테스트
python test_wallet_connection.py
```

## 💰 주요 기능 사용법

### 1. 기본 API 사용
```python
from src.exchange.UpbitAPI import UpbitAPI

# API 클라이언트 초기화
upbit = UpbitAPI()

# 계정 정보 조회
accounts = upbit.GetAccountInfo()
print("보유 자산:", accounts)

# 원화 잔고 조회
krw_balance = upbit.GetKRWBalance()
print(f"원화 잔고: {krw_balance:,}원")

# BTC 현재가 조회
ticker = upbit.GetTicker(['KRW-BTC'])
print(f"BTC 현재가: {ticker[0].trade_price:,}원")
```

### 2. 주문 실행
```python
# 시장가 매수 (10,000원어치)
result = upbit.BuyMarket('KRW-BTC', 10000)
if result:
    print("매수 주문 성공:", result['uuid'])

# 지정가 매도 (0.001 BTC를 50,000,000원에)
result = upbit.SellLimit('KRW-BTC', 0.001, 50000000)
if result:
    print("매도 주문 성공:", result['uuid'])

# 미체결 주문 조회
open_orders = upbit.GetOpenOrders()
print("미체결 주문:", len(open_orders))
```

### 3. 지갑 연결 및 입출금

#### 입금 (외부 지갑 → 업비트)
```python
# BTC 입금 주소 조회/생성
deposit_address = upbit.SetupWallet('BTC')
print(f"BTC 입금 주소: {deposit_address}")

# 입금 내역 확인
deposits = upbit.GetDepositHistory('BTC', limit=10)
for deposit in deposits:
    print(f"입금: {deposit['amount']} BTC ({deposit['state']})")
```

#### 출금 (업비트 → 외부 지갑)
```python
# 출금 한도 확인
limit_info = upbit.GetWithdrawLimit('BTC')
print(f"BTC 출금 가능: {limit_info['limit_available']}")

# 외부 지갑으로 출금 (0.001 BTC)
external_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # 예시 주소
result = upbit.TransferToExternalWallet('BTC', 0.001, external_address)
if result:
    print("출금 요청 성공:", result['uuid'])

# 출금 상태 확인
withdraws = upbit.GetWithdrawHistory('BTC', state='PROCESSING')
for withdraw in withdraws:
    print(f"출금 처리중: {withdraw['amount']} BTC")
```

## 🚀 자동매매 AI 실행

### 1. 모의투자로 시작
```bash
# 환경 설정에서 TRADING_MODE=paper 확인 후
python main.py
```

### 2. 실거래 전환 (충분한 테스트 후)
```env
# .env 파일에서
TRADING_MODE=live
```

### 3. 웹 대시보드 실행
```bash
python run_dashboard.py
```
브라우저에서 `http://localhost:8080` 접속

## ⚠️ 주의사항 및 보안

### 1. API 키 보안
- ✅ `.env` 파일을 절대 공개 저장소에 업로드하지 마세요
- ✅ API 키에 IP 제한을 설정하세요
- ✅ 필요한 권한만 부여하세요
- ✅ 정기적으로 키를 갱신하세요

### 2. 거래 안전수칙
- ✅ 모의투자로 충분히 테스트하세요
- ✅ 손절매 설정을 확인하세요
- ✅ 투자 금액을 제한하세요
- ✅ 리스크 관리 설정을 점검하세요

### 3. 출금 관련 주의
- ✅ 출금 주소를 반드시 재확인하세요
- ✅ 소액으로 테스트 후 본 거래하세요
- ✅ 네트워크 수수료를 고려하세요
- ✅ 최소 출금 수량을 확인하세요

## 🔧 문제 해결

### 1. API 연결 실패
```
❌ API 키가 설정되지 않았습니다
```
→ `.env` 파일에서 `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY` 확인

### 2. 권한 오류
```
❌ 계정 정보 조회 실패 (API 키 권한을 확인해주세요)
```
→ 업비트에서 API 키에 필요한 권한이 부여되었는지 확인

### 3. 지갑 주소 생성 실패
```
⚠️ BTC 입금 주소 없음 - 새로 생성 필요
```
→ `upbit.CreateCoinAddress('BTC')` 실행 후 잠시 대기

### 4. 출금 실패
```
❌ 출금 한도 초과 또는 잔고 부족
```
→ `GetWithdrawLimit()`과 `GetCoinBalance()`로 한도/잔고 확인

## 📞 지원

### 공식 문서
- [업비트 Open API 문서](https://docs.upbit.com/)
- [업비트 개발자 센터](https://upbit.com/service_center/open_api_guide)

### 문의사항
프로젝트 이슈나 문의사항은 GitHub Issues에 등록해주세요.

---

## 📈 추가 기능

### 실시간 시세 구독
```python
def on_ticker(data):
    print(f"실시간 시세: {data['code']} {data['trade_price']:,}원")

# 웹소켓 연결 설정
upbit.ConnectWebSocket(on_message=on_ticker)
upbit.SubscribeTicker(['KRW-BTC', 'KRW-ETH'])
upbit.StartWebSocket()  # 블로킹 모드로 실행
```

### 캔들 데이터 조회
```python
# 1분봉 데이터 (최근 100개)
candles = upbit.GetCandles('KRW-BTC', 'minutes', 1, 100)
for candle in candles[:5]:
    print(f"시간: {candle['candle_date_time_kst']}, "
          f"시가: {candle['opening_price']:,}, "
          f"고가: {candle['high_price']:,}, "
          f"저가: {candle['low_price']:,}, "
          f"종가: {candle['trade_price']:,}")
```

이제 업비트와 코인 지갑이 성공적으로 연결되었습니다! 🚀
