#!/usr/bin/env python3
"""
업비트 API 연결 테스트 스크립트
- API 키 검증
- 잔고 조회
- 마켓 정보 조회
"""

import os
import sys
from dotenv import load_dotenv

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from exchange.UpbitAPI import UpbitAPI

def test_upbit_connection():
    """업비트 API 연결 테스트"""

    # 환경 변수 로드
    load_dotenv()

    print("=== 업비트 API 연결 테스트 ===\n")

    # API 키 확인
    access_key = os.getenv('UPBIT_ACCESS_KEY')
    secret_key = os.getenv('UPBIT_SECRET_KEY')

    if not access_key or access_key == 'your_upbit_access_key_here':
        print("❌ 업비트 ACCESS KEY가 설정되지 않았습니다.")
        print("   .env 파일에서 UPBIT_ACCESS_KEY를 설정해주세요.")
        return False

    if not secret_key or secret_key == 'your_upbit_secret_key_here':
        print("❌ 업비트 SECRET KEY가 설정되지 않았습니다.")
        print("   .env 파일에서 UPBIT_SECRET_KEY를 설정해주세요.")
        return False

    print("✅ API 키 설정 확인됨")

    # API 클라이언트 초기화
    try:
        upbit = UpbitAPI(access_key, secret_key)
        print("✅ 업비트 API 클라이언트 초기화 성공")
    except Exception as e:
        print(f"❌ API 클라이언트 초기화 실패: {e}")
        return False

    # 마켓 정보 조회 (공개 API)
    print("\n📊 마켓 정보 조회 테스트...")
    try:
        markets = upbit.GetAllMarkets()
        if markets:
            print(f"✅ 마켓 정보 조회 성공: {len(markets)}개 마켓")
            print(f"   주요 마켓: {', '.join([m.market for m in markets[:5]])}")
        else:
            print("❌ 마켓 정보 조회 실패")
            return False
    except Exception as e:
        print(f"❌ 마켓 정보 조회 오류: {e}")
        return False

    # 현재 시세 조회
    print("\n💰 BTC 현재 시세 조회...")
    try:
        ticker = upbit.GetTicker(['KRW-BTC'])
        if ticker:
            btc_price = ticker[0].trade_price
            print(f"✅ BTC 현재가: {btc_price:,}원")
        else:
            print("❌ 시세 정보 조회 실패")
    except Exception as e:
        print(f"❌ 시세 조회 오류: {e}")

    # 계정 정보 조회 (인증 필요)
    print("\n👤 계정 정보 조회 테스트...")
    try:
        accounts = upbit.GetAccountInfo()
        if accounts:
            print("✅ 계정 정보 조회 성공")
            print("   보유 자산:")
            for account in accounts:
                balance = float(account['balance'])
                if balance > 0:
                    currency = account['currency']
                    print(f"     {currency}: {balance:,.8f}")
        else:
            print("❌ 계정 정보 조회 실패 (API 키 권한을 확인해주세요)")
            return False
    except Exception as e:
        print(f"❌ 계정 정보 조회 오류: {e}")
        print("   API 키가 올바른지, 자산조회 권한이 있는지 확인해주세요.")
        return False

    # 원화 잔고 조회
    try:
        krw_balance = upbit.GetKRWBalance()
        print(f"   💴 원화 잔고: {krw_balance:,}원")
    except Exception as e:
        print(f"   ❌ 원화 잔고 조회 오류: {e}")

    print("\n🎉 업비트 API 연결 테스트 완료!")
    return True

def show_setup_guide():
    """설정 가이드 출력"""

    print("\n" + "="*50)
    print("📋 업비트 API 설정 가이드")
    print("="*50)
    print()
    print("1. 업비트 사이트 접속 및 로그인")
    print("   https://upbit.com")
    print()
    print("2. API 키 발급")
    print("   마이페이지 → Open API 관리 → API 키 발급")
    print()
    print("3. 필요한 권한 선택:")
    print("   ✅ 자산 조회 (필수)")
    print("   ✅ 주문 조회 (필수)")
    print("   ✅ 주문하기 (매매 시 필수)")
    print("   ✅ 출금하기 (지갑 연결 시 선택)")
    print()
    print("4. .env 파일 설정")
    print("   UPBIT_ACCESS_KEY=발급받은_액세스키")
    print("   UPBIT_SECRET_KEY=발급받은_시크릿키")
    print()
    print("5. 테스트 실행")
    print("   python test_upbit_connection.py")
    print()

if __name__ == "__main__":
    success = test_upbit_connection()

    if not success:
        show_setup_guide()
        sys.exit(1)
    else:
        print("\n🚀 업비트 연동이 정상적으로 설정되었습니다!")
        print("   이제 자동매매 AI를 실행할 수 있습니다.")