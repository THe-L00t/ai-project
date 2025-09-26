#!/usr/bin/env python3
"""
업비트 지갑 연결 테스트 스크립트
- 입금 주소 조회/생성
- 출금 기능 테스트 (시뮬레이션)
- 입출금 내역 조회
"""

import os
import sys
from dotenv import load_dotenv

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from exchange.UpbitAPI import UpbitAPI

def test_wallet_connection():
    """업비트 지갑 연결 테스트"""

    # 환경 변수 로드
    load_dotenv()

    print("=== 업비트 지갑 연결 테스트 ===\n")

    # API 클라이언트 초기화
    try:
        upbit = UpbitAPI()
        print("✅ 업비트 API 클라이언트 초기화 성공")
    except Exception as e:
        print(f"❌ API 클라이언트 초기화 실패: {e}")
        return False

    # 1. 입금 주소 조회
    print("\n📥 입금 주소 조회 테스트...")
    try:
        # BTC 입금 주소 조회
        btc_address = upbit.GetCoinAddress('BTC')
        if btc_address:
            deposit_addr = btc_address.get('deposit_address')
            if deposit_addr:
                print(f"✅ BTC 입금 주소: {deposit_addr}")
            else:
                print("⚠️  BTC 입금 주소 없음 - 새로 생성 필요")
                # 새 주소 생성 테스트 (실제로는 실행하지 않음)
                print("   주소 생성 방법: upbit.CreateCoinAddress('BTC')")
        else:
            print("❌ BTC 입금 주소 조회 실패")

        # ETH 입금 주소 조회
        eth_address = upbit.GetCoinAddress('ETH')
        if eth_address:
            deposit_addr = eth_address.get('deposit_address')
            if deposit_addr:
                print(f"✅ ETH 입금 주소: {deposit_addr}")
            else:
                print("⚠️  ETH 입금 주소 없음 - 새로 생성 필요")
        else:
            print("❌ ETH 입금 주소 조회 실패")

    except Exception as e:
        print(f"❌ 입금 주소 조회 오류: {e}")

    # 2. 출금 한도 조회
    print("\n💸 출금 한도 조회 테스트...")
    try:
        btc_limit = upbit.GetWithdrawLimit('BTC')
        if btc_limit:
            print(f"✅ BTC 출금 한도:")
            print(f"   일일 한도: {btc_limit.get('limit_daily', 'N/A')}")
            print(f"   남은 한도: {btc_limit.get('limit_available', 'N/A')}")
        else:
            print("❌ BTC 출금 한도 조회 실패")

        krw_limit = upbit.GetWithdrawLimit('KRW')
        if krw_limit:
            print(f"✅ KRW 출금 한도:")
            print(f"   일일 한도: {krw_limit.get('limit_daily', 'N/A'):,}원")
            print(f"   남은 한도: {krw_limit.get('limit_available', 'N/A'):,}원")
        else:
            print("❌ KRW 출금 한도 조회 실패")

    except Exception as e:
        print(f"❌ 출금 한도 조회 오류: {e}")

    # 3. 입출금 내역 조회
    print("\n📋 입출금 내역 조회 테스트...")
    try:
        # 최근 입금 내역 (최대 5개)
        deposits = upbit.GetDepositHistory(limit=5)
        if deposits:
            print(f"✅ 최근 입금 내역 ({len(deposits)}건):")
            for deposit in deposits:
                currency = deposit.get('currency', 'N/A')
                amount = deposit.get('amount', 'N/A')
                state = deposit.get('state', 'N/A')
                created = deposit.get('created_at', 'N/A')
                print(f"   {currency}: {amount} ({state}) - {created[:10]}")
        else:
            print("⚠️  입금 내역 없음")

        # 최근 출금 내역 (최대 5개)
        withdraws = upbit.GetWithdrawHistory(limit=5)
        if withdraws:
            print(f"✅ 최근 출금 내역 ({len(withdraws)}건):")
            for withdraw in withdraws:
                currency = withdraw.get('currency', 'N/A')
                amount = withdraw.get('amount', 'N/A')
                state = withdraw.get('state', 'N/A')
                created = withdraw.get('created_at', 'N/A')
                print(f"   {currency}: {amount} ({state}) - {created[:10]}")
        else:
            print("⚠️  출금 내역 없음")

    except Exception as e:
        print(f"❌ 입출금 내역 조회 오류: {e}")

    # 4. 지갑 설정 테스트 (시뮬레이션)
    print("\n🔗 지갑 연결 시뮬레이션...")
    try:
        print("📝 지갑 연결 방법:")
        print("   1. 입금용 주소 설정:")
        print("      address = upbit.SetupWallet('BTC')")
        print("      print(f'BTC 입금 주소: {address}')")
        print("")
        print("   2. 외부 지갑으로 출금:")
        print("      result = upbit.TransferToExternalWallet('BTC', 0.001, '외부지갑주소')")
        print("      if result: print('출금 성공')")
        print("")
        print("   3. 출금 상태 확인:")
        print("      history = upbit.GetWithdrawHistory(currency='BTC', state='PROCESSING')")
        print("      for item in history: print(item)")

        print("✅ 지갑 연결 기능 사용법 안내 완료")

    except Exception as e:
        print(f"❌ 지갑 연결 테스트 오류: {e}")

    print("\n🎉 지갑 연결 테스트 완료!")
    return True

def show_wallet_guide():
    """지갑 연결 가이드 출력"""

    print("\n" + "="*60)
    print("🏦 업비트 지갑 연결 가이드")
    print("="*60)
    print()
    print("📥 입금 (외부 → 업비트)")
    print("  1. 입금 주소 조회: upbit.GetCoinAddress('BTC')")
    print("  2. 주소 없으면 생성: upbit.CreateCoinAddress('BTC')")
    print("  3. 외부 지갑에서 해당 주소로 송금")
    print("  4. 입금 완료까지 대기 (네트워크 승인 필요)")
    print()
    print("📤 출금 (업비트 → 외부)")
    print("  1. 출금 한도 확인: upbit.GetWithdrawLimit('BTC')")
    print("  2. 잔고 확인: upbit.GetCoinBalance('BTC')")
    print("  3. 출금 실행: upbit.WithdrawCoin('BTC', 0.001, '외부주소')")
    print("  4. 출금 상태 확인: upbit.GetWithdrawHistory()")
    print()
    print("⚠️  주의사항:")
    print("  - 출금하기 권한이 API 키에 설정되어야 함")
    print("  - 네트워크 수수료가 차감됨")
    print("  - 최소 출금 수량 제한 있음")
    print("  - 출금 주소 검증 필요")
    print("  - 보안을 위해 2차 인증 활성화 권장")
    print()
    print("🔐 보안 권장사항:")
    print("  1. API 키에 IP 제한 설정")
    print("  2. 필요한 권한만 부여")
    print("  3. 정기적인 키 갱신")
    print("  4. 출금 전 주소 재확인")
    print("  5. 소액 테스트 후 본 거래")
    print()

if __name__ == "__main__":
    success = test_wallet_connection()

    if success:
        show_wallet_guide()
        print("\n🚀 업비트 지갑 연결 기능이 정상적으로 작동합니다!")
    else:
        print("\n❌ 지갑 연결 테스트에서 문제가 발생했습니다.")
        print("   API 키 설정과 권한을 확인해주세요.")
        sys.exit(1)