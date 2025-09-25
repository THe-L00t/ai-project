#!/usr/bin/env python3
"""
ETH 매도 및 자동매매 시작 스크립트
"""

import os
import sys
import time
from dotenv import load_dotenv

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from exchange.UpbitAPI import UpbitAPI

def sell_eth_and_start_trading():
    """ETH 매도 및 자동매매 시작"""

    # 환경 변수 로드
    load_dotenv()

    print("=== ETH 매도 및 자동매매 시작 ===\n")

    # API 클라이언트 초기화
    try:
        upbit = UpbitAPI()
        print("✅ 업비트 API 연결 성공")
    except Exception as e:
        print(f"❌ API 연결 실패: {e}")
        return False

    # 1. 현재 ETH 보유량 확인
    print("\n1️⃣ ETH 보유량 확인...")
    try:
        eth_balance = upbit.GetCoinBalance('ETH')
        print(f"   현재 ETH 보유량: {eth_balance} ETH")

        if eth_balance <= 0:
            print("❌ 매도할 ETH가 없습니다.")
            return False

    except Exception as e:
        print(f"❌ ETH 잔고 조회 실패: {e}")
        return False

    # 2. ETH 현재 시세 확인
    print("\n2️⃣ ETH 현재 시세 확인...")
    try:
        ticker = upbit.GetTicker(['KRW-ETH'])
        if ticker:
            eth_price = ticker[0].trade_price
            estimated_krw = eth_balance * eth_price
            print(f"   ETH 현재가: {eth_price:,}원")
            print(f"   매도 예상 금액: {estimated_krw:,.0f}원")
        else:
            print("❌ ETH 시세 조회 실패")
            return False
    except Exception as e:
        print(f"❌ 시세 조회 실패: {e}")
        return False

    # 3. ETH 전량 매도 실행
    print(f"\n3️⃣ ETH {eth_balance} 전량 매도 실행...")
    try:
        # 시장가 매도 (전량)
        sell_result = upbit.SellMarket('KRW-ETH', eth_balance)

        if sell_result:
            order_uuid = sell_result['uuid']
            print(f"✅ 매도 주문 성공!")
            print(f"   주문 UUID: {order_uuid}")
            print(f"   매도 수량: {eth_balance} ETH")

            # 주문 완료까지 대기
            print("   매도 체결 대기 중...")
            for i in range(30):  # 최대 30초 대기
                time.sleep(1)
                order_info = upbit.GetOrder(order_uuid)
                if order_info and order_info.state == 'done':
                    print(f"✅ 매도 완료!")
                    print(f"   체결 수량: {order_info.executed_volume} ETH")
                    print(f"   체결 금액: {float(order_info.executed_volume) * eth_price:,.0f}원")
                    break
                elif i % 5 == 0:  # 5초마다 상태 출력
                    print(f"   대기 중... ({i+1}/30초)")
            else:
                print("⚠️  매도 체결 확인 타임아웃 (주문은 진행 중)")
        else:
            print("❌ 매도 주문 실패")
            return False

    except Exception as e:
        print(f"❌ 매도 실행 실패: {e}")
        return False

    # 4. 매도 후 잔고 확인
    print("\n4️⃣ 매도 후 잔고 확인...")
    try:
        time.sleep(2)  # 잠시 대기
        krw_balance = upbit.GetKRWBalance()
        eth_balance_after = upbit.GetCoinBalance('ETH')

        print(f"   원화 잔고: {krw_balance:,.2f}원")
        print(f"   ETH 잔고: {eth_balance_after} ETH")

        if krw_balance < 100:
            print("⚠️  원화 잔고가 부족합니다. 자동매매에는 최소 100원 이상 필요합니다.")

    except Exception as e:
        print(f"❌ 잔고 확인 실패: {e}")

    print("\n🎉 ETH 매도 완료!")
    return True

def show_trading_options():
    """자동매매 옵션 안내"""
    print("\n" + "="*60)
    print("🤖 자동매매 설정 옵션")
    print("="*60)
    print()
    print("📊 거래 모드:")
    print("   - paper: 모의투자 (안전한 테스트)")
    print("   - live: 실거래 (실제 자금 사용)")
    print()
    print("💰 주요 설정:")
    print("   - 최대 포지션 크기: 10% (총 자금의 10%)")
    print("   - 손절매: 5% 손실시 자동 매도")
    print("   - 익절매: 10% 이익시 자동 매도")
    print("   - 일일 최대 손실: 100,000원")
    print()
    print("🎯 추천 시작 방법:")
    print("   1. 모의투자로 3-7일 테스트")
    print("   2. 수익률 확인 후 실거래 전환")
    print("   3. 소액부터 시작하여 점진적 확대")
    print()
    print("🚀 자동매매 시작 명령어:")
    print("   python3 main.py")
    print()

if __name__ == "__main__":
    success = sell_eth_and_start_trading()

    if success:
        show_trading_options()

        print("\n💡 다음 단계:")
        print("1. ETH 매도 완료 ✅")
        print("2. 자동매매 AI 실행 준비 완료")
        print("3. 명령어 실행: python3 main.py")
        print()

    else:
        print("\n❌ ETH 매도 실패")
        print("   문제를 해결한 후 다시 시도해주세요.")
        sys.exit(1)