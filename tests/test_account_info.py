#!/usr/bin/env python3
"""
업비트 계정 정보 상세 조회 테스트
평균 매수가 정보가 있는지 확인
"""

import os
import sys
from dotenv import load_dotenv
import json

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from exchange.UpbitAPI import UpbitAPI

def test_account_info():
    """계정 정보 상세 조회"""
    load_dotenv()

    upbit = UpbitAPI()

    try:
        accounts = upbit.GetAccountInfo()

        print("📊 업비트 계정 정보 상세:")
        print("=" * 50)

        for account in accounts:
            currency = account.get('currency', '?')
            balance = float(account.get('balance', 0))

            if balance > 0:
                print(f"\n🪙 {currency}:")
                print(f"   balance: {balance}")

                # 모든 필드 출력
                for key, value in account.items():
                    if key != 'currency':
                        print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    test_account_info()