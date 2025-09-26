#!/usr/bin/env python3
"""
업비트 API 인증 문제 진단 및 수정
"""

import os
import sys
import jwt
import uuid
import hashlib
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

def test_jwt_generation():
    """JWT 토큰 생성 테스트"""
    access_key = os.getenv('UPBIT_ACCESS_KEY')
    secret_key = os.getenv('UPBIT_SECRET_KEY')

    print(f"🔑 액세스 키: {access_key[:10]}...")
    print(f"🔑 시크릿 키: {secret_key[:10]}...")

    if not access_key or not secret_key:
        print("❌ API 키가 설정되지 않았습니다")
        return None

    # JWT 페이로드 생성
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }

    # JWT 토큰 생성
    try:
        token = jwt.encode(payload, secret_key, algorithm='HS256')
        print(f"✅ JWT 토큰 생성 성공: {token[:50]}...")
        return token
    except Exception as e:
        print(f"❌ JWT 토큰 생성 실패: {e}")
        return None

def test_api_request(token):
    """API 요청 테스트"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    url = 'https://api.upbit.com/v1/accounts'

    try:
        print("🌐 계정 정보 요청 중...")
        response = requests.get(url, headers=headers, timeout=10)

        print(f"📊 응답 코드: {response.status_code}")

        if response.status_code == 200:
            accounts = response.json()
            print(f"✅ 계정 정보 조회 성공: {len(accounts)}개 계정")

            total_krw = 0
            for account in accounts:
                if account['currency'] == 'KRW':
                    balance = float(account['balance'])
                    total_krw += balance
                    print(f"💰 KRW 잔고: {balance:,.0f}원")
                else:
                    balance = float(account['balance'])
                    if balance > 0:
                        print(f"🪙 {account['currency']}: {balance}")

            return True

        else:
            print(f"❌ API 요청 실패: {response.status_code}")
            print(f"📄 응답 내용: {response.text}")
            return False

    except Exception as e:
        print(f"❌ API 요청 오류: {e}")
        return False

def check_api_keys():
    """API 키 유효성 확인"""
    access_key = os.getenv('UPBIT_ACCESS_KEY')
    secret_key = os.getenv('UPBIT_SECRET_KEY')

    # 키 길이 확인 (업비트 API 키는 보통 32자)
    if len(access_key) != 32:
        print(f"⚠️  액세스 키 길이 이상: {len(access_key)}자 (예상: 32자)")

    if len(secret_key) != 32:
        print(f"⚠️  시크릿 키 길이 이상: {len(secret_key)}자 (예상: 32자)")

    # 키 형식 확인 (영숫자만 포함해야 함)
    import re
    if not re.match(r'^[A-Za-z0-9]+$', access_key):
        print("⚠️  액세스 키에 비정상적 문자 포함")

    if not re.match(r'^[A-Za-z0-9]+$', secret_key):
        print("⚠️  시크릿 키에 비정상적 문자 포함")

def main():
    """메인 함수"""
    print("🔍 업비트 API 인증 진단 시작")
    print("=" * 50)

    # 1. API 키 확인
    print("\n📋 1단계: API 키 유효성 확인")
    check_api_keys()

    # 2. JWT 토큰 생성 테스트
    print("\n🔐 2단계: JWT 토큰 생성 테스트")
    token = test_jwt_generation()

    if not token:
        print("❌ JWT 토큰 생성 실패로 테스트 중단")
        return

    # 3. API 요청 테스트
    print("\n🌐 3단계: API 요청 테스트")
    success = test_api_request(token)

    if success:
        print("\n✅ 모든 테스트 통과! API 연결 정상")
    else:
        print("\n❌ API 연결 실패. 키를 다시 확인해주세요")

if __name__ == "__main__":
    main()