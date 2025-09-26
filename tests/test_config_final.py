#!/usr/bin/env python3
"""
최종 설정 로드 테스트
AI_SETTINGS.md 파일의 정확한 값이 읽히는지 확인
"""

from config_loader import get_config

def test_config():
    """설정 로드 테스트"""
    config = get_config()

    print("📋 AI_SETTINGS.md에서 로드된 설정값:")
    print("=" * 50)

    # 중요한 설정들 확인
    settings = [
        'MAX_POSITION_SIZE',
        'STOP_LOSS_PERCENTAGE',
        'TAKE_PROFIT_PERCENTAGE',
        'BUY_THRESHOLD_CHANGE',
        'SELL_THRESHOLD_CHANGE'
    ]

    for key in settings:
        value = config.get(key, '설정없음')
        print(f"{key}: {value}")

    print("\n예상값 (AI_SETTINGS.md):")
    print("MAX_POSITION_SIZE: 0.15")
    print("STOP_LOSS_PERCENTAGE: 0.7")
    print("TAKE_PROFIT_PERCENTAGE: 1.3")
    print("BUY_THRESHOLD_CHANGE: 3.0")
    print("SELL_THRESHOLD_CHANGE: -1.0")

if __name__ == "__main__":
    test_config()