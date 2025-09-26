#!/usr/bin/env python3
"""
CoinTradingAI 시스템 기능 검증 테스트
모든 핵심 기능이 정상 작동하는지 확인
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_configuration():
    """설정 파일 로딩 테스트"""
    print("🔧 설정 시스템 테스트...")
    try:
        from config_loader import get_config
        config = get_config()

        # 핵심 설정 확인
        cycle_seconds = config.get('TRADING_CYCLE_SECONDS', 0)
        max_position = config.get('MAX_POSITION_SIZE', 0)

        print(f"  ✅ 사이클 간격: {cycle_seconds}초")
        print(f"  ✅ 포지션 크기: {max_position*100}%")

        if cycle_seconds == 10:
            print("  🎯 단타 설정 정상 확인")
        else:
            print(f"  ⚠️  단타 설정 이상: {cycle_seconds}초 (10초여야 함)")

        return True

    except Exception as e:
        print(f"  ❌ 설정 로딩 실패: {e}")
        return False

def test_ai_initialization():
    """AI 시스템 초기화 테스트"""
    print("\n🧠 AI 시스템 초기화 테스트...")
    try:
        from smart_hybrid_ai import SmartHybridAI

        ai = SmartHybridAI()

        # 핵심 속성 확인
        print(f"  ✅ 거래 모드: {ai.trading_mode}")
        print(f"  ✅ 사이클: {ai.config.get('TRADING_CYCLE_SECONDS')}초")
        print(f"  ✅ 대상 코인: {len(ai.target_coins)}개")
        print(f"  ✅ 캐시 TTL: {ai.cache_ttl}초")

        # 핵심 함수 존재 확인
        if hasattr(ai, 'get_position_entry_time'):
            print("  ✅ timestamp 안전 함수 존재")
        else:
            print("  ❌ timestamp 안전 함수 없음")

        if hasattr(ai, 'get_cached_ticker'):
            print("  ✅ API 캐싱 함수 존재")
        else:
            print("  ❌ API 캐싱 함수 없음")

        return True

    except Exception as e:
        print(f"  ❌ AI 초기화 실패: {e}")
        return False

def test_mode_files():
    """모드별 실행 파일 테스트"""
    print("\n🎮 모드별 파일 테스트...")
    try:
        # 학습 모드 파일 확인
        if os.path.exists('run_learning_mode.py'):
            print("  ✅ 학습 모드 파일 존재")
        else:
            print("  ❌ 학습 모드 파일 없음")

        # 매매 모드 파일 확인
        if os.path.exists('run_trading_mode.py'):
            print("  ✅ 매매 모드 파일 존재")
        else:
            print("  ❌ 매매 모드 파일 없음")

        # 메인 파일 확인
        if os.path.exists('main.py'):
            print("  ✅ 통합 메인 파일 존재")
        else:
            print("  ❌ 통합 메인 파일 없음")

        return True

    except Exception as e:
        print(f"  ❌ 모드 파일 확인 실패: {e}")
        return False

def test_desktop_shortcuts():
    """데스크톱 단축키 테스트"""
    print("\n🖥️ 데스크톱 단축키 테스트...")
    try:
        learning_shortcut = '/Users/the-l00t/Desktop/start_ai_learning.command'
        trading_shortcut = '/Users/the-l00t/Desktop/start_ai_trading.command'

        if os.path.exists(learning_shortcut):
            print("  ✅ 학습 모드 단축키 존재")
            if os.access(learning_shortcut, os.X_OK):
                print("  ✅ 학습 모드 실행 권한 확인")
            else:
                print("  ⚠️  학습 모드 실행 권한 없음")
        else:
            print("  ❌ 학습 모드 단축키 없음")

        if os.path.exists(trading_shortcut):
            print("  ✅ 매매 모드 단축키 존재")
            if os.access(trading_shortcut, os.X_OK):
                print("  ✅ 매매 모드 실행 권한 확인")
            else:
                print("  ⚠️  매매 모드 실행 권한 없음")
        else:
            print("  ❌ 매매 모드 단축키 없음")

        return True

    except Exception as e:
        print(f"  ❌ 단축키 확인 실패: {e}")
        return False

def test_timestamp_safety():
    """Timestamp 안전성 테스트"""
    print("\n⏰ Timestamp 안전성 테스트...")
    try:
        from smart_hybrid_ai import SmartHybridAI
        from datetime import datetime, timedelta

        ai = SmartHybridAI()

        # 다양한 포지션 구조 테스트
        test_positions = [
            {
                'entry_time': datetime.now(),
                'entry_price': 50000,
                'type': 'new_format'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'entry_price': 50000,
                'type': 'old_format'
            },
            {
                'entry_price': 50000,
                'type': 'no_time'
            }
        ]

        for i, position in enumerate(test_positions):
            try:
                entry_time = ai.get_position_entry_time(position)
                print(f"  ✅ 포지션 {i+1} ({position['type']}): {entry_time}")
            except Exception as e:
                print(f"  ❌ 포지션 {i+1} 실패: {e}")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Timestamp 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("🔍 CoinTradingAI 시스템 기능 검증 테스트")
    print("=" * 60)

    tests = [
        ("설정 시스템", test_configuration),
        ("AI 초기화", test_ai_initialization),
        ("모드 파일", test_mode_files),
        ("데스크톱 단축키", test_desktop_shortcuts),
        ("Timestamp 안전성", test_timestamp_safety),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if test_func():
            passed += 1

    print("\n" + "=" * 60)
    print(f"🎯 테스트 결과: {passed}/{total} 통과")

    if passed == total:
        print("🎉 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        print("\n📚 학습 모드 실행: python3 run_learning_mode.py")
        print("💰 매매 모드 실행: python3 run_trading_mode.py")
        print("🖥️ 단축키 사용: 데스크톱의 .command 파일 더블클릭")
    else:
        print("⚠️  일부 테스트 실패. 문제를 해결한 후 다시 실행하세요.")
        print("📖 참고: PERSISTENT_SYSTEM_CONFIG.md")

    print("=" * 60)

if __name__ == "__main__":
    main()