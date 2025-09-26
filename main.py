#!/usr/bin/env python3
"""
CoinTradingAI - 스마트 하이브리드 AI 거래 시스템
최적화된 단일 진입점

Features:
- API 429 에러 완전 해결 (배치 요청 + 캐싱)
- AI_SETTINGS.md 완전 통합
- 가중치 기반 통합 신호 생성
- timestamp 에러 완전 해결
- 60초 최적화 사이클

Usage:
    python3 main.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from smart_hybrid_ai import SmartHybridAI

def main():
    """메인 실행 함수"""
    print("🚀 CoinTradingAI 스마트 하이브리드 시스템 시작")
    print("=" * 60)
    print("✨ 최적화된 기능:")
    print("  📊 API 429 에러 완전 해결 (배치 + 캐싱)")
    print("  🧠 AI_SETTINGS.md 완전 통합")
    print("  ⚖️  가중치 기반 신호 통합")
    print("  🕐 timestamp 에러 완전 해결")
    print("  ⚡ 60초 최적화 사이클")
    print("=" * 60)

    try:
        # SmartHybridAI 인스턴스 생성 및 실행
        ai_trader = SmartHybridAI()
        ai_trader.run_smart_cycle()

    except KeyboardInterrupt:
        print("\n🛑 사용자에 의한 시스템 중단")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 CoinTradingAI 시스템 종료")

if __name__ == "__main__":
    main()