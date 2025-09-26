#!/usr/bin/env python3
"""
CoinTradingAI - 실제 매매 모드
학습된 AI로 실제 거래 진행

매매 내용:
- 학습된 모델 기반 실시간 거래
- 10초 사이클 단타 매매
- 리스크 관리 및 손익 제어
- 포지션 모니터링
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from smart_hybrid_ai import SmartHybridAI

def main():
    """실제 매매 모드 메인 실행"""
    print("💰 CoinTradingAI 실제 매매 모드 시작")
    print("=" * 50)
    print("⚡ 단타 매매 설정:")
    print("  🕐 사이클: 10초 간격")
    print("  📊 신호: AI 통합 분석")
    print("  💹 매매: 실시간 자동 실행")
    print("  🛡️ 리스크: 손절/익절 자동")
    print("  💰 실제 매매: ✅ (진행함)")
    print("=" * 50)

    # 경고 메시지
    print("⚠️  주의사항:")
    print("  - 실제 자금으로 거래가 진행됩니다")
    print("  - 손실 가능성이 있습니다")
    print("  - 시작 전 설정을 다시 확인하세요")
    print()

    # 사용자 확인
    confirm = input("실제 매매를 시작하시겠습니까? (y/N): ")
    if confirm.lower() not in ['y', 'yes']:
        print("🛑 매매 모드 취소됨")
        return

    try:
        # 실제 매매 모드 실행
        trading_ai = SmartHybridAI()
        # live 모드로 강제 설정
        trading_ai.trading_mode = 'live'
        print(f"🚀 매매 모드: {trading_ai.trading_mode}")
        trading_ai.run_smart_cycle()

    except KeyboardInterrupt:
        print("\n🛑 매매 모드 중단")
    except Exception as e:
        print(f"❌ 매매 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 매매 모드 종료")

if __name__ == "__main__":
    main()