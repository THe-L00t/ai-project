#!/usr/bin/env python3
"""
CoinTradingAI - 학습 전용 모드
매매 없이 순수 학습만 진행

학습 내용:
- 시장 데이터 수집 및 분석
- 패턴 학습 및 모델 훈련
- 뉴스 감정 분석 학습
- 강화학습 모델 업데이트
- 백테스팅 및 성능 검증
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from smart_hybrid_ai import SmartHybridAI

class LearningOnlyMode(SmartHybridAI):
    """학습 전용 모드 - 매매 없이 학습만"""

    def __init__(self):
        super().__init__()
        # 학습 전용 모드로 설정
        self.trading_mode = 'learning_only'
        print("📚 학습 전용 모드로 초기화됨")
        print("💡 실제 매매는 진행하지 않고 학습만 수행합니다")

    def execute_smart_trade(self, market, signal, confidence, reasons, current_price):
        """학습 모드에서는 실제 거래 없이 로그만"""
        coin = market.split('-')[1]

        # 매매 시뮬레이션만 진행
        if signal == 'BUY' and market not in self.positions:
            logger.info(f"📚 [학습] {market} 매수 신호 감지 (신뢰도: {confidence:.2f})")
            for reason in reasons:
                logger.info(f"   💡 이유: {reason}")

            # 가상 포지션 생성 (학습용)
            self.positions[market] = {
                'type': 'long',
                'quantity': 0.001,  # 가상 수량
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'conditions': {'confidence': confidence, 'reasons': reasons},
                'context': {'sentiment': self.get_current_sentiment(coin)},
                'is_simulation': True
            }

        elif signal == 'SELL' and market in self.positions:
            position = self.positions[market]
            profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100

            logger.info(f"📚 [학습] {market} 매도 신호 감지 (수익률: {profit_pct:+.2f}%)")
            for reason in reasons:
                logger.info(f"   💡 이유: {reason}")

            # 강화학습에 결과 기록
            entry_data_for_learning = {
                'timestamp': self.get_position_entry_time(position),
                'price': position['entry_price'],
                'conditions': position.get('conditions', {})
            }
            self.reinforcement_learner.record_trade_result(
                coin, entry_data_for_learning, {'price': current_price}, profit_pct
            )

            del self.positions[market]

        return True  # 학습 모드에서는 항상 성공

def main():
    """학습 모드 메인 실행"""
    print("📚 CoinTradingAI 학습 전용 모드 시작")
    print("=" * 50)
    print("🔍 수행 작업:")
    print("  📊 시장 데이터 수집 및 분석")
    print("  🧠 AI 모델 학습 및 훈련")
    print("  📰 뉴스 감정 분석 학습")
    print("  🎯 강화학습 모델 업데이트")
    print("  📈 백테스팅 및 성능 검증")
    print("  💰 실제 매매: ❌ (진행하지 않음)")
    print("=" * 50)

    try:
        # 학습 전용 모드 실행
        learning_ai = LearningOnlyMode()
        learning_ai.run_smart_cycle()

    except KeyboardInterrupt:
        print("\n🛑 학습 모드 중단")
    except Exception as e:
        print(f"❌ 학습 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 학습 모드 종료")

if __name__ == "__main__":
    main()