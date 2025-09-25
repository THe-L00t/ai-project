#!/usr/bin/env python3
"""
공격적 업비트 자동매매 트레이더
- 더 민감한 매매 조건으로 활발한 거래
- 수익률 목표 상향 조정
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from exchange.UpbitAPI import UpbitAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/aggressive_trader.log')
    ]
)
logger = logging.getLogger(__name__)

class AggressiveTradingBot:
    """공격적 자동매매 봇"""

    def __init__(self):
        load_dotenv()

        # 공격적 설정값 (기존 대비 +5%p 상향)
        self.trading_mode = os.getenv('TRADING_MODE', 'live')
        self.max_position_size = 0.15  # 15%로 증가 (기존 10% + 5%p)
        self.stop_loss_percentage = 3.0  # 3%로 완화 (더 참을성 있게)
        self.take_profit_percentage = 15.0  # 15%로 상향 (기존 10% + 5%p)

        # 더 민감한 매매 조건
        self.buy_threshold_change = 1.0  # 1%로 완화 (기존 2%)
        self.sell_threshold_change = -1.0  # -1%로 완화 (기존 -2%)
        self.momentum_threshold = 0.5  # 0.5%로 완화 (기존 1%)

        # 업비트 API 초기화
        self.upbit = UpbitAPI()

        # 거래 대상 코인 (더 다양하게)
        self.target_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']

        # 거래 기록
        self.positions = {}
        self.last_prices = {}
        self.price_history = {}  # 가격 히스토리 추가

        logger.info(f"🔥 AggressiveTradingBot 초기화 완료 (모드: {self.trading_mode})")
        logger.info(f"📈 공격적 설정: 포지션크기 {self.max_position_size*100}%, 익절 {self.take_profit_percentage}%")

    def get_signal(self, market):
        """공격적인 매매 신호 생성"""
        try:
            # 현재가 조회
            ticker = self.upbit.GetTicker([market])
            if not ticker:
                return 'HOLD', 0

            current_price = ticker[0].trade_price
            change_rate = ticker[0].change_rate * 100

            # 가격 히스토리 관리
            if market not in self.price_history:
                self.price_history[market] = []

            self.price_history[market].append(current_price)

            # 최근 5개 가격만 유지
            if len(self.price_history[market]) > 5:
                self.price_history[market] = self.price_history[market][-5:]

            # 이전 가격 기록
            if market in self.last_prices:
                prev_price = self.last_prices[market]
                price_momentum = (current_price - prev_price) / prev_price * 100
            else:
                price_momentum = 0

            self.last_prices[market] = current_price

            # 단기 추세 분석 (최근 3개 가격)
            short_term_trend = 0
            if len(self.price_history[market]) >= 3:
                recent_prices = self.price_history[market][-3:]
                if recent_prices[-1] > recent_prices[0]:
                    short_term_trend = 1  # 상승 추세
                elif recent_prices[-1] < recent_prices[0]:
                    short_term_trend = -1  # 하락 추세

            # 공격적인 매매 신호 조건 (더 민감하게)
            buy_conditions = [
                change_rate > self.buy_threshold_change,  # 1% 상승
                price_momentum > self.momentum_threshold,  # 0.5% 모멘텀
                short_term_trend >= 0  # 단기 상승 또는 횡보
            ]

            sell_conditions = [
                change_rate < self.sell_threshold_change,  # -1% 하락
                price_momentum < -self.momentum_threshold,  # -0.5% 모멘텀
                short_term_trend <= 0  # 단기 하락 또는 횡보
            ]

            # 매수 신호 (2개 이상 조건 만족)
            if sum(buy_conditions) >= 2:
                confidence = sum(buy_conditions) / len(buy_conditions)
                logger.info(f"🎯 {market} 매수 신호 발생! (신뢰도: {confidence:.2f})")
                logger.info(f"   변동률: {change_rate:+.2f}%, 모멘텀: {price_momentum:+.2f}%, 추세: {short_term_trend}")
                return 'BUY', current_price

            # 매도 신호 (2개 이상 조건 만족 && 포지션 보유 중)
            elif sum(sell_conditions) >= 2 and market in self.positions:
                confidence = sum(sell_conditions) / len(sell_conditions)
                logger.info(f"🎯 {market} 매도 신호 발생! (신뢰도: {confidence:.2f})")
                logger.info(f"   변동률: {change_rate:+.2f}%, 모멘텀: {price_momentum:+.2f}%, 추세: {short_term_trend}")
                return 'SELL', current_price

            else:
                return 'HOLD', current_price

        except Exception as e:
            logger.error(f"신호 생성 오류 ({market}): {e}")
            return 'HOLD', 0

    def execute_trade(self, market, signal, price):
        """거래 실행"""
        try:
            if self.trading_mode == 'paper':
                return self.execute_paper_trade(market, signal, price)
            else:
                return self.execute_live_trade(market, signal, price)
        except Exception as e:
            logger.error(f"거래 실행 오류: {e}")
            return False

    def execute_paper_trade(self, market, signal, price):
        """모의투자 실행"""
        coin = market.split('-')[1]

        if signal == 'BUY' and market not in self.positions:
            # 매수
            krw_balance = self.upbit.GetKRWBalance()
            buy_amount = krw_balance * self.max_position_size

            if buy_amount >= 5000:  # 최소 주문 금액
                quantity = buy_amount / price
                self.positions[market] = {
                    'type': 'long',
                    'quantity': quantity,
                    'entry_price': price,
                    'timestamp': datetime.now()
                }
                logger.info(f"🔥 [모의] 공격적 매수: {market} {quantity:.8f} @ {price:,}원 (총 {buy_amount:,.0f}원)")
                return True

        elif signal == 'SELL' and market in self.positions:
            # 매도
            position = self.positions[market]
            pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
            pnl_amount = position['quantity'] * (price - position['entry_price'])

            logger.info(f"🔥 [모의] 공격적 매도: {market} {position['quantity']:.8f} @ {price:,}원")
            logger.info(f"   💰 손익: {pnl_pct:+.2f}% ({pnl_amount:+,.0f}원)")

            del self.positions[market]
            return True

        return False

    def execute_live_trade(self, market, signal, price):
        """실거래 실행"""
        coin = market.split('-')[1]

        if signal == 'BUY' and market not in self.positions:
            # 매수
            krw_balance = self.upbit.GetKRWBalance()
            buy_amount = krw_balance * self.max_position_size

            if buy_amount >= 5000:  # 최소 주문 금액
                result = self.upbit.BuyMarket(market, buy_amount)
                if result:
                    quantity = buy_amount / price  # 근사치
                    self.positions[market] = {
                        'type': 'long',
                        'quantity': quantity,
                        'entry_price': price,
                        'timestamp': datetime.now(),
                        'order_uuid': result['uuid']
                    }
                    logger.info(f"🔥 [실거래] 공격적 매수: {market} {quantity:.8f} @ {price:,}원 (UUID: {result['uuid']})")
                    return True

        elif signal == 'SELL' and market in self.positions:
            # 매도
            position = self.positions[market]
            coin_balance = self.upbit.GetCoinBalance(coin)

            if coin_balance > 0:
                result = self.upbit.SellMarket(market, coin_balance)
                if result:
                    pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                    logger.info(f"🔥 [실거래] 공격적 매도: {market} {coin_balance:.8f} @ {price:,}원 (UUID: {result['uuid']})")
                    logger.info(f"   💰 예상손익: {pnl_pct:+.2f}%")
                    del self.positions[market]
                    return True

        return False

    def check_risk_management(self):
        """공격적 리스크 관리"""
        for market, position in list(self.positions.items()):
            try:
                ticker = self.upbit.GetTicker([market])
                if not ticker:
                    continue

                current_price = ticker[0].trade_price
                entry_price = position['entry_price']

                # 손익 계산
                if position['type'] == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                # 손절매 체크 (3%로 완화)
                if pnl_pct <= -self.stop_loss_percentage:
                    logger.warning(f"🛑 손절매 발동: {market} ({pnl_pct:.2f}%)")
                    self.execute_trade(market, 'SELL', current_price)

                # 익절매 체크 (15%로 상향)
                elif pnl_pct >= self.take_profit_percentage:
                    logger.info(f"🎯 익절매 발동: {market} (+{pnl_pct:.2f}%)")
                    self.execute_trade(market, 'SELL', current_price)

            except Exception as e:
                logger.error(f"리스크 관리 오류 ({market}): {e}")

    def run(self):
        """공격적 자동매매 실행"""
        logger.info("🔥 공격적 자동매매 시작!")
        logger.info(f"거래 모드: {self.trading_mode}")
        logger.info(f"대상 코인: {self.target_coins}")
        logger.info(f"최대 포지션 크기: {self.max_position_size*100}%")
        logger.info(f"익절 목표: {self.take_profit_percentage}%")
        logger.info(f"손절 한계: {self.stop_loss_percentage}%")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                logger.info(f"\n🔄 공격적 매매 사이클 #{cycle_count}")

                # 현재 잔고 확인
                krw_balance = self.upbit.GetKRWBalance()
                logger.info(f"💰 원화 잔고: {krw_balance:,.0f}원")

                # 각 코인별 신호 확인 및 거래
                for market in self.target_coins:
                    try:
                        signal, price = self.get_signal(market)

                        if signal != 'HOLD':
                            logger.info(f"🔥 {market}: {signal} 신호 (가격: {price:,}원)")
                            self.execute_trade(market, signal, price)
                        else:
                            logger.info(f"⏸️  {market}: HOLD (가격: {price:,}원)")

                    except Exception as e:
                        logger.error(f"{market} 처리 오류: {e}")

                # 리스크 관리
                self.check_risk_management()

                # 현재 포지션 상태 출력
                if self.positions:
                    logger.info("📊 현재 포지션:")
                    for market, pos in self.positions.items():
                        ticker = self.upbit.GetTicker([market])
                        if ticker:
                            current_price = ticker[0].trade_price
                            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                            logger.info(f"   {market}: {pos['quantity']:.8f} ({pnl_pct:+.2f}%)")
                else:
                    logger.info("📊 현재 포지션: 없음")

                # 20초 대기 (더 빠른 반응)
                logger.info("⏱️  20초 대기...")
                time.sleep(20)

        except KeyboardInterrupt:
            logger.info("\n🛑 사용자에 의한 정지")
        except Exception as e:
            logger.error(f"❌ 치명적 오류: {e}")
        finally:
            logger.info("🏁 공격적 자동매매 종료")

def main():
    """메인 함수"""
    print("🔥 Aggressive Trading Bot")
    print("=" * 50)

    try:
        bot = AggressiveTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"봇 실행 실패: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())