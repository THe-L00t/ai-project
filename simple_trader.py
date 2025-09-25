#!/usr/bin/env python3
"""
간단한 업비트 자동매매 트레이더
- 기본적인 매매 로직으로 실행
- Python 3.13 호환
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
        logging.FileHandler('logs/simple_trader.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleTradingBot:
    """간단한 자동매매 봇"""

    def __init__(self):
        load_dotenv()

        # 설정값
        self.trading_mode = os.getenv('TRADING_MODE', 'paper')
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10%
        self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', '5.0'))
        self.take_profit_percentage = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '10.0'))

        # 업비트 API 초기화
        self.upbit = UpbitAPI()

        # 거래 대상 코인
        self.target_coins = ['KRW-BTC', 'KRW-ETH']

        # 거래 기록
        self.positions = {}
        self.last_prices = {}

        logger.info(f"🤖 SimpleTradingBot 초기화 완료 (모드: {self.trading_mode})")

    def get_signal(self, market):
        """간단한 매매 신호 생성 (데모용)"""
        try:
            # 현재가 조회
            ticker = self.upbit.GetTicker([market])
            if not ticker:
                return 'HOLD', 0

            current_price = ticker[0].trade_price
            change_rate = ticker[0].change_rate * 100

            # 이전 가격 기록
            if market in self.last_prices:
                prev_price = self.last_prices[market]
                price_momentum = (current_price - prev_price) / prev_price * 100
            else:
                price_momentum = 0

            self.last_prices[market] = current_price

            # 간단한 모멘텀 기반 신호
            if change_rate > 2 and price_momentum > 1:
                return 'BUY', current_price
            elif change_rate < -2 and price_momentum < -1:
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
                logger.info(f"📈 모의매수: {market} {quantity:.8f} @ {price:,}원 (총 {buy_amount:,.0f}원)")
                return True

        elif signal == 'SELL' and market in self.positions:
            # 매도
            position = self.positions[market]
            pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
            pnl_amount = position['quantity'] * (price - position['entry_price'])

            logger.info(f"📉 모의매도: {market} {position['quantity']:.8f} @ {price:,}원")
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
                    logger.info(f"📈 실제매수: {market} {quantity:.8f} @ {price:,}원 (UUID: {result['uuid']})")
                    return True

        elif signal == 'SELL' and market in self.positions:
            # 매도
            position = self.positions[market]
            coin_balance = self.upbit.GetCoinBalance(coin)

            if coin_balance > 0:
                result = self.upbit.SellMarket(market, coin_balance)
                if result:
                    pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                    logger.info(f"📉 실제매도: {market} {coin_balance:.8f} @ {price:,}원 (UUID: {result['uuid']})")
                    logger.info(f"   💰 예상손익: {pnl_pct:+.2f}%")
                    del self.positions[market]
                    return True

        return False

    def check_risk_management(self):
        """리스크 관리 확인"""
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

                # 손절매 체크
                if pnl_pct <= -self.stop_loss_percentage:
                    logger.warning(f"🛑 손절매 발동: {market} ({pnl_pct:.2f}%)")
                    self.execute_trade(market, 'SELL', current_price)

                # 익절매 체크
                elif pnl_pct >= self.take_profit_percentage:
                    logger.info(f"🎯 익절매 발동: {market} (+{pnl_pct:.2f}%)")
                    self.execute_trade(market, 'SELL', current_price)

            except Exception as e:
                logger.error(f"리스크 관리 오류 ({market}): {e}")

    def run(self):
        """자동매매 실행"""
        logger.info("🚀 자동매매 시작!")
        logger.info(f"거래 모드: {self.trading_mode}")
        logger.info(f"대상 코인: {self.target_coins}")
        logger.info(f"최대 포지션 크기: {self.max_position_size*100}%")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                logger.info(f"\n🔄 매매 사이클 #{cycle_count}")

                # 현재 잔고 확인
                krw_balance = self.upbit.GetKRWBalance()
                logger.info(f"💰 원화 잔고: {krw_balance:,.0f}원")

                # 각 코인별 신호 확인 및 거래
                for market in self.target_coins:
                    try:
                        signal, price = self.get_signal(market)

                        if signal != 'HOLD':
                            logger.info(f"🎯 {market}: {signal} 신호 (가격: {price:,}원)")
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

                # 30초 대기
                logger.info("⏱️  30초 대기...")
                time.sleep(30)

        except KeyboardInterrupt:
            logger.info("\n🛑 사용자에 의한 정지")
        except Exception as e:
            logger.error(f"❌ 치명적 오류: {e}")
        finally:
            logger.info("🏁 자동매매 종료")

def main():
    """메인 함수"""
    print("🤖 Simple Trading Bot")
    print("=" * 50)

    try:
        bot = SimpleTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"봇 실행 실패: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())