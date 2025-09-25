"""
실시간 거래 실행기
실제 업비트에서 AI 예측 기반 자동매매를 수행합니다.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ..exchange.UpbitAPI import UpbitAPI
from ..models.EnsemblePredictor import EnsemblePredictor
from ..data.MarketDataCollector import MarketDataCollector
from ..data.DataStorage import DataStorage


class LiveTradingExecutor:
    """실시간 거래 실행 시스템"""

    def __init__(self, access_key: str, secret_key: str,
                 initial_balance: float = 1000000,
                 max_position_ratio: float = 0.3,
                 stop_loss_ratio: float = 0.05,
                 take_profit_ratio: float = 0.1):
        """
        초기화

        Args:
            access_key: 업비트 ACCESS KEY
            secret_key: 업비트 SECRET KEY
            initial_balance: 초기 자본금 (원)
            max_position_ratio: 최대 포지션 비율 (0.3 = 30%)
            stop_loss_ratio: 손절 비율 (0.05 = 5%)
            take_profit_ratio: 익절 비율 (0.1 = 10%)
        """
        # API 초기화
        self.upbit = UpbitAPI(access_key, secret_key)

        # 거래 설정
        self.initial_balance = initial_balance
        self.max_position_ratio = max_position_ratio
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio

        # 컴포넌트 초기화
        self.predictor = EnsemblePredictor()
        self.data_collector = MarketDataCollector()
        self.storage = DataStorage()

        # 상태 관리
        self.is_running = False
        self.positions = {}  # {symbol: position_info}
        self.last_prices = {}
        self.trade_history = []

        # 안전장치
        self.emergency_stop = False
        self.daily_loss_limit = 0.1  # 일일 최대 손실 10%
        self.max_trades_per_day = 20
        self.trades_today = 0
        self.daily_start_balance = initial_balance

        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def start_trading(self, symbols: List[str] = None):
        """실시간 거래 시작"""
        if symbols is None:
            symbols = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']

        self.logger.info(f"실시간 거래 시작 - 대상 코인: {symbols}")
        self.is_running = True

        # 초기 잔고 확인
        await self._check_balance()

        # 기존 포지션 확인
        await self._load_existing_positions(symbols)

        try:
            # 메인 거래 루프
            while self.is_running and not self.emergency_stop:
                await self._trading_cycle(symbols)
                await asyncio.sleep(60)  # 1분 주기

        except Exception as e:
            self.logger.error(f"거래 중 오류 발생: {e}")
            await self._emergency_stop_all()

        self.logger.info("실시간 거래 종료")

    async def _trading_cycle(self, symbols: List[str]):
        """거래 사이클 실행"""
        try:
            # 1. 안전장치 체크
            if not await self._safety_check():
                return

            # 2. 각 코인별 거래 신호 확인
            for symbol in symbols:
                if self.emergency_stop:
                    break

                await self._process_symbol(symbol)
                await asyncio.sleep(1)  # API 제한 고려

            # 3. 기존 포지션 관리
            await self._manage_positions()

        except Exception as e:
            self.logger.error(f"거래 사이클 오류: {e}")

    async def _process_symbol(self, symbol: str):
        """개별 코인 처리"""
        try:
            # 현재 시세 조회
            current_price = await self.upbit.get_current_price(symbol)
            if not current_price:
                return

            self.last_prices[symbol] = current_price

            # AI 예측 수행
            prediction = await self._get_prediction(symbol)
            if not prediction:
                return

            signal = prediction['signal']
            confidence = prediction['confidence']

            # 신뢰도 필터링
            if confidence < 0.7:
                return

            # 거래 실행 결정
            if signal in ['STRONG_BUY', 'BUY'] and symbol not in self.positions:
                await self._execute_buy(symbol, current_price, signal)

            elif signal in ['STRONG_SELL', 'SELL'] and symbol in self.positions:
                await self._execute_sell(symbol, current_price, signal)

        except Exception as e:
            self.logger.error(f"{symbol} 처리 중 오류: {e}")

    async def _get_prediction(self, symbol: str) -> Optional[Dict]:
        """AI 예측 수행"""
        try:
            # 최근 데이터 조회
            data = self.storage.get_market_data(
                symbol=symbol,
                start_time=datetime.now() - timedelta(hours=24),
                end_time=datetime.now()
            )

            if len(data) < 100:  # 충분한 데이터 필요
                return None

            # 예측 수행
            df = pd.DataFrame(data)
            prediction = self.predictor.predict(df)

            return {
                'signal': prediction['signal'],
                'confidence': prediction['confidence'],
                'price_prediction': prediction.get('price_prediction', 0),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"{symbol} 예측 중 오류: {e}")
            return None

    async def _execute_buy(self, symbol: str, price: float, signal: str):
        """매수 실행"""
        try:
            # 매수 가능 금액 계산
            balance = await self.upbit.get_balance('KRW')
            if balance < 5000:  # 최소 거래 금액
                return

            # 포지션 크기 결정
            position_size = min(
                balance * self.max_position_ratio,
                balance * (0.5 if signal == 'STRONG_BUY' else 0.3)
            )

            if position_size < 5000:
                return

            # 매수 주문
            result = await self.upbit.buy_market_order(symbol, position_size)

            if result and result.get('uuid'):
                # 포지션 기록
                self.positions[symbol] = {
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'quantity': position_size / price,
                    'signal': signal,
                    'stop_loss': price * (1 - self.stop_loss_ratio),
                    'take_profit': price * (1 + self.take_profit_ratio),
                    'order_uuid': result['uuid']
                }

                self.trades_today += 1
                self.logger.info(f"{symbol} 매수 완료 - 가격: {price:,.0f}, 금액: {position_size:,.0f}")

        except Exception as e:
            self.logger.error(f"{symbol} 매수 중 오류: {e}")

    async def _execute_sell(self, symbol: str, price: float, signal: str):
        """매도 실행"""
        try:
            if symbol not in self.positions:
                return

            position = self.positions[symbol]

            # 보유 수량 확인
            balance = await self.upbit.get_balance(symbol.split('-')[1])
            if balance <= 0:
                del self.positions[symbol]
                return

            # 매도 주문
            result = await self.upbit.sell_market_order(symbol, balance)

            if result and result.get('uuid'):
                # 손익 계산
                entry_price = position['entry_price']
                profit_rate = (price - entry_price) / entry_price * 100

                # 거래 기록
                trade_record = {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'profit_rate': profit_rate,
                    'signal': signal,
                    'reason': 'AI_SIGNAL'
                }

                self.trade_history.append(trade_record)
                del self.positions[symbol]
                self.trades_today += 1

                self.logger.info(f"{symbol} 매도 완료 - 수익률: {profit_rate:.2f}%")

        except Exception as e:
            self.logger.error(f"{symbol} 매도 중 오류: {e}")

    async def _manage_positions(self):
        """기존 포지션 관리 (손절/익절)"""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.last_prices.get(symbol)
                if not current_price:
                    continue

                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']

                # 손절 체크
                if current_price <= stop_loss:
                    await self._execute_sell(symbol, current_price, 'STOP_LOSS')
                    self.logger.warning(f"{symbol} 손절 실행 - 가격: {current_price:,.0f}")

                # 익절 체크
                elif current_price >= take_profit:
                    await self._execute_sell(symbol, current_price, 'TAKE_PROFIT')
                    self.logger.info(f"{symbol} 익절 실행 - 가격: {current_price:,.0f}")

                # 오래된 포지션 체크 (24시간 초과)
                elif datetime.now() - position['entry_time'] > timedelta(hours=24):
                    await self._execute_sell(symbol, current_price, 'TIME_EXIT')
                    self.logger.info(f"{symbol} 시간 만료 매도 - 가격: {current_price:,.0f}")

            except Exception as e:
                self.logger.error(f"{symbol} 포지션 관리 중 오류: {e}")

    async def _safety_check(self) -> bool:
        """안전장치 체크"""
        try:
            # 일일 거래 횟수 제한
            if self.trades_today >= self.max_trades_per_day:
                self.logger.warning("일일 거래 횟수 한도 도달")
                return False

            # 일일 손실 한도 체크
            current_balance = await self.upbit.get_balance('KRW')
            total_balance = current_balance

            # 보유 포지션 가치 추가
            for symbol, position in self.positions.items():
                current_price = self.last_prices.get(symbol, position['entry_price'])
                position_value = position['quantity'] * current_price
                total_balance += position_value

            daily_loss_rate = (self.daily_start_balance - total_balance) / self.daily_start_balance

            if daily_loss_rate > self.daily_loss_limit:
                self.logger.error(f"일일 손실 한도 초과: {daily_loss_rate:.2%}")
                await self._emergency_stop_all()
                return False

            return True

        except Exception as e:
            self.logger.error(f"안전장치 체크 중 오류: {e}")
            return False

    async def _check_balance(self):
        """잔고 확인"""
        try:
            balance = await self.upbit.get_balance('KRW')
            self.logger.info(f"현재 KRW 잔고: {balance:,.0f}원")

        except Exception as e:
            self.logger.error(f"잔고 조회 중 오류: {e}")

    async def _load_existing_positions(self, symbols: List[str]):
        """기존 포지션 로드"""
        try:
            for symbol in symbols:
                coin = symbol.split('-')[1]
                balance = await self.upbit.get_balance(coin)

                if balance > 0:
                    # 기존 포지션이 있는 경우 (단순히 현재가로 설정)
                    current_price = await self.upbit.get_current_price(symbol)

                    self.positions[symbol] = {
                        'entry_price': current_price,  # 실제 매수가는 알 수 없음
                        'entry_time': datetime.now(),
                        'quantity': balance,
                        'signal': 'EXISTING',
                        'stop_loss': current_price * (1 - self.stop_loss_ratio),
                        'take_profit': current_price * (1 + self.take_profit_ratio),
                        'order_uuid': None
                    }

                    self.logger.info(f"기존 포지션 발견: {symbol} - {balance:.8f}")

        except Exception as e:
            self.logger.error(f"기존 포지션 로드 중 오류: {e}")

    async def _emergency_stop_all(self):
        """비상 정지 - 모든 포지션 청산"""
        self.emergency_stop = True
        self.logger.error("비상 정지 활성화 - 모든 포지션 청산")

        try:
            for symbol in list(self.positions.keys()):
                current_price = self.last_prices.get(symbol)
                if current_price:
                    await self._execute_sell(symbol, current_price, 'EMERGENCY_STOP')

        except Exception as e:
            self.logger.error(f"비상 정지 중 오류: {e}")

    def stop_trading(self):
        """거래 중지"""
        self.is_running = False
        self.logger.info("거래 중지 요청됨")

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'positions': len(self.positions),
            'trades_today': self.trades_today,
            'position_details': self.positions,
            'trade_history': self.trade_history[-10:]  # 최근 10개 거래
        }