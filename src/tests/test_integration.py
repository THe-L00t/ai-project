"""
통합 테스트 스위트
전체 시스템의 통합 테스트를 수행합니다.
"""

import unittest
import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exchange.UpbitAPI import UpbitAPI
from src.data.MarketDataCollector import MarketDataCollector
from src.data.DataStorage import DataStorage
from src.models.EnsemblePredictor import EnsemblePredictor
from src.trading.LiveTradingExecutor import LiveTradingExecutor
from src.trading.PositionManager import PositionManager, Position
from src.trading.RiskManager import RiskManager
from src.backtesting.BacktestEngine import BacktestEngine


class TestSystemIntegration(unittest.TestCase):
    """시스템 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        # 테스트용 가짜 API 키
        self.test_access_key = "test_access_key"
        self.test_secret_key = "test_secret_key"

        # 컴포넌트 초기화
        self.upbit = UpbitAPI(self.test_access_key, self.test_secret_key)
        self.data_collector = MarketDataCollector()
        self.storage = DataStorage()
        self.predictor = EnsemblePredictor()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()

        # 테스트 데이터 생성
        self.test_symbol = 'KRW-BTC'
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> pd.DataFrame:
        """테스트용 시세 데이터 생성"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        n_points = len(dates)

        # 가상의 비트코인 가격 데이터
        base_price = 50000000  # 5천만원
        price_changes = np.random.normal(0, 0.02, n_points)  # 2% 변동성
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # 최소 50% 하락 제한

        volumes = np.random.uniform(100, 1000, n_points)

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'close': prices,
            'volume': volumes
        })

    def test_data_flow_integration(self):
        """데이터 수집 → 저장 → 조회 플로우 테스트"""
        print("\n=== 데이터 플로우 통합 테스트 ===")

        # 1. 데이터 저장
        for _, row in self.test_data.iterrows():
            self.storage.save_market_data(
                symbol=self.test_symbol,
                timestamp=row['timestamp'],
                open_price=row['open'],
                high_price=row['high'],
                low_price=row['low'],
                close_price=row['close'],
                volume=row['volume']
            )

        # 2. 데이터 조회
        start_time = self.test_data['timestamp'].iloc[0]
        end_time = self.test_data['timestamp'].iloc[-1]

        retrieved_data = self.storage.get_market_data(
            symbol=self.test_symbol,
            start_time=start_time,
            end_time=end_time
        )

        # 3. 검증
        self.assertGreater(len(retrieved_data), 0, "저장된 데이터가 조회되어야 함")
        print(f"✅ 데이터 {len(retrieved_data)}개 저장/조회 성공")

    def test_prediction_pipeline(self):
        """예측 파이프라인 통합 테스트"""
        print("\n=== AI 예측 파이프라인 테스트 ===")

        # 1. 예측 수행
        prediction = self.predictor.predict(self.test_data)

        # 2. 검증
        self.assertIn('signal', prediction, "예측 결과에 신호가 포함되어야 함")
        self.assertIn('confidence', prediction, "예측 결과에 신뢰도가 포함되어야 함")
        self.assertIn(prediction['signal'],
                      ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'],
                      "유효한 거래 신호여야 함")

        print(f"✅ 예측 완료 - 신호: {prediction['signal']}, 신뢰도: {prediction['confidence']:.2f}")

    def test_position_management(self):
        """포지션 관리 시스템 테스트"""
        print("\n=== 포지션 관리 시스템 테스트 ===")

        # 1. 포지션 추가
        current_prices = {self.test_symbol: 50000000}  # 5천만원

        can_add = self.position_manager.can_add_position(self.test_symbol, current_prices)
        self.assertTrue(can_add, "첫 포지션은 추가 가능해야 함")

        position = Position(
            symbol=self.test_symbol,
            entry_price=50000000,
            quantity=0.001,
            entry_time=datetime.now(),
            stop_loss=47500000,  # 5% 손절
            take_profit=55000000,  # 10% 익절
            signal='BUY'
        )

        success = self.position_manager.add_position(position)
        self.assertTrue(success, "포지션 추가가 성공해야 함")

        # 2. 포지션 상태 확인
        summary = self.position_manager.get_position_summary(current_prices)
        self.assertEqual(summary['total_positions'], 1, "포지션 수가 1이어야 함")

        print(f"✅ 포지션 관리 테스트 완료 - 총 포지션: {summary['total_positions']}")

    def test_risk_management(self):
        """리스크 관리 시스템 테스트"""
        print("\n=== 리스크 관리 시스템 테스트 ===")

        # 1. 거래 리스크 평가
        risk_metrics = self.risk_manager.assess_trade_risk(
            symbol=self.test_symbol,
            signal_strength=0.8,
            position_size=1000000,  # 100만원
            current_price=50000000,
            market_data=self.test_data
        )

        self.assertIsNotNone(risk_metrics, "리스크 메트릭이 생성되어야 함")
        self.assertIn(risk_metrics.level.value, ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                      "유효한 리스크 레벨이어야 함")

        # 2. 포지션 사이즈 계산
        position_size = self.risk_manager.calculate_position_size(
            signal_strength=0.8,
            available_balance=10000000,  # 1천만원
            volatility=0.2
        )

        self.assertGreater(position_size, 0, "포지션 크기가 0보다 커야 함")
        self.assertLessEqual(position_size, 3000000, "포지션 크기가 30% 이하여야 함")

        print(f"✅ 리스크 관리 테스트 완료 - 레벨: {risk_metrics.level.value}, 포지션 크기: {position_size:,.0f}원")

    @patch('src.exchange.UpbitAPI.get_current_price')
    @patch('src.exchange.UpbitAPI.get_balance')
    def test_trading_executor_simulation(self, mock_get_balance, mock_get_current_price):
        """거래 실행기 시뮬레이션 테스트"""
        print("\n=== 거래 실행기 시뮬레이션 테스트 ===")

        # Mock 설정
        mock_get_current_price.return_value = 50000000  # 5천만원
        mock_get_balance.return_value = 1000000  # 100만원 잔고

        # 거래 실행기 생성
        executor = LiveTradingExecutor(
            access_key=self.test_access_key,
            secret_key=self.test_secret_key,
            initial_balance=1000000
        )

        # 상태 확인
        status = executor.get_status()
        self.assertFalse(status['is_running'], "초기에는 거래가 중지된 상태여야 함")
        self.assertEqual(status['positions'], 0, "초기 포지션 수는 0이어야 함")

        print(f"✅ 거래 실행기 시뮬레이션 완료 - 상태: {status}")

    def test_backtest_integration(self):
        """백테스팅 시스템 통합 테스트"""
        print("\n=== 백테스팅 시스템 통합 테스트 ===")

        # 백테스트 엔진 생성
        backtest_engine = BacktestEngine(initial_capital=10000000)  # 1천만원

        # 간단한 전략으로 백테스트 실행
        results = backtest_engine.run_backtest(
            data=self.test_data,
            strategy_name='test_strategy'
        )

        # 결과 검증
        self.assertIsNotNone(results, "백테스트 결과가 생성되어야 함")

        if 'total_return' in results:
            print(f"✅ 백테스트 완료 - 총 수익률: {results.get('total_return', 0):.2f}%")
        else:
            print("✅ 백테스트 시스템 기본 동작 확인 완료")

    def test_system_error_handling(self):
        """시스템 오류 처리 테스트"""
        print("\n=== 시스템 오류 처리 테스트 ===")

        # 1. 잘못된 데이터 처리
        invalid_data = pd.DataFrame()  # 빈 데이터프레임

        try:
            prediction = self.predictor.predict(invalid_data)
            print("✅ 빈 데이터에 대한 예측 처리 완료")
        except Exception as e:
            print(f"✅ 예상된 오류 처리: {type(e).__name__}")

        # 2. 서킷 브레이커 테스트
        self.risk_manager.daily_start_balance = 10000000  # 1천만원
        circuit_breaker = self.risk_manager.check_circuit_breaker(8000000)  # 20% 손실

        self.assertTrue(circuit_breaker, "20% 손실시 서킷 브레이커가 작동해야 함")
        print("✅ 서킷 브레이커 테스트 완료")

    def test_performance_benchmarks(self):
        """성능 벤치마크 테스트"""
        print("\n=== 성능 벤치마크 테스트 ===")

        # 1. 예측 성능 테스트
        start_time = time.time()
        for _ in range(10):
            self.predictor.predict(self.test_data.sample(min(100, len(self.test_data))))
        prediction_time = (time.time() - start_time) / 10

        self.assertLess(prediction_time, 5.0, "예측 시간이 5초를 초과하면 안됨")
        print(f"✅ 예측 성능: 평균 {prediction_time:.2f}초")

        # 2. 데이터 저장/조회 성능
        start_time = time.time()
        for i in range(100):
            self.storage.save_market_data(
                symbol='TEST',
                timestamp=datetime.now(),
                open_price=50000000,
                high_price=51000000,
                low_price=49000000,
                close_price=50500000,
                volume=100
            )
        storage_time = (time.time() - start_time) / 100

        self.assertLess(storage_time, 0.1, "데이터 저장 시간이 0.1초를 초과하면 안됨")
        print(f"✅ 데이터 저장 성능: 평균 {storage_time:.4f}초")

    def test_end_to_end_workflow(self):
        """종단간 워크플로우 테스트"""
        print("\n=== 종단간 워크플로우 테스트 ===")

        # 1. 데이터 준비
        self.storage.save_market_data(
            symbol=self.test_symbol,
            timestamp=datetime.now(),
            open_price=50000000,
            high_price=51000000,
            low_price=49000000,
            close_price=50500000,
            volume=100
        )

        # 2. 데이터 조회
        recent_data = self.storage.get_market_data(
            symbol=self.test_symbol,
            start_time=datetime.now() - timedelta(hours=24),
            end_time=datetime.now()
        )

        # 3. 예측 수행 (데이터가 충분할 때만)
        if len(recent_data) >= 50:
            df = pd.DataFrame(recent_data)
            prediction = self.predictor.predict(df)

            # 4. 리스크 평가
            risk_metrics = self.risk_manager.assess_trade_risk(
                symbol=self.test_symbol,
                signal_strength=prediction.get('confidence', 0.5),
                position_size=1000000,
                current_price=50500000,
                market_data=df
            )

            print(f"✅ 종단간 워크플로우 완료 - 신호: {prediction['signal']}, 리스크: {risk_metrics.level.value}")
        else:
            print("✅ 데이터 부족으로 기본 워크플로우만 테스트 완료")

    def tearDown(self):
        """테스트 정리"""
        # 테스트 데이터 정리 (필요한 경우)
        pass


def run_integration_tests():
    """통합 테스트 실행"""
    print("🚀 CoinTradingAI 통합 테스트 시작")
    print("=" * 50)

    # 테스트 스위트 생성
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIntegration)

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print(f"총 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")

    if result.failures:
        print("\n❌ 실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\n🚨 오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    if result.wasSuccessful():
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)}개 테스트가 실패했습니다.")
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)