#!/usr/bin/env python3
"""
코인 자동매매 AI 시스템 메인 실행 파일 v2.0
- 완전한 라이브 거래 시스템
- AI 예측 기반 자동매매
- 실시간 모니터링 및 리스크 관리
- 웹 대시보드 통합
"""

import os
import sys
import logging
import time
import signal
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.stability_manager.StabilityManager import StabilityManager, SystemStatus
from src.news_analyzer.NewsCollector import NewsCollector
from src.trading.LiveTradingExecutor import LiveTradingExecutor
from src.data.MarketDataCollector import MarketDataCollector
from src.data.DataStorage import DataStorage
from src.models.EnsemblePredictor import EnsemblePredictor
from src.dashboard.NotificationSystem import NotificationSystem

class CoinTradingAI:
    """
    코인 자동매매 AI 메인 시스템 v2.0
    완전한 라이브 거래 시스템 구현
    """

    def __init__(self, trading_mode: str = 'paper'):
        """
        초기화

        Args:
            trading_mode: 'paper' (모의거래) 또는 'live' (실거래)
        """
        # 로깅 설정
        self._SetupLogging()
        self.logger = logging.getLogger(__name__)

        # 거래 모드
        self.trading_mode = trading_mode
        self.is_running = False

        # 시스템 컴포넌트
        self.stability_manager = None
        self.news_collector = None
        self.data_collector = None
        self.predictor = None
        self.trading_executor = None
        self.notification_system = None
        self.storage = None

        # API 키 (환경변수에서 로드)
        self.access_key = os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = os.getenv('UPBIT_SECRET_KEY')

        # 설정 파일 경로
        self.config_dir = project_root / "config"
        self.news_config_path = self.config_dir / "news_sources.yaml"

        # 거래 설정
        self.trading_symbols = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']
        self.update_interval = 60  # 1분 주기

        self.logger.info(f"=== 코인 자동매매 AI 시스템 v2.0 초기화 (모드: {trading_mode}) ===")

    def _SetupLogging(self):
        """전역 로깅 설정"""
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "trading_ai.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _InitializeComponents(self):
        """시스템 컴포넌트들 초기화"""
        try:
            self.logger.info("🔧 시스템 컴포넌트 초기화 시작")

            # 1. 안정성 관리자 초기화
            stability_config = {
                'error_threshold': 5,
                'monitoring_interval': 10
            }
            self.stability_manager = StabilityManager(stability_config)
            self.logger.info("✅ 안정성 관리자 초기화 완료")

            # 2. 데이터 저장소 초기화
            self.storage = DataStorage()
            self.logger.info("✅ 데이터 저장소 초기화 완료")

            # 3. 시장 데이터 수집기 초기화
            self.data_collector = MarketDataCollector()
            self.logger.info("✅ 시장 데이터 수집기 초기화 완료")

            # 4. AI 예측기 초기화
            self.predictor = EnsemblePredictor()
            self.logger.info("✅ AI 예측기 초기화 완료")

            # 5. 뉴스 수집기 초기화
            if self.news_config_path.exists():
                self.news_collector = NewsCollector(str(self.news_config_path))
                self.logger.info("✅ 뉴스 수집기 초기화 완료")
            else:
                self.logger.warning(f"뉴스 설정 파일 없음: {self.news_config_path}")

            # 6. 알림 시스템 초기화
            notification_config = {
                'telegram': {
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                }
            }
            self.notification_system = NotificationSystem(notification_config)
            self.logger.info("✅ 알림 시스템 초기화 완료")

            # 7. 거래 실행기 초기화 (API 키가 있는 경우만)
            if self.trading_mode == 'live' and self.access_key and self.secret_key:
                self.trading_executor = LiveTradingExecutor(
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    initial_balance=1000000  # 100만원 기본값
                )
                self.logger.info("✅ 라이브 거래 실행기 초기화 완료")
            elif self.trading_mode == 'paper':
                # 모의거래용 실행기 (API 키 불필요)
                self.trading_executor = LiveTradingExecutor(
                    access_key="demo_key",
                    secret_key="demo_secret",
                    initial_balance=10000000  # 1천만원 모의자금
                )
                self.logger.info("✅ 모의거래 실행기 초기화 완료")
            else:
                self.logger.error("API 키가 없어 거래 실행기를 초기화할 수 없습니다")
                return False

            # 컴포넌트들을 안정성 관리자에 등록
            self._RegisterHealthChecks()

            self.logger.info("🎉 모든 컴포넌트 초기화 완료")
            return True

        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            return False

    def _RegisterHealthChecks(self):
        """컴포넌트 건강 검사 등록"""
        try:
            if self.news_collector:
                self.stability_manager.RegisterComponent(
                    "news_collector",
                    self._CheckNewsCollectorHealth
                )

            if self.data_collector:
                self.stability_manager.RegisterComponent(
                    "data_collector",
                    self._CheckDataCollectorHealth
                )

            if self.predictor:
                self.stability_manager.RegisterComponent(
                    "predictor",
                    self._CheckPredictorHealth
                )

            if self.trading_executor:
                self.stability_manager.RegisterComponent(
                    "trading_executor",
                    self._CheckTradingExecutorHealth
                )

            self.logger.info("건강 검사 등록 완료")

        except Exception as e:
            self.logger.error(f"건강 검사 등록 실패: {e}")

    def _CheckNewsCollectorHealth(self) -> bool:
        """뉴스 수집기 건강 상태 확인"""
        try:
            if not self.news_collector:
                return False
            stats = self.news_collector.GetCollectionStats()
            return stats['total_articles'] >= 0
        except Exception as e:
            self.logger.error(f"뉴스 수집기 건강 검사 실패: {e}")
            return False

    def _CheckDataCollectorHealth(self) -> bool:
        """데이터 수집기 건강 상태 확인"""
        try:
            return self.data_collector is not None
        except Exception:
            return False

    def _CheckPredictorHealth(self) -> bool:
        """AI 예측기 건강 상태 확인"""
        try:
            return self.predictor is not None
        except Exception:
            return False

    def _CheckTradingExecutorHealth(self) -> bool:
        """거래 실행기 건강 상태 확인"""
        try:
            if not self.trading_executor:
                return False
            status = self.trading_executor.get_status()
            return not status.get('emergency_stop', False)
        except Exception:
            return False

    def Start(self):
        """시스템 시작"""
        self.logger.info("🚀 코인 자동매매 AI 시스템 시작")

        # 컴포넌트 초기화
        if not self._InitializeComponents():
            self.logger.error("컴포넌트 초기화 실패로 시스템 종료")
            return

        # 안정성 모니터링 시작
        self.stability_manager.StartMonitoring()

        self.is_running = True
        self.logger.info("✅ 시스템 시작 완료 - 메인 루프 진입")

        # 메인 실행 루프
        self._MainLoop()

    def _MainLoop(self):
        """메인 실행 루프 - 완전한 자동매매 시스템"""
        loop_count = 0

        while self.is_running:
            try:
                loop_count += 1
                self.logger.info(f"=== 메인 루프 {loop_count} 실행 ===")

                # 시스템 상태 확인
                system_health = self.stability_manager.GetSystemHealth()
                self.logger.info(f"시스템 상태: {system_health['overall_status']}")

                # 시스템이 위험 상태면 대기
                if system_health['overall_status'] == SystemStatus.CRITICAL.value:
                    self.logger.error("시스템 위험 상태 - 거래 중단하고 대기")
                    if self.notification_system:
                        asyncio.run(self.notification_system.send_system_alert(
                            'ERROR', '시스템 위험 상태 - 거래 중단'
                        ))
                    time.sleep(30)
                    continue

                # Phase 1: 시장 데이터 수집
                self._ExecuteDataCollection()

                # Phase 2: 뉴스 수집 및 분석
                if self.news_collector:
                    self._ExecuteNewsCollection()

                # Phase 3: AI 예측 실행
                predictions = self._ExecuteAIPrediction()

                # Phase 4: 거래 실행 (예측 결과 기반)
                if predictions and self.trading_executor:
                    self._ExecuteTrading(predictions)

                # Phase 5: 포트폴리오 모니터링
                self._MonitorPortfolio()

                # 다음 실행까지 대기
                self.logger.info(f"메인 루프 {loop_count} 완료 - {self.update_interval}초 대기")
                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                self.logger.info("사용자에 의한 시스템 중단 요청")
                break
            except Exception as e:
                self.logger.error(f"메인 루프 실행 중 오류: {e}")
                if self.notification_system:
                    asyncio.run(self.notification_system.send_system_alert(
                        'ERROR', f'메인 루프 오류: {str(e)}'
                    ))
                time.sleep(10)  # 오류 발생시 10초 대기

        self.logger.info("메인 루프 종료")

    def _ExecuteNewsCollection(self):
        """뉴스 수집 및 분석 실행"""
        try:
            self.logger.info("📰 뉴스 수집 시작")

            # 안전한 뉴스 수집 실행
            news_articles = self.stability_manager.SafeExecute(
                self.news_collector.CollectAllNews
            )

            self.logger.info(f"✅ {len(news_articles)}개 뉴스 수집 완료")

            # 수집 통계 로깅
            stats = self.news_collector.GetCollectionStats()
            self.logger.info(f"수집 통계: {stats}")

            # 향후 뉴스 분석 로직 추가 예정
            # - 감정 분석
            # - 코인별 영향도 계산
            # - 예측 신호 생성

        except Exception as e:
            self.logger.error(f"뉴스 수집 실행 실패: {e}")

    def Stop(self):
        """시스템 정지"""
        self.logger.info("🛑 시스템 정지 시작")

        self.is_running = False

        # 안정성 관리자 정지
        if self.stability_manager:
            self.stability_manager.StopMonitoring()

        self.logger.info("✅ 시스템 정지 완료")

    def _SignalHandler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 등)"""
        self.logger.info(f"시그널 {signum} 수신 - 안전한 종료 시작")
        self.Stop()

def main():
    """메인 함수"""
    print("🤖 코인 자동매매 AI 시스템")
    print("=" * 50)

    # 시그널 핸들러 등록
    trading_ai = CoinTradingAI()
    signal.signal(signal.SIGINT, trading_ai._SignalHandler)
    signal.signal(signal.SIGTERM, trading_ai._SignalHandler)

    try:
        # 시스템 시작
        trading_ai.Start()
    except Exception as e:
        print(f"시스템 시작 실패: {e}")
        return 1
    finally:
        # 정리 작업
        trading_ai.Stop()

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)