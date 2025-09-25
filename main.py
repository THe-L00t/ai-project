#!/usr/bin/env python3
"""
코인 자동매매 AI 시스템 메인 실행 파일
- 안정적인 시스템 구동
- 뉴스 기반 예측 및 거래 실행
- 실시간 모니터링 및 리스크 관리
"""

import os
import sys
import logging
import time
import signal
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.stability_manager.StabilityManager import StabilityManager, SystemStatus
from src.news_analyzer.NewsCollector import NewsCollector

class CoinTradingAI:
    """
    코인 자동매매 AI 메인 시스템
    모든 컴포넌트를 통합하여 안정적으로 운영
    """

    def __init__(self):
        # 로깅 설정
        self._SetupLogging()
        self.logger = logging.getLogger(__name__)

        # 시스템 컴포넌트 초기화
        self.stability_manager = None
        self.news_collector = None
        self.is_running = False

        # 설정 파일 경로
        self.config_dir = project_root / "config"
        self.news_config_path = self.config_dir / "news_sources.yaml"

        self.logger.info("=== 코인 자동매매 AI 시스템 초기화 ===")

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
            # 안정성 관리자 초기화
            stability_config = {
                'error_threshold': 5,
                'monitoring_interval': 10
            }
            self.stability_manager = StabilityManager(stability_config)

            # 뉴스 수집기 초기화
            if self.news_config_path.exists():
                self.news_collector = NewsCollector(str(self.news_config_path))
            else:
                self.logger.error(f"뉴스 설정 파일 없음: {self.news_config_path}")
                return False

            # 컴포넌트들을 안정성 관리자에 등록
            self.stability_manager.RegisterComponent(
                "news_collector",
                self._CheckNewsCollectorHealth
            )

            self.logger.info("모든 컴포넌트 초기화 완료")
            return True

        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            return False

    def _CheckNewsCollectorHealth(self) -> bool:
        """뉴스 수집기 건강 상태 확인"""
        try:
            if not self.news_collector:
                return False

            # 간단한 건강 검사 (마지막 수집 시간 등)
            stats = self.news_collector.GetCollectionStats()
            return stats['total_articles'] >= 0  # 기본적인 상태 확인

        except Exception as e:
            self.logger.error(f"뉴스 수집기 건강 검사 실패: {e}")
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
        """메인 실행 루프"""
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
                    time.sleep(30)
                    continue

                # Phase 1: 뉴스 수집 및 분석
                self._ExecuteNewsCollection()

                # Phase 2: AI 예측 (향후 구현)
                # self._ExecuteAIPrediction()

                # Phase 3: 거래 실행 (향후 구현)
                # self._ExecuteTrading()

                # Phase 4: 성과 모니터링 (향후 구현)
                # self._MonitorPerformance()

                # 다음 실행까지 대기 (60초 주기)
                self.logger.info("메인 루프 완료 - 60초 대기")
                time.sleep(60)

            except KeyboardInterrupt:
                self.logger.info("사용자에 의한 시스템 중단 요청")
                break
            except Exception as e:
                self.logger.error(f"메인 루프 실행 중 오류: {e}")
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