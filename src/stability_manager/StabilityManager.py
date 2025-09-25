"""
안정성 관리 모듈
AI 시스템의 안정적 운영을 보장하는 핵심 모듈
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import traceback
from retrying import retry
import timeout_decorator

class SystemStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"

@dataclass
class HealthCheck:
    component: str
    status: SystemStatus
    last_check: float
    error_count: int
    message: str

class StabilityManager:
    """
    시스템 안정성을 관리하는 핵심 클래스
    - 모든 AI 예측에 안전장치 적용
    - 예외 상황 자동 처리 및 복구
    - 시스템 건전성 실시간 모니터링
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_status = SystemStatus.HEALTHY
        self.error_threshold = config.get('error_threshold', 5)
        self.monitoring_interval = config.get('monitoring_interval', 10)  # 초
        self.is_monitoring = False

        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # 모니터링 스레드
        self.monitor_thread: Optional[threading.Thread] = None

        self.logger.info("StabilityManager 초기화 완료")

    def _setup_logging(self):
        """로깅 시스템 설정"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def StartMonitoring(self):
        """시스템 모니터링 시작"""
        if self.is_monitoring:
            self.logger.warning("이미 모니터링이 실행중입니다")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._MonitoringLoop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.logger.info("시스템 모니터링 시작")

    def StopMonitoring(self):
        """시스템 모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        self.logger.info("시스템 모니터링 중지")

    def _MonitoringLoop(self):
        """모니터링 메인 루프"""
        while self.is_monitoring:
            try:
                self._PerformHealthChecks()
                self._EvaluateSystemStatus()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"모니터링 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생시 잠시 대기

    def RegisterComponent(self, component_name: str, health_check_func: Callable[[], bool]):
        """시스템 컴포넌트 등록"""
        self.health_checks[component_name] = HealthCheck(
            component=component_name,
            status=SystemStatus.HEALTHY,
            last_check=time.time(),
            error_count=0,
            message="등록됨"
        )

        # 건강 검사 함수도 저장 (추후 구현)
        setattr(self, f'_check_{component_name}', health_check_func)

        self.logger.info(f"컴포넌트 등록: {component_name}")

    def _PerformHealthChecks(self):
        """모든 등록된 컴포넌트의 건강 검사 수행"""
        for component_name, health_check in self.health_checks.items():
            try:
                check_func = getattr(self, f'_check_{component_name}', None)
                if check_func:
                    result = check_func()
                    if result:
                        health_check.status = SystemStatus.HEALTHY
                        health_check.error_count = 0
                        health_check.message = "정상"
                    else:
                        health_check.error_count += 1
                        if health_check.error_count >= self.error_threshold:
                            health_check.status = SystemStatus.CRITICAL
                            health_check.message = f"연속 {health_check.error_count}회 실패"
                        else:
                            health_check.status = SystemStatus.WARNING
                            health_check.message = f"{health_check.error_count}회 실패"

                health_check.last_check = time.time()

            except Exception as e:
                health_check.error_count += 1
                health_check.status = SystemStatus.CRITICAL
                health_check.message = f"검사 중 예외: {str(e)}"
                self.logger.error(f"{component_name} 건강 검사 실패: {e}")

    def _EvaluateSystemStatus(self):
        """전체 시스템 상태 평가"""
        critical_count = sum(1 for hc in self.health_checks.values()
                           if hc.status == SystemStatus.CRITICAL)
        warning_count = sum(1 for hc in self.health_checks.values()
                          if hc.status == SystemStatus.WARNING)

        if critical_count > 0:
            self.system_status = SystemStatus.CRITICAL
            self.logger.error(f"시스템 위험 상태: {critical_count}개 컴포넌트 오류")
        elif warning_count > 2:  # 경고가 2개 이상이면 주의
            self.system_status = SystemStatus.WARNING
            self.logger.warning(f"시스템 주의 상태: {warning_count}개 컴포넌트 경고")
        else:
            self.system_status = SystemStatus.HEALTHY

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def SafeExecute(self, func: Callable, *args, **kwargs) -> Any:
        """
        함수를 안전하게 실행 (재시도 로직 포함)
        AI 예측 함수 등 중요한 작업에 사용
        """
        try:
            # 시스템이 위험 상태면 실행 중단
            if self.system_status == SystemStatus.CRITICAL:
                raise Exception("시스템이 위험 상태입니다. 작업을 중단합니다.")

            result = func(*args, **kwargs)
            return result

        except Exception as e:
            self.logger.error(f"SafeExecute 실행 중 오류: {e}")
            self.logger.error(traceback.format_exc())
            raise

    @timeout_decorator.timeout(30)  # 30초 타임아웃
    def SafePrediction(self, prediction_func: Callable, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        AI 예측을 안전하게 실행
        신뢰도가 임계값 미만이면 거래 보류
        """
        try:
            prediction_result = prediction_func()

            # 예측 결과 검증
            if not isinstance(prediction_result, dict):
                raise ValueError("예측 결과가 딕셔너리 형태가 아닙니다")

            confidence = prediction_result.get('confidence', 0.0)
            if confidence < confidence_threshold:
                return {
                    'action': 'hold',
                    'confidence': confidence,
                    'reason': f'신뢰도 부족 ({confidence:.2f} < {confidence_threshold})'
                }

            return prediction_result

        except Exception as e:
            self.logger.error(f"AI 예측 중 오류: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'예측 실패: {str(e)}'
            }

    def ValidateTradeSignal(self, signal: Dict[str, Any]) -> bool:
        """
        거래 신호의 유효성 검증
        모든 안전장치를 통과해야 True 반환
        """
        try:
            # 필수 필드 검증
            required_fields = ['action', 'symbol', 'confidence', 'amount']
            for field in required_fields:
                if field not in signal:
                    self.logger.error(f"거래 신호에 필수 필드 누락: {field}")
                    return False

            # 액션 유효성 검증
            valid_actions = ['buy', 'sell', 'hold']
            if signal['action'] not in valid_actions:
                self.logger.error(f"잘못된 거래 액션: {signal['action']}")
                return False

            # 신뢰도 검증
            confidence = signal.get('confidence', 0.0)
            if confidence < 0.7:
                self.logger.warning(f"신뢰도 부족으로 거래 신호 거부: {confidence}")
                return False

            # 금액 검증
            amount = signal.get('amount', 0)
            if amount <= 0:
                self.logger.error(f"잘못된 거래 금액: {amount}")
                return False

            # 시스템 상태 검증
            if self.system_status == SystemStatus.CRITICAL:
                self.logger.error("시스템 위험 상태로 거래 신호 거부")
                return False

            return True

        except Exception as e:
            self.logger.error(f"거래 신호 검증 중 오류: {e}")
            return False

    def GetSystemHealth(self) -> Dict[str, Any]:
        """현재 시스템 건강 상태 반환"""
        return {
            'overall_status': self.system_status.value,
            'components': {
                name: {
                    'status': hc.status.value,
                    'last_check': hc.last_check,
                    'error_count': hc.error_count,
                    'message': hc.message
                } for name, hc in self.health_checks.items()
            },
            'timestamp': time.time()
        }

    def EmergencyShutdown(self, reason: str):
        """긴급 시스템 종료"""
        self.logger.critical(f"긴급 종료 실행: {reason}")
        self.system_status = SystemStatus.SHUTDOWN
        self.StopMonitoring()

        # 여기에 모든 거래 중단, 포지션 정리 등의 로직 추가
        # (추후 TradeExecutor와 연동)

    def __del__(self):
        """소멸자 - 모니터링 정리"""
        try:
            self.StopMonitoring()
        except:
            pass