#!/usr/bin/env python3
"""
앙상블 예측 시스템
- Random Forest + 강화학습 + 뉴스 감정 분석 통합
- 가중 투표를 통한 최종 예측
- 신뢰도 기반 동적 가중치 조정
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .RandomForestPredictor import RandomForestPredictor, PredictionResult
from .ReinforcementLearningAgent import ReinforcementLearningAgent, ActionResult


class SignalType(Enum):
    """거래 신호 타입"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class EnsemblePrediction:
    """앙상블 예측 결과"""
    market: str
    timestamp: datetime
    final_signal: SignalType
    confidence: float
    expected_return: float

    # 개별 모델 결과
    rf_prediction: Optional[PredictionResult] = None
    rl_action: Optional[ActionResult] = None
    news_sentiment: Optional[float] = None

    # 가중치 정보
    model_weights: Optional[Dict[str, float]] = None

    # 추론 근거
    reasoning: Optional[str] = None


class ModelWeightManager:
    """
    모델 가중치 관리자
    각 모델의 최근 성능에 따라 동적으로 가중치 조정
    """

    def __init__(self, window_size: int = 100):
        """
        초기화

        Args:
            window_size: 성능 평가 윈도우 크기
        """
        self.window_size = window_size
        self.performance_history = {
            'random_forest': [],
            'reinforcement_learning': [],
            'news_sentiment': []
        }

        # 기본 가중치
        self.base_weights = {
            'random_forest': 0.4,
            'reinforcement_learning': 0.35,
            'news_sentiment': 0.25
        }

        self.logger = logging.getLogger(__name__)

    def update_performance(self, model_name: str, accuracy: float):
        """
        모델 성능 업데이트

        Args:
            model_name: 모델명
            accuracy: 정확도 점수 (0-1)
        """
        if model_name in self.performance_history:
            self.performance_history[model_name].append(accuracy)

            # 윈도우 크기 유지
            if len(self.performance_history[model_name]) > self.window_size:
                self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        성능 기반 동적 가중치 계산

        Returns:
            모델별 가중치 딕셔너리
        """
        weights = self.base_weights.copy()

        # 각 모델의 최근 성능 평균 계산
        recent_performance = {}
        for model_name, history in self.performance_history.items():
            if len(history) > 0:
                # 최근 성능에 더 높은 가중치 부여
                recent_scores = history[-min(20, len(history)):]  # 최근 20개
                recent_performance[model_name] = np.mean(recent_scores)
            else:
                recent_performance[model_name] = 0.5  # 기본값

        # 성능이 높은 모델에 더 높은 가중치 할당
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            performance_weights = {
                model: perf / total_performance
                for model, perf in recent_performance.items()
            }

            # 기본 가중치와 성능 가중치를 조합 (0.7:0.3 비율)
            for model in weights:
                weights[model] = (0.7 * weights[model] +
                                0.3 * performance_weights.get(model, 0))

        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {model: weight / total_weight for model, weight in weights.items()}

        return weights


class EnsemblePredictor:
    """
    앙상블 예측 시스템
    여러 모델의 예측을 통합하여 최종 거래 신호 생성
    """

    def __init__(self, config: Dict = None):
        """
        초기화

        Args:
            config: 앙상블 설정
        """
        self.logger = logging.getLogger(__name__)

        # 기본 설정
        default_config = {
            'signal_thresholds': {
                'strong_buy': 0.8,
                'buy': 0.6,
                'hold': 0.4,
                'sell': 0.2,
                'strong_sell': 0.0
            },
            'min_confidence': 0.6,
            'news_sentiment_weight': 0.3,
            'technical_weight': 0.5,
            'rl_weight': 0.2,
            'performance_tracking': True
        }

        self.config = {**default_config, **(config or {})}

        # 개별 모델들
        self.rf_predictor = None
        self.rl_agent = None

        # 가중치 관리자
        self.weight_manager = ModelWeightManager()

        # 예측 이력
        self.prediction_history = []

        self.logger.info("앙상블 예측 시스템 초기화 완료")

    def set_models(self, rf_predictor: RandomForestPredictor = None,
                  rl_agent: ReinforcementLearningAgent = None):
        """
        예측 모델들 설정

        Args:
            rf_predictor: Random Forest 예측기
            rl_agent: 강화학습 에이전트
        """
        if rf_predictor:
            self.rf_predictor = rf_predictor
            self.logger.info("Random Forest 예측기 설정 완료")

        if rl_agent:
            self.rl_agent = rl_agent
            self.logger.info("강화학습 에이전트 설정 완료")

    def predict(self, market_data: pd.DataFrame, news_sentiment: float = None) -> EnsemblePrediction:
        """
        앙상블 예측 수행

        Args:
            market_data: 시장 데이터
            news_sentiment: 뉴스 감정 점수 (-1 ~ 1)

        Returns:
            앙상블 예측 결과
        """
        try:
            current_time = datetime.now()
            market = market_data['market'].iloc[-1] if 'market' in market_data.columns else 'UNKNOWN'

            # 개별 모델 예측 수집
            rf_predictions = []
            rl_action = None

            # Random Forest 예측
            if self.rf_predictor and self.rf_predictor.is_trained:
                try:
                    rf_results = self.rf_predictor.Predict(market_data, 'both')
                    rf_predictions = rf_results
                    self.logger.debug(f"RF 예측 완료: {len(rf_results)}개 결과")
                except Exception as e:
                    self.logger.error(f"Random Forest 예측 오류: {e}")

            # 강화학습 예측
            if self.rl_agent and self.rl_agent.is_trained:
                try:
                    rl_action = self.rl_agent.predict(market_data)
                    self.logger.debug(f"RL 예측 완료: {rl_action.action_name}")
                except Exception as e:
                    self.logger.error(f"강화학습 예측 오류: {e}")

            # 예측 결과 통합
            ensemble_result = self._integrate_predictions(
                market=market,
                timestamp=current_time,
                rf_predictions=rf_predictions,
                rl_action=rl_action,
                news_sentiment=news_sentiment
            )

            # 예측 이력 저장
            self.prediction_history.append(ensemble_result)

            # 이력 크기 제한
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            return ensemble_result

        except Exception as e:
            self.logger.error(f"앙상블 예측 오류: {e}")
            return self._create_fallback_prediction(market_data)

    def _integrate_predictions(self, market: str, timestamp: datetime,
                             rf_predictions: List[PredictionResult],
                             rl_action: Optional[ActionResult],
                             news_sentiment: Optional[float]) -> EnsemblePrediction:
        """
        개별 예측 결과를 통합하여 최종 신호 생성

        Args:
            market: 마켓명
            timestamp: 예측 시간
            rf_predictions: Random Forest 예측 결과들
            rl_action: 강화학습 액션 결과
            news_sentiment: 뉴스 감정 점수

        Returns:
            통합된 예측 결과
        """
        # 동적 가중치 계산
        weights = self.weight_manager.get_dynamic_weights()

        # 각 모델의 신호 점수 계산 (0-1 범위)
        signals = []
        confidences = []

        # Random Forest 신호
        rf_signal = 0.5  # 중립
        rf_confidence = 0.0
        rf_prediction = None

        if rf_predictions:
            # 방향 예측 결과 우선 사용
            direction_pred = next((p for p in rf_predictions if p.prediction_type == 'direction'), None)
            if direction_pred:
                rf_prediction = direction_pred
                rf_signal = direction_pred.predicted_value  # 0 또는 1
                rf_confidence = direction_pred.confidence
            elif rf_predictions:
                # 가격 예측 결과 사용
                price_pred = rf_predictions[0]
                rf_prediction = price_pred
                rf_signal = 0.6 if price_pred.predicted_value > 0 else 0.4  # 간단한 변환
                rf_confidence = price_pred.confidence

        signals.append(rf_signal)
        confidences.append(rf_confidence)

        # 강화학습 신호
        rl_signal = 0.5  # 중립
        rl_confidence = 0.0

        if rl_action:
            # 액션을 신호로 변환
            action_to_signal = {
                0: 0.5,   # HOLD
                1: 0.65,  # BUY_WEAK
                2: 0.85,  # BUY_STRONG
                3: 0.35,  # SELL_WEAK
                4: 0.15   # SELL_STRONG
            }

            rl_signal = action_to_signal.get(rl_action.action, 0.5)
            rl_confidence = rl_action.confidence

        signals.append(rl_signal)
        confidences.append(rl_confidence)

        # 뉴스 감정 신호
        news_signal = 0.5  # 중립
        news_confidence = 0.3  # 기본 신뢰도

        if news_sentiment is not None:
            # 감정 점수를 0-1 범위로 변환
            news_signal = (news_sentiment + 1) / 2  # -1~1 -> 0~1
            news_confidence = min(abs(news_sentiment), 1.0)  # 절댓값이 클수록 신뢰도 높음

        signals.append(news_signal)
        confidences.append(news_confidence)

        # 가중 평균 계산
        model_names = ['random_forest', 'reinforcement_learning', 'news_sentiment']

        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for i, (signal, confidence, model_name) in enumerate(zip(signals, confidences, model_names)):
            weight = weights.get(model_name, 0.0) * (1 + confidence)  # 신뢰도에 따른 가중치 조정
            weighted_signal += signal * weight
            weighted_confidence += confidence * weight
            total_weight += weight

        if total_weight > 0:
            final_signal_score = weighted_signal / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_signal_score = 0.5
            final_confidence = 0.0

        # 신호 타입 결정
        final_signal = self._score_to_signal(final_signal_score)

        # 최소 신뢰도 체크
        if final_confidence < self.config['min_confidence']:
            final_signal = SignalType.HOLD
            final_confidence = 0.0

        # 예상 수익률 계산
        expected_return = self._calculate_expected_return(final_signal_score, final_confidence)

        # 추론 근거 생성
        reasoning = self._generate_reasoning(rf_prediction, rl_action, news_sentiment, weights)

        return EnsemblePrediction(
            market=market,
            timestamp=timestamp,
            final_signal=final_signal,
            confidence=final_confidence,
            expected_return=expected_return,
            rf_prediction=rf_prediction,
            rl_action=rl_action,
            news_sentiment=news_sentiment,
            model_weights=weights,
            reasoning=reasoning
        )

    def _score_to_signal(self, score: float) -> SignalType:
        """
        점수를 신호 타입으로 변환

        Args:
            score: 신호 점수 (0-1)

        Returns:
            신호 타입
        """
        thresholds = self.config['signal_thresholds']

        if score >= thresholds['strong_buy']:
            return SignalType.STRONG_BUY
        elif score >= thresholds['buy']:
            return SignalType.BUY
        elif score >= thresholds['sell']:
            return SignalType.HOLD
        elif score >= thresholds['strong_sell']:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL

    def _calculate_expected_return(self, signal_score: float, confidence: float) -> float:
        """
        예상 수익률 계산

        Args:
            signal_score: 신호 점수
            confidence: 신뢰도

        Returns:
            예상 수익률 (%)
        """
        # 신호 강도와 신뢰도를 기반으로 예상 수익률 계산
        base_return = (signal_score - 0.5) * 2  # -1 ~ 1 범위
        expected_return = base_return * confidence * 5  # 최대 5% 수익률 가정

        return expected_return

    def _generate_reasoning(self, rf_prediction: Optional[PredictionResult],
                          rl_action: Optional[ActionResult],
                          news_sentiment: Optional[float],
                          weights: Dict[str, float]) -> str:
        """
        예측 근거 생성

        Args:
            rf_prediction: RF 예측 결과
            rl_action: RL 액션 결과
            news_sentiment: 뉴스 감정
            weights: 모델 가중치

        Returns:
            추론 근거 문자열
        """
        reasoning_parts = []

        # 모델 가중치 정보
        reasoning_parts.append(f"가중치 - RF:{weights['random_forest']:.2f} RL:{weights['reinforcement_learning']:.2f} News:{weights['news_sentiment']:.2f}")

        # Random Forest 결과
        if rf_prediction:
            reasoning_parts.append(f"RF: {rf_prediction.prediction_type} 신뢰도 {rf_prediction.confidence:.2f}")

        # 강화학습 결과
        if rl_action:
            reasoning_parts.append(f"RL: {rl_action.action_name} 신뢰도 {rl_action.confidence:.2f}")

        # 뉴스 감정
        if news_sentiment is not None:
            sentiment_label = "긍정" if news_sentiment > 0.1 else "부정" if news_sentiment < -0.1 else "중립"
            reasoning_parts.append(f"뉴스: {sentiment_label}({news_sentiment:.2f})")

        return " | ".join(reasoning_parts)

    def _create_fallback_prediction(self, market_data: pd.DataFrame) -> EnsemblePrediction:
        """
        폴백 예측 생성 (오류 발생시)

        Args:
            market_data: 시장 데이터

        Returns:
            폴백 예측 결과
        """
        market = market_data['market'].iloc[-1] if 'market' in market_data.columns else 'UNKNOWN'

        return EnsemblePrediction(
            market=market,
            timestamp=datetime.now(),
            final_signal=SignalType.HOLD,
            confidence=0.0,
            expected_return=0.0,
            reasoning="예측 오류로 인한 홀드 신호"
        )

    def update_model_performance(self, model_name: str, accuracy: float):
        """
        모델 성능 업데이트

        Args:
            model_name: 모델명
            accuracy: 정확도
        """
        self.weight_manager.update_performance(model_name, accuracy)
        self.logger.debug(f"모델 성능 업데이트: {model_name} = {accuracy:.3f}")

    def get_recent_predictions(self, hours: int = 24) -> List[EnsemblePrediction]:
        """
        최근 예측 결과 조회

        Args:
            hours: 조회할 시간 범위 (시간)

        Returns:
            최근 예측 결과들
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [pred for pred in self.prediction_history if pred.timestamp >= cutoff_time]

    def get_performance_summary(self) -> Dict:
        """
        성능 요약 정보

        Returns:
            성능 요약 딕셔너리
        """
        if not self.prediction_history:
            return {'total_predictions': 0}

        recent_predictions = self.get_recent_predictions(24)  # 최근 24시간

        # 신호 분포 계산
        signal_counts = {}
        confidence_scores = []

        for pred in recent_predictions:
            signal_type = pred.final_signal.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            confidence_scores.append(pred.confidence)

        return {
            'total_predictions': len(self.prediction_history),
            'recent_predictions_24h': len(recent_predictions),
            'signal_distribution': signal_counts,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'current_weights': self.weight_manager.get_dynamic_weights(),
            'model_performance_history': {
                name: history[-10:] if history else []  # 최근 10개
                for name, history in self.weight_manager.performance_history.items()
            }
        }

    def clear_history(self):
        """예측 이력 초기화"""
        self.prediction_history.clear()
        self.logger.info("예측 이력 초기화 완료")