#!/usr/bin/env python3
"""
적응형 학습 AI 시스템
- 실시간 트레이딩 경험 학습
- 예측 정확도 피드백 시스템
- 지속적 모델 개선
- 모든 상호작용에서 학습
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque
import pickle
import json

# 머신러닝
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.linear_model import SGDRegressor  # 온라인 학습용

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from exchange.UpbitAPI import UpbitAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/adaptive_learning.log')
    ]
)
logger = logging.getLogger(__name__)

class TradingExperienceCollector:
    """트레이딩 경험 수집기"""

    def __init__(self):
        self.trade_history = deque(maxlen=10000)
        self.prediction_history = deque(maxlen=5000)
        self.market_events = deque(maxlen=1000)
        self.learning_sessions = []

    def record_trade(self, coin, action, price, quantity, timestamp=None):
        """거래 기록"""
        trade_record = {
            'coin': coin,
            'action': action,  # 'BUY', 'SELL', 'HOLD'
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp or datetime.now(),
            'market_context': self._get_market_context(coin)
        }
        self.trade_history.append(trade_record)
        logger.info(f"📝 거래 기록: {coin} {action} @ {price:,.0f}원")

    def record_prediction(self, coin, predicted_change, actual_change, confidence, prediction_time):
        """예측 기록 및 정확도 추적"""
        prediction_record = {
            'coin': coin,
            'predicted_change': predicted_change,
            'actual_change': actual_change,
            'confidence': confidence,
            'prediction_time': prediction_time,
            'verification_time': datetime.now(),
            'accuracy': self._calculate_prediction_accuracy(predicted_change, actual_change),
            'direction_correct': (predicted_change * actual_change) > 0
        }
        self.prediction_history.append(prediction_record)
        logger.info(f"🎯 예측 검증: {coin} 예측 {predicted_change:+.2f}% vs 실제 {actual_change:+.2f}% (정확도: {prediction_record['accuracy']:.2f})")

    def record_market_event(self, event_type, description, impact_coins=None):
        """시장 이벤트 기록"""
        event_record = {
            'event_type': event_type,  # 'NEWS', 'PRICE_SPIKE', 'VOLUME_SURGE', 'ERROR'
            'description': description,
            'impact_coins': impact_coins or [],
            'timestamp': datetime.now()
        }
        self.market_events.append(event_record)
        logger.info(f"⚡ 시장 이벤트: {event_type} - {description}")

    def _get_market_context(self, coin):
        """거래 당시 시장 컨텍스트"""
        # 간단한 시장 정보 수집
        try:
            from news_sentiment_ai import NewsSentimentAI
            # 현재 감정 점수, 볼륨 등 수집
            return {
                'timestamp': datetime.now(),
                'volatility': 'medium',  # 실제로는 계산
                'volume_trend': 'normal'
            }
        except:
            return {'timestamp': datetime.now()}

    def _calculate_prediction_accuracy(self, predicted, actual):
        """예측 정확도 계산"""
        if actual == 0:
            return 0.5  # 중립

        # 방향 정확도 + 크기 정확도
        direction_score = 1.0 if (predicted * actual) > 0 else 0.0
        magnitude_error = abs(predicted - actual) / (abs(actual) + 0.1)
        magnitude_score = max(0, 1 - magnitude_error)

        return (direction_score * 0.7) + (magnitude_score * 0.3)

class AdaptiveLearningEngine:
    """적응형 학습 엔진"""

    def __init__(self):
        load_dotenv()

        self.upbit = UpbitAPI()
        self.experience_collector = TradingExperienceCollector()

        # 온라인 학습 모델들
        self.online_models = {}
        self.online_scalers = {}

        # 성능 추적
        self.model_performance = {}
        self.learning_rate = 0.01

        # 학습 데이터 버퍼
        self.learning_buffer = deque(maxlen=1000)

        logger.info("🧠 적응형 학습 엔진 초기화 완료")

    def initialize_online_models(self, coins=['BTC', 'ETH', 'ADA', 'DOT']):
        """온라인 학습 모델 초기화"""
        for coin in coins:
            # SGD 기반 온라인 학습 모델
            self.online_models[coin] = SGDRegressor(
                learning_rate='adaptive',
                eta0=self.learning_rate,
                random_state=42
            )
            self.online_scalers[coin] = StandardScaler()
            self.model_performance[coin] = {
                'predictions': 0,
                'correct_directions': 0,
                'avg_accuracy': 0.5,
                'last_update': datetime.now()
            }

        logger.info(f"🎯 {len(coins)}개 코인 온라인 학습 모델 초기화")

    def collect_real_time_features(self, coin):
        """실시간 특성 수집"""
        features = []

        try:
            # 1. 가격 데이터
            ticker = self.upbit.GetTicker([f'KRW-{coin}'])
            if ticker:
                price_data = ticker[0]
                features.extend([
                    float(price_data.change_rate),
                    float(price_data.acc_trade_volume_24h),
                    float(price_data.trade_price) / 100000,  # 정규화
                    float(price_data.high_price - price_data.low_price) / price_data.trade_price  # 변동성
                ])
            else:
                features.extend([0.0] * 4)

            # 2. 최근 거래 히스토리 특성
            recent_trades = [t for t in self.experience_collector.trade_history
                           if t['coin'] == coin and
                           (datetime.now() - t['timestamp']).total_seconds() < 3600]

            if recent_trades:
                buy_count = sum(1 for t in recent_trades if t['action'] == 'BUY')
                sell_count = sum(1 for t in recent_trades if t['action'] == 'SELL')
                features.extend([
                    len(recent_trades),
                    buy_count - sell_count,  # 순 매수/매도
                    buy_count / (len(recent_trades) + 0.1)  # 매수 비율
                ])
            else:
                features.extend([0.0] * 3)

            # 3. 예측 성능 메타 특성
            performance = self.model_performance.get(coin, {})
            features.extend([
                performance.get('avg_accuracy', 0.5),
                performance.get('correct_directions', 0) / max(performance.get('predictions', 1), 1)
            ])

            return np.array(features)

        except Exception as e:
            logger.error(f"특성 수집 실패 ({coin}): {e}")
            return np.zeros(9)

    def learn_from_trade_outcome(self, coin, trade_action, entry_price, exit_price, duration_minutes):
        """거래 결과로부터 학습"""
        try:
            # 거래 성과 계산
            if trade_action == 'BUY':
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:
                return_pct = (entry_price - exit_price) / entry_price * 100

            # 학습 데이터 준비
            features = self.collect_real_time_features(coin)

            # 온라인 학습
            if coin in self.online_models and len(features) > 0:
                # 스케일러 업데이트
                features_scaled = self.online_scalers[coin].fit_transform(features.reshape(1, -1))

                # 모델 부분 학습
                self.online_models[coin].partial_fit(features_scaled, [return_pct])

                logger.info(f"💡 {coin} 거래 결과 학습: {return_pct:+.2f}% 수익")

                # 학습 버퍼에 추가
                self.learning_buffer.append({
                    'coin': coin,
                    'features': features,
                    'target': return_pct,
                    'timestamp': datetime.now(),
                    'trade_type': 'outcome'
                })

        except Exception as e:
            logger.error(f"거래 결과 학습 실패 ({coin}): {e}")

    def learn_from_prediction_feedback(self, coin, predicted_change, actual_change, prediction_confidence):
        """예측 피드백으로부터 학습"""
        try:
            # 예측 정확도 기록
            self.experience_collector.record_prediction(
                coin, predicted_change, actual_change, prediction_confidence, datetime.now()
            )

            # 성능 업데이트
            if coin in self.model_performance:
                perf = self.model_performance[coin]
                perf['predictions'] += 1

                if (predicted_change * actual_change) > 0:
                    perf['correct_directions'] += 1

                # 이동 평균으로 평균 정확도 업데이트
                accuracy = self.experience_collector._calculate_prediction_accuracy(predicted_change, actual_change)
                perf['avg_accuracy'] = (perf['avg_accuracy'] * 0.9) + (accuracy * 0.1)
                perf['last_update'] = datetime.now()

            # 오차가 큰 경우 추가 학습
            error = abs(predicted_change - actual_change)
            if error > 2.0:  # 2% 이상 오차
                features = self.collect_real_time_features(coin)

                if coin in self.online_models and len(features) > 0:
                    features_scaled = self.online_scalers[coin].transform(features.reshape(1, -1))

                    # 오차에 비례한 가중 학습
                    weight = min(error / 5.0, 2.0)
                    for _ in range(int(weight)):
                        self.online_models[coin].partial_fit(features_scaled, [actual_change])

                    logger.info(f"🔄 {coin} 예측 오차 교정 학습: {error:.2f}% 오차")

        except Exception as e:
            logger.error(f"예측 피드백 학습 실패 ({coin}): {e}")

    def learn_from_market_patterns(self):
        """시장 패턴 학습"""
        try:
            # 최근 시장 이벤트 분석
            recent_events = [e for e in self.experience_collector.market_events
                           if (datetime.now() - e['timestamp']).total_seconds() < 7200]  # 2시간

            if len(recent_events) < 2:
                return

            # 패턴 인식 및 학습
            for coin in self.online_models.keys():
                features = self.collect_real_time_features(coin)

                # 현재 가격 변화 계산
                ticker = self.upbit.GetTicker([f'KRW-{coin}'])
                if ticker:
                    current_change = ticker[0].change_rate * 100

                    # 패턴 기반 학습 데이터 추가
                    self.learning_buffer.append({
                        'coin': coin,
                        'features': features,
                        'target': current_change,
                        'timestamp': datetime.now(),
                        'trade_type': 'pattern',
                        'market_events': [e['event_type'] for e in recent_events]
                    })

            logger.info(f"📊 시장 패턴 학습: {len(recent_events)}개 이벤트 분석")

        except Exception as e:
            logger.error(f"시장 패턴 학습 실패: {e}")

    def continuous_learning_cycle(self):
        """지속적 학습 사이클"""
        logger.info("🔄 지속적 학습 사이클 시작")

        try:
            # 1. 학습 버퍼 데이터로 배치 학습
            if len(self.learning_buffer) > 50:
                self._batch_learning_from_buffer()

            # 2. 모델 성능 평가 및 조정
            self._evaluate_and_adjust_models()

            # 3. 시장 패턴 학습
            self.learn_from_market_patterns()

            # 4. 학습 상태 리포트
            self._report_learning_status()

        except Exception as e:
            logger.error(f"지속적 학습 실패: {e}")

    def _batch_learning_from_buffer(self):
        """학습 버퍼 데이터로 배치 학습"""
        try:
            coin_data = {}
            for record in list(self.learning_buffer):
                coin = record['coin']
                if coin not in coin_data:
                    coin_data[coin] = {'features': [], 'targets': []}

                coin_data[coin]['features'].append(record['features'])
                coin_data[coin]['targets'].append(record['target'])

            # 코인별 배치 학습
            for coin, data in coin_data.items():
                if coin in self.online_models and len(data['features']) > 10:
                    X = np.array(data['features'])
                    y = np.array(data['targets'])

                    # 스케일링
                    X_scaled = self.online_scalers[coin].fit_transform(X)

                    # 배치 학습
                    self.online_models[coin].partial_fit(X_scaled, y)

                    logger.info(f"📚 {coin} 배치 학습: {len(data['features'])}개 샘플")

            # 버퍼 정리
            self.learning_buffer.clear()

        except Exception as e:
            logger.error(f"배치 학습 실패: {e}")

    def _evaluate_and_adjust_models(self):
        """모델 성능 평가 및 조정"""
        for coin, performance in self.model_performance.items():
            try:
                accuracy = performance['avg_accuracy']
                correct_ratio = performance['correct_directions'] / max(performance['predictions'], 1)

                # 성능이 낮으면 학습률 조정
                if accuracy < 0.4 or correct_ratio < 0.45:
                    if coin in self.online_models:
                        self.online_models[coin].eta0 *= 1.1  # 학습률 증가
                        logger.warning(f"📈 {coin} 학습률 증가: 성능 개선 필요 (정확도: {accuracy:.2f})")

                elif accuracy > 0.7 and correct_ratio > 0.65:
                    if coin in self.online_models:
                        self.online_models[coin].eta0 *= 0.9  # 학습률 감소
                        logger.info(f"📉 {coin} 학습률 안정화: 좋은 성능 유지 (정확도: {accuracy:.2f})")

            except Exception as e:
                logger.error(f"{coin} 모델 평가 실패: {e}")

    def _report_learning_status(self):
        """학습 상태 리포트"""
        logger.info("\n🎓 학습 상태 리포트:")
        for coin, performance in self.model_performance.items():
            predictions = performance['predictions']
            correct = performance['correct_directions']
            accuracy = performance['avg_accuracy']

            if predictions > 0:
                logger.info(f"   {coin}: {predictions}번 예측, {correct}번 방향 맞춤 ({correct/predictions:.1%}), 평균 정확도 {accuracy:.2f}")
            else:
                logger.info(f"   {coin}: 예측 기록 없음")

    def get_enhanced_prediction(self, coin):
        """강화된 예측 (학습 경험 반영)"""
        try:
            if coin not in self.online_models:
                return None

            features = self.collect_real_time_features(coin)
            if len(features) == 0:
                return None

            # 온라인 모델 예측
            features_scaled = self.online_scalers[coin].transform(features.reshape(1, -1))
            prediction = self.online_models[coin].predict(features_scaled)[0]

            # 모델 성능 기반 신뢰도 조정
            performance = self.model_performance[coin]
            base_confidence = performance['avg_accuracy']

            # 최근 성과 기반 신뢰도 보정
            recent_predictions = [p for p in self.experience_collector.prediction_history
                                if p['coin'] == coin and
                                (datetime.now() - p['verification_time']).total_seconds() < 1800]  # 30분

            if recent_predictions:
                recent_accuracy = np.mean([p['accuracy'] for p in recent_predictions])
                confidence = (base_confidence * 0.7) + (recent_accuracy * 0.3)
            else:
                confidence = base_confidence

            return {
                'coin': coin,
                'predicted_change_pct': prediction,
                'confidence': confidence,
                'model_predictions': performance['predictions'],
                'model_accuracy': performance['avg_accuracy'],
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"{coin} 강화 예측 실패: {e}")
            return None

    def save_learning_state(self, filepath='models/adaptive_learning_state.pkl'):
        """학습 상태 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            state_data = {
                'online_models': self.online_models,
                'online_scalers': self.online_scalers,
                'model_performance': self.model_performance,
                'trade_history': list(self.experience_collector.trade_history)[-500:],  # 최근 500개
                'prediction_history': list(self.experience_collector.prediction_history)[-200:],
                'market_events': list(self.experience_collector.market_events)[-100:],
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)

            logger.info(f"💾 학습 상태 저장 완료: {filepath}")

        except Exception as e:
            logger.error(f"학습 상태 저장 실패: {e}")

    def load_learning_state(self, filepath='models/adaptive_learning_state.pkl'):
        """학습 상태 로드"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    state_data = pickle.load(f)

                self.online_models = state_data.get('online_models', {})
                self.online_scalers = state_data.get('online_scalers', {})
                self.model_performance = state_data.get('model_performance', {})
                self.learning_rate = state_data.get('learning_rate', 0.01)

                # 히스토리 복원
                if 'trade_history' in state_data:
                    self.experience_collector.trade_history.extend(state_data['trade_history'])
                if 'prediction_history' in state_data:
                    self.experience_collector.prediction_history.extend(state_data['prediction_history'])
                if 'market_events' in state_data:
                    self.experience_collector.market_events.extend(state_data['market_events'])

                logger.info(f"📂 학습 상태 로드 완료: {filepath}")
                return True
            else:
                logger.info("저장된 학습 상태가 없습니다. 새로 시작합니다.")
                return False

        except Exception as e:
            logger.error(f"학습 상태 로드 실패: {e}")
            return False

def main():
    """테스트 메인 함수"""
    print("🧠 적응형 학습 AI 테스트")
    print("=" * 50)

    try:
        ai = AdaptiveLearningEngine()
        ai.load_learning_state()
        ai.initialize_online_models()

        # 테스트 학습 사이클
        for i in range(5):
            logger.info(f"\n🔄 학습 사이클 #{i+1}")
            ai.continuous_learning_cycle()

            # 예측 테스트
            for coin in ['BTC', 'ETH']:
                prediction = ai.get_enhanced_prediction(coin)
                if prediction:
                    logger.info(f"🔮 {coin} 예측: {prediction['predicted_change_pct']:+.2f}% (신뢰도: {prediction['confidence']:.2f})")

            time.sleep(10)  # 10초 대기

        ai.save_learning_state()

    except KeyboardInterrupt:
        logger.info("🛑 사용자 중지")
    except Exception as e:
        logger.error(f"테스트 실패: {e}")

if __name__ == "__main__":
    main()