#!/usr/bin/env python3
"""
코인 가격 변동 패턴 학습 AI
- 실시간 데이터 수집 및 패턴 분석
- 머신러닝을 통한 가격 예측 모델 훈련
- 패턴 인식 및 트레이딩 신호 생성
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque
import pickle
import json

# 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from exchange.UpbitAPI import UpbitAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pattern_learning.log')
    ]
)
logger = logging.getLogger(__name__)

class PricePredictor(nn.Module):
    """LSTM 기반 가격 예측 뉴럴 네트워크"""

    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=1):
        super(PricePredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class PatternLearningAI:
    """코인 가격 변동 패턴 학습 AI"""

    def __init__(self):
        load_dotenv()

        # 업비트 API 초기화
        self.upbit = UpbitAPI()

        # 학습 대상 코인
        self.target_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']

        # 데이터 저장소
        self.price_history = {}  # 가격 히스토리
        self.features_history = {}  # 특성 히스토리
        self.patterns_db = {}  # 패턴 데이터베이스

        # 학습 설정
        self.sequence_length = 50  # LSTM 시퀀스 길이
        self.feature_window = 20  # 특성 계산 윈도우
        self.update_interval = 30  # 데이터 업데이트 간격 (초)

        # 모델 저장소
        self.models = {}
        self.scalers = {}

        # 성능 지표
        self.model_performance = {}

        logger.info("🧠 PatternLearningAI 초기화 완료")

    def load_historical_data(self):
        """과거 데이터 로드 (업비트 API에서 대용량 캔들 데이터)"""
        logger.info("📚 과거 데이터 로딩 중...")

        for market in self.target_coins:
            try:
                # 1시간봉 데이터 (최근 30일)
                hourly_candles = self.upbit.GetCandles(market, 'minutes', 60, 200)  # 최대 200개

                if hourly_candles:
                    logger.info(f"📊 {market} 과거 데이터 로드: {len(hourly_candles)}개 1시간봉")

                    if market not in self.price_history:
                        self.price_history[market] = deque(maxlen=2000)  # 용량 확대

                    # 과거 데이터를 현재 형식으로 변환하여 추가
                    for candle in reversed(hourly_candles):  # 오래된 것부터
                        historical_data = {
                            'timestamp': datetime.fromisoformat(candle['candle_date_time_kst'].replace('T', ' ')),
                            'price': candle['trade_price'],
                            'volume': candle['candle_acc_trade_volume'],
                            'change_rate': ((candle['trade_price'] - candle['opening_price']) / candle['opening_price']) * 100,
                            'high_price': candle['high_price'],
                            'low_price': candle['low_price'],
                            'candles': [{
                                'open': candle['opening_price'],
                                'high': candle['high_price'],
                                'low': candle['low_price'],
                                'close': candle['trade_price'],
                                'volume': candle['candle_acc_trade_volume']
                            }]
                        }
                        self.price_history[market].append(historical_data)

            except Exception as e:
                logger.error(f"과거 데이터 로드 실패 ({market}): {e}")

    def collect_real_time_data(self):
        """실시간 데이터 수집"""
        for market in self.target_coins:
            try:
                # 현재 시세 정보
                ticker = self.upbit.GetTicker([market])
                if not ticker:
                    continue

                current_data = {
                    'timestamp': datetime.now(),
                    'price': ticker[0].trade_price,
                    'volume': ticker[0].acc_trade_volume_24h,
                    'change_rate': ticker[0].change_rate * 100,
                    'high_price': ticker[0].prev_closing_price * (1 + ticker[0].change_rate),
                    'low_price': ticker[0].prev_closing_price * (1 + ticker[0].change_rate)
                }

                # 캔들 데이터 (1분봉 최근 100개)
                candles = self.upbit.GetCandles(market, 'minutes', 1, 100)
                if candles:
                    # 최신 캔들 데이터 추가
                    candle_data = []
                    for candle in candles[:10]:  # 최근 10개만 사용
                        candle_data.append({
                            'open': candle['opening_price'],
                            'high': candle['high_price'],
                            'low': candle['low_price'],
                            'close': candle['trade_price'],
                            'volume': candle['candle_acc_trade_volume']
                        })
                    current_data['candles'] = candle_data

                # 히스토리에 저장
                if market not in self.price_history:
                    self.price_history[market] = deque(maxlen=1000)

                self.price_history[market].append(current_data)

                logger.info(f"📊 {market} 데이터 수집: {current_data['price']:,}원 ({current_data['change_rate']:+.2f}%)")

            except Exception as e:
                logger.error(f"데이터 수집 오류 ({market}): {e}")

    def extract_technical_features(self, market):
        """기술적 지표 특성 추출"""
        if market not in self.price_history or len(self.price_history[market]) < self.feature_window:
            return None

        try:
            # 가격 데이터 추출
            prices = [data['price'] for data in list(self.price_history[market])[-self.feature_window:]]
            volumes = [data['volume'] for data in list(self.price_history[market])[-self.feature_window:]]

            prices_array = np.array(prices)
            volumes_array = np.array(volumes)

            features = {}

            # 이동평균
            features['sma_5'] = np.mean(prices_array[-5:])
            features['sma_10'] = np.mean(prices_array[-10:])
            features['sma_20'] = np.mean(prices_array[-20:]) if len(prices_array) >= 20 else np.mean(prices_array)

            # 변동성 지표
            features['volatility'] = np.std(prices_array[-10:])
            features['price_change'] = (prices_array[-1] - prices_array[-2]) / prices_array[-2] * 100

            # RSI 계산 (단순화)
            price_changes = np.diff(prices_array[-14:])
            gains = price_changes[price_changes > 0]
            losses = -price_changes[price_changes < 0]

            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100

            # 볼린저 밴드
            sma_20 = features['sma_20']
            std_20 = np.std(prices_array[-20:]) if len(prices_array) >= 20 else np.std(prices_array)
            features['bb_upper'] = sma_20 + (2 * std_20)
            features['bb_lower'] = sma_20 - (2 * std_20)
            features['bb_position'] = (prices_array[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # 거래량 지표
            features['volume_avg'] = np.mean(volumes_array[-5:])
            features['volume_ratio'] = volumes_array[-1] / features['volume_avg'] if features['volume_avg'] > 0 else 1

            return features

        except Exception as e:
            logger.error(f"특성 추출 오류 ({market}): {e}")
            return None

    def detect_patterns(self, market):
        """가격 패턴 감지"""
        if market not in self.price_history or len(self.price_history[market]) < 20:
            return None

        try:
            # 최근 20개 가격 데이터
            recent_data = list(self.price_history[market])[-20:]
            prices = [data['price'] for data in recent_data]

            patterns = {}

            # 추세 패턴
            price_changes = np.diff(prices)

            # 상승 추세
            if np.sum(price_changes > 0) >= 15:  # 20개 중 15개 이상 상승
                patterns['trend'] = 'bullish'
                patterns['trend_strength'] = np.sum(price_changes > 0) / len(price_changes)
            # 하락 추세
            elif np.sum(price_changes < 0) >= 15:  # 20개 중 15개 이상 하락
                patterns['trend'] = 'bearish'
                patterns['trend_strength'] = np.sum(price_changes < 0) / len(price_changes)
            else:
                patterns['trend'] = 'sideways'
                patterns['trend_strength'] = 0.5

            # 지지/저항선 패턴
            recent_highs = []
            recent_lows = []

            for i in range(2, len(prices)-2):
                # 지역 최고점
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    recent_highs.append(prices[i])
                # 지역 최저점
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    recent_lows.append(prices[i])

            if recent_highs:
                patterns['resistance'] = np.mean(recent_highs)
            if recent_lows:
                patterns['support'] = np.mean(recent_lows)

            # 반복 패턴 감지 (단순 주기성)
            if len(prices) >= 10:
                # 가격을 정규화하고 주기성 체크
                normalized_prices = (np.array(prices) - np.mean(prices)) / np.std(prices)

                # 5분 주기 패턴 체크
                cycle_5 = 0
                for i in range(5, len(normalized_prices)):
                    if abs(normalized_prices[i] - normalized_prices[i-5]) < 0.5:
                        cycle_5 += 1

                if cycle_5 >= 3:
                    patterns['cycle_5min'] = True
                    patterns['cycle_strength'] = cycle_5 / (len(normalized_prices) - 5)

            return patterns

        except Exception as e:
            logger.error(f"패턴 감지 오류 ({market}): {e}")
            return None

    def prepare_training_data(self, market):
        """모델 훈련 데이터 준비"""
        if market not in self.price_history or len(self.price_history[market]) < self.sequence_length + 10:
            return None, None

        try:
            # 특성 데이터 수집
            features_list = []
            prices_list = []

            data_points = list(self.price_history[market])

            for i in range(len(data_points) - 1):
                # 현재까지의 가격을 특성으로 사용
                if i >= self.feature_window:
                    current_features = self.extract_technical_features_at_point(data_points[:i+1])
                    next_price = data_points[i+1]['price']

                    if current_features:
                        features_list.append(list(current_features.values()))
                        prices_list.append(next_price)

            if len(features_list) < 10:
                return None, None

            X = np.array(features_list)
            y = np.array(prices_list)

            return X, y

        except Exception as e:
            logger.error(f"훈련 데이터 준비 오류 ({market}): {e}")
            return None, None

    def extract_technical_features_at_point(self, data_points):
        """특정 시점에서의 기술적 지표 계산"""
        if len(data_points) < 5:
            return None

        try:
            prices = [data['price'] for data in data_points[-20:]]
            prices_array = np.array(prices)

            features = {}

            # 기본 통계
            features['current_price'] = prices_array[-1]
            features['price_change_1'] = (prices_array[-1] - prices_array[-2]) / prices_array[-2] * 100 if len(prices_array) >= 2 else 0
            features['price_change_5'] = (prices_array[-1] - prices_array[-6]) / prices_array[-6] * 100 if len(prices_array) >= 6 else 0

            # 이동평균
            features['sma_5'] = np.mean(prices_array[-5:])
            features['sma_10'] = np.mean(prices_array[-10:]) if len(prices_array) >= 10 else np.mean(prices_array)

            # 변동성
            features['volatility_5'] = np.std(prices_array[-5:])
            features['volatility_10'] = np.std(prices_array[-10:]) if len(prices_array) >= 10 else np.std(prices_array)

            # 최고/최저가
            features['high_5'] = np.max(prices_array[-5:])
            features['low_5'] = np.min(prices_array[-5:])

            # 현재 가격의 상대적 위치
            features['price_position'] = (prices_array[-1] - features['low_5']) / (features['high_5'] - features['low_5']) if features['high_5'] != features['low_5'] else 0.5

            return features

        except Exception as e:
            logger.error(f"시점별 특성 추출 오류: {e}")
            return None

    def train_models(self, market):
        """모델 훈련"""
        logger.info(f"🎓 {market} 모델 훈련 시작...")

        X, y = self.prepare_training_data(market)
        if X is None or len(X) < 20:
            logger.warning(f"{market}: 훈련 데이터 부족 (필요: 20개 이상, 현재: {len(X) if X is not None else 0}개)")
            return

        try:
            # 데이터 정규화
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 훈련/테스트 분할 (80:20)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Random Forest 모델
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # 예측 및 평가
            y_pred = rf_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # 정확도 계산 (방향 예측 정확도)
            price_changes_actual = np.diff(y_test)
            price_changes_pred = np.diff(y_pred)

            direction_correct = np.sum((price_changes_actual > 0) == (price_changes_pred > 0))
            direction_accuracy = direction_correct / len(price_changes_actual) if len(price_changes_actual) > 0 else 0

            # 모델 저장
            self.models[market] = rf_model
            self.scalers[market] = scaler

            # 성능 기록
            self.model_performance[market] = {
                'mse': mse,
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'training_samples': len(X_train),
                'last_trained': datetime.now().isoformat()
            }

            logger.info(f"✅ {market} 모델 훈련 완료:")
            logger.info(f"   방향 정확도: {direction_accuracy:.2%}")
            logger.info(f"   평균 절대 오차: {mae:,.0f}원")
            logger.info(f"   훈련 샘플: {len(X_train)}개")

        except Exception as e:
            logger.error(f"{market} 모델 훈련 실패: {e}")

    def predict_price(self, market, horizon_minutes=5):
        """가격 예측"""
        if market not in self.models or market not in self.scalers:
            return None

        try:
            # 현재 특성 추출
            current_features = self.extract_technical_features(market)
            if not current_features:
                return None

            # 특성 벡터 생성
            feature_vector = np.array(list(current_features.values())).reshape(1, -1)

            # 정규화
            feature_vector_scaled = self.scalers[market].transform(feature_vector)

            # 예측
            predicted_price = self.models[market].predict(feature_vector_scaled)[0]

            # 현재 가격과 비교
            current_price = list(self.price_history[market])[-1]['price']
            price_change = (predicted_price - current_price) / current_price * 100

            prediction = {
                'market': market,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change,
                'horizon_minutes': horizon_minutes,
                'confidence': self.model_performance[market]['direction_accuracy'],
                'timestamp': datetime.now()
            }

            return prediction

        except Exception as e:
            logger.error(f"{market} 가격 예측 오류: {e}")
            return None

    def generate_trading_signals(self):
        """학습된 패턴을 바탕으로 거래 신호 생성"""
        signals = {}

        for market in self.target_coins:
            try:
                # 가격 예측
                prediction = self.predict_price(market)
                if not prediction:
                    continue

                # 패턴 감지
                patterns = self.detect_patterns(market)
                if not patterns:
                    continue

                # 기술적 지표
                features = self.extract_technical_features(market)
                if not features:
                    continue

                # 신호 생성 로직
                signal = 'HOLD'
                confidence = 0.5
                reasons = []

                # 예측 기반 신호
                if prediction['price_change_pct'] > 2.0 and prediction['confidence'] > 0.6:
                    signal = 'BUY'
                    confidence = prediction['confidence']
                    reasons.append(f"가격 상승 예측 ({prediction['price_change_pct']:+.1f}%)")
                elif prediction['price_change_pct'] < -2.0 and prediction['confidence'] > 0.6:
                    signal = 'SELL'
                    confidence = prediction['confidence']
                    reasons.append(f"가격 하락 예측 ({prediction['price_change_pct']:+.1f}%)")

                # 패턴 기반 보정
                if patterns.get('trend') == 'bullish' and patterns.get('trend_strength', 0) > 0.7:
                    if signal == 'BUY':
                        confidence += 0.1
                        reasons.append("강한 상승 추세")
                    elif signal == 'SELL':
                        confidence -= 0.2
                        signal = 'HOLD'
                        reasons.append("상승 추세와 충돌")

                elif patterns.get('trend') == 'bearish' and patterns.get('trend_strength', 0) > 0.7:
                    if signal == 'SELL':
                        confidence += 0.1
                        reasons.append("강한 하락 추세")
                    elif signal == 'BUY':
                        confidence -= 0.2
                        signal = 'HOLD'
                        reasons.append("하락 추세와 충돌")

                # RSI 기반 보정
                rsi = features.get('rsi', 50)
                if rsi > 70:  # 과매수
                    if signal == 'BUY':
                        confidence -= 0.1
                        reasons.append("RSI 과매수 구간")
                elif rsi < 30:  # 과매도
                    if signal == 'SELL':
                        confidence -= 0.1
                        reasons.append("RSI 과매도 구간")

                # 신뢰도 제한
                confidence = max(0.0, min(1.0, confidence))

                signals[market] = {
                    'signal': signal,
                    'confidence': confidence,
                    'reasons': reasons,
                    'prediction': prediction,
                    'patterns': patterns,
                    'features': features,
                    'timestamp': datetime.now()
                }

            except Exception as e:
                logger.error(f"{market} 신호 생성 오류: {e}")
                continue

        return signals

    def save_models(self):
        """학습된 모델 저장"""
        try:
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)

            # 모델 저장
            for market, model in self.models.items():
                model_path = os.path.join(models_dir, f'{market}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                scaler_path = os.path.join(models_dir, f'{market}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[market], f)

            # 성능 지표 저장
            performance_path = os.path.join(models_dir, 'model_performance.json')
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)

            logger.info("💾 모델 저장 완료")

        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

    def run_learning_mode(self):
        """학습 모드 실행"""
        logger.info("🧠 코인 가격 패턴 학습 AI 시작!")
        logger.info(f"학습 대상: {self.target_coins}")
        logger.info(f"데이터 수집 간격: {self.update_interval}초")

        # 과거 데이터 로드
        self.load_historical_data()

        # 과거 데이터로 초기 모델 훈련
        logger.info("🎓 과거 데이터로 초기 모델 훈련...")
        for market in self.target_coins:
            if market in self.price_history and len(self.price_history[market]) >= 50:
                self.train_models(market)

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                logger.info(f"\n🔄 학습 사이클 #{cycle_count}")

                # 1. 실시간 데이터 수집
                logger.info("📊 실시간 데이터 수집 중...")
                self.collect_real_time_data()

                # 2. 충분한 데이터가 쌓이면 모델 훈련
                if cycle_count % 10 == 0:  # 10사이클마다 재훈련
                    logger.info("🎓 모델 재훈련 시작...")
                    for market in self.target_coins:
                        if market in self.price_history and len(self.price_history[market]) >= 50:
                            self.train_models(market)

                    # 모델 저장
                    self.save_models()

                # 3. 패턴 감지 및 신호 생성
                if len(self.models) > 0:
                    logger.info("🎯 거래 신호 생성 중...")
                    signals = self.generate_trading_signals()

                    for market, signal_data in signals.items():
                        signal = signal_data['signal']
                        confidence = signal_data['confidence']
                        reasons = ', '.join(signal_data['reasons'])

                        if signal != 'HOLD':
                            logger.info(f"🚨 {market}: {signal} (신뢰도: {confidence:.2%})")
                            logger.info(f"   이유: {reasons}")
                        else:
                            logger.info(f"⏸️  {market}: HOLD (신뢰도: {confidence:.2%})")

                # 4. 데이터 현황 출력
                logger.info("📈 수집된 데이터 현황:")
                for market in self.target_coins:
                    if market in self.price_history:
                        data_count = len(self.price_history[market])
                        latest_price = list(self.price_history[market])[-1]['price'] if data_count > 0 else 0
                        logger.info(f"   {market}: {data_count}개 ({latest_price:,}원)")

                # 5. 모델 성능 출력
                if self.model_performance:
                    logger.info("🎯 모델 성능:")
                    for market, perf in self.model_performance.items():
                        accuracy = perf['direction_accuracy']
                        samples = perf['training_samples']
                        logger.info(f"   {market}: 방향 정확도 {accuracy:.2%} (샘플: {samples}개)")

                # 대기
                logger.info(f"⏱️  {self.update_interval}초 대기...")
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            logger.info("\n🛑 사용자에 의한 학습 중지")
        except Exception as e:
            logger.error(f"❌ 학습 오류: {e}")
        finally:
            # 최종 모델 저장
            self.save_models()
            logger.info("🏁 패턴 학습 AI 종료")

def main():
    """메인 함수"""
    print("🧠 Pattern Learning AI")
    print("=" * 50)
    print("코인 가격 변동 패턴을 학습하고 예측 모델을 훈련합니다.")
    print("Ctrl+C로 중지할 수 있습니다.")
    print()

    try:
        ai = PatternLearningAI()
        ai.run_learning_mode()
    except Exception as e:
        logger.error(f"AI 실행 실패: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())