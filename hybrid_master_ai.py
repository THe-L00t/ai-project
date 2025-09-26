#!/usr/bin/env python3
"""
하이브리드 마스터 AI 트레이더
- 뉴스 감정 분석 + 패턴 학습 + 공격적 매매 + 적응형 학습
- 모든 AI 기능을 통합한 최강 트레이딩 시스템
"""

import os
import sys
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque
import pickle

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from exchange.UpbitAPI import UpbitAPI

# 컴포넌트 임포트
from news_sentiment_ai import NewsCollector, SentimentAnalyzer
from adaptive_learning_ai import AdaptiveLearningEngine

# 머신러닝
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/hybrid_master_ai.log')
    ]
)
logger = logging.getLogger(__name__)

class HybridMasterAI:
    """하이브리드 마스터 AI 트레이더"""

    def __init__(self):
        load_dotenv()

        # 업비트 API 초기화
        self.upbit = UpbitAPI()

        # 뉴스 분석 컴포넌트
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()

        # 적응형 학습 엔진
        self.adaptive_learner = AdaptiveLearningEngine()
        self.adaptive_learner.load_learning_state()
        self.adaptive_learner.initialize_online_models()

        # 거래 설정 (공격적)
        self.trading_mode = os.getenv('TRADING_MODE', 'live')
        self.max_position_size = 0.15  # 15% (공격적)
        self.stop_loss_percentage = 3.0  # 3% (공격적)
        self.take_profit_percentage = 15.0  # 15% (공격적)

        # 매매 임계값 (공격적)
        self.buy_threshold_change = 1.0  # 1%
        self.sell_threshold_change = -1.0  # -1%
        self.momentum_threshold = 0.5  # 0.5%

        # 대상 코인
        self.target_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']

        # 데이터 저장소
        self.news_history = deque(maxlen=1000)
        self.price_history = {}
        self.positions = {}
        self.last_prices = {}

        # 패턴 학습 모델
        self.pattern_models = {}
        self.pattern_scalers = {}

        # 예측 기록
        self.pending_predictions = deque(maxlen=1000)

        logger.info("🚀 하이브리드 마스터 AI 트레이더 초기화 완료")
        logger.info(f"💪 공격적 설정: 포지션 {self.max_position_size*100}%, 익절 {self.take_profit_percentage}%, 손절 {self.stop_loss_percentage}%")

    async def collect_news_and_sentiment(self):
        """뉴스 수집 및 감정 분석"""
        try:
            logger.info("📰 뉴스 수집 시작...")

            # 비동기 뉴스 수집
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            articles = loop.run_until_complete(
                self.news_collector.collect_news_async(max_articles=100)
            )
            loop.close()

            # 코인별 필터링
            filtered_news = self.news_collector.filter_crypto_news(articles)

            # 감정 분석
            coin_sentiments = {}
            for coin, news_list in filtered_news.items():
                if news_list:
                    sentiment_results = self.sentiment_analyzer.batch_analyze(news_list)
                    scores = [r['sentiment']['score'] for r in sentiment_results]
                    confidences = [r['sentiment']['confidence'] for r in sentiment_results]

                    coin_sentiments[coin] = {
                        'avg_sentiment': np.mean(scores) if scores else 0.0,
                        'sentiment_strength': np.std(scores) if len(scores) > 1 else 0.0,
                        'avg_confidence': np.mean(confidences) if confidences else 0.0,
                        'news_count': len(news_list),
                        'timestamp': datetime.now()
                    }

            # 히스토리에 저장
            self.news_history.append({
                'timestamp': datetime.now(),
                'articles_total': len(articles),
                'coin_sentiments': coin_sentiments
            })

            logger.info(f"✅ 뉴스 분석 완료: {len(articles)}개 기사")
            return coin_sentiments

        except Exception as e:
            logger.error(f"뉴스 분석 실패: {e}")
            return {}

    def create_hybrid_features(self, market):
        """하이브리드 특성 생성 (뉴스 + 패턴 + 기술지표)"""
        coin = market.split('-')[1]
        features = []

        try:
            # 1. 현재 가격 데이터
            ticker = self.upbit.GetTicker([market])
            if not ticker:
                return np.array([])

            current_price = ticker[0].trade_price
            change_rate = ticker[0].change_rate * 100
            volume = ticker[0].acc_trade_volume_24h

            # 2. 가격 히스토리 관리
            if market not in self.price_history:
                self.price_history[market] = deque(maxlen=20)

            self.price_history[market].append({
                'price': current_price,
                'timestamp': datetime.now(),
                'change_rate': change_rate,
                'volume': volume
            })

            # 3. 기술적 지표 특성
            if len(self.price_history[market]) >= 5:
                prices = [h['price'] for h in self.price_history[market]]
                volumes = [h['volume'] for h in self.price_history[market]]

                # 이동평균
                sma_5 = np.mean(prices[-5:])
                sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else sma_5

                # 가격 모멘텀
                price_momentum = (current_price - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0

                # 볼륨 추세
                volume_trend = (volume - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) * 100 if len(volumes) >= 5 else 0

                # RSI 근사치
                price_changes = np.diff(prices[-14:]) if len(prices) >= 14 else np.diff(prices)
                gains = price_changes[price_changes > 0]
                losses = -price_changes[price_changes < 0]
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0.1
                rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

                features.extend([
                    change_rate / 100,  # 정규화된 변동률
                    price_momentum / 100,  # 가격 모멘텀
                    (current_price - sma_5) / sma_5,  # SMA 대비 위치
                    (sma_5 - sma_10) / sma_10 if sma_10 > 0 else 0,  # SMA 추세
                    volume_trend / 100,  # 볼륨 추세
                    (rsi - 50) / 50,  # 정규화된 RSI
                ])
            else:
                features.extend([0.0] * 6)

            # 4. 뉴스 감정 특성
            recent_sentiment = 0.0
            sentiment_strength = 0.0
            news_count = 0

            if self.news_history and coin in self.news_history[-1]['coin_sentiments']:
                sentiment_data = self.news_history[-1]['coin_sentiments'][coin]
                recent_sentiment = sentiment_data['avg_sentiment']
                sentiment_strength = sentiment_data['sentiment_strength']
                news_count = sentiment_data['news_count']

            features.extend([
                recent_sentiment,
                sentiment_strength,
                min(news_count / 10, 1.0)  # 정규화된 뉴스 수
            ])

            # 5. 적응형 학습 특성
            adaptive_features = self.adaptive_learner.collect_real_time_features(coin)
            if len(adaptive_features) > 0:
                features.extend(adaptive_features[-3:])  # 마지막 3개만 사용
            else:
                features.extend([0.0] * 3)

            return np.array(features)

        except Exception as e:
            logger.error(f"하이브리드 특성 생성 실패 ({market}): {e}")
            return np.array([])

    def get_hybrid_signal(self, market):
        """하이브리드 매매 신호 생성 (모든 AI 결합)"""
        coin = market.split('-')[1]
        signals = []

        try:
            # 현재가 정보
            ticker = self.upbit.GetTicker([market])
            if not ticker:
                return 'HOLD', 0, []

            current_price = ticker[0].trade_price
            change_rate = ticker[0].change_rate * 100

            # 1. 공격적 패턴 신호
            pattern_signal = self.get_aggressive_pattern_signal(market, current_price, change_rate)
            if pattern_signal != 'HOLD':
                signals.append({
                    'source': 'aggressive_pattern',
                    'signal': pattern_signal,
                    'confidence': 0.7,
                    'reason': f'공격적 패턴: {change_rate:+.2f}%'
                })

            # 2. 뉴스 감정 신호
            if self.news_history and coin in self.news_history[-1]['coin_sentiments']:
                sentiment_data = self.news_history[-1]['coin_sentiments'][coin]
                sentiment_score = sentiment_data['avg_sentiment']

                if sentiment_score > 0.3:
                    signals.append({
                        'source': 'news_sentiment',
                        'signal': 'BUY',
                        'confidence': min(sentiment_score + 0.3, 1.0),
                        'reason': f'긍정 뉴스: {sentiment_score:.2f}'
                    })
                elif sentiment_score < -0.3:
                    signals.append({
                        'source': 'news_sentiment',
                        'signal': 'SELL',
                        'confidence': min(abs(sentiment_score) + 0.3, 1.0),
                        'reason': f'부정 뉴스: {sentiment_score:.2f}'
                    })

            # 3. 적응형 학습 신호
            if self.adaptive_learner:
                adaptive_prediction = self.adaptive_learner.get_enhanced_prediction(coin)
                if adaptive_prediction and adaptive_prediction['confidence'] > 0.5:
                    predicted_change = adaptive_prediction['predicted_change_pct']

                    if predicted_change > 2.0:
                        signals.append({
                            'source': 'adaptive_learning',
                            'signal': 'BUY',
                            'confidence': adaptive_prediction['confidence'],
                            'reason': f'AI 예측 상승: {predicted_change:+.2f}%'
                        })
                    elif predicted_change < -2.0:
                        signals.append({
                            'source': 'adaptive_learning',
                            'signal': 'SELL',
                            'confidence': adaptive_prediction['confidence'],
                            'reason': f'AI 예측 하락: {predicted_change:+.2f}%'
                        })

            # 4. 패턴 학습 모델 신호
            if coin in self.pattern_models:
                features = self.create_hybrid_features(market)
                if len(features) > 0:
                    try:
                        features_scaled = self.pattern_scalers[coin].transform(features.reshape(1, -1))
                        predicted_change = self.pattern_models[coin].predict(features_scaled)[0]

                        if predicted_change > 2.0:
                            signals.append({
                                'source': 'pattern_model',
                                'signal': 'BUY',
                                'confidence': 0.6,
                                'reason': f'패턴 예측 상승: {predicted_change:+.2f}%'
                            })
                        elif predicted_change < -2.0:
                            signals.append({
                                'source': 'pattern_model',
                                'signal': 'SELL',
                                'confidence': 0.6,
                                'reason': f'패턴 예측 하락: {predicted_change:+.2f}%'
                            })
                    except Exception as e:
                        logger.debug(f"패턴 모델 예측 실패: {e}")

            # 5. 신호 통합 (가중 투표)
            if not signals:
                return 'HOLD', current_price, []

            buy_weight = sum(s['confidence'] for s in signals if s['signal'] == 'BUY')
            sell_weight = sum(s['confidence'] for s in signals if s['signal'] == 'SELL')

            # 공격적 임계값 (낮은 기준으로 거래 활성화)
            min_confidence = 0.3

            if buy_weight > sell_weight and buy_weight >= min_confidence:
                return 'BUY', current_price, [s for s in signals if s['signal'] == 'BUY']
            elif sell_weight > buy_weight and sell_weight >= min_confidence:
                return 'SELL', current_price, [s for s in signals if s['signal'] == 'SELL']
            else:
                return 'HOLD', current_price, signals

        except Exception as e:
            logger.error(f"하이브리드 신호 생성 실패 ({market}): {e}")
            return 'HOLD', 0, []

    def get_aggressive_pattern_signal(self, market, current_price, change_rate):
        """공격적 패턴 신호 (기존 aggressive_trader 로직)"""
        if market not in self.price_history or len(self.price_history[market]) < 3:
            return 'HOLD'

        # 이전 가격과의 모멘텀 계산
        if market in self.last_prices:
            prev_price = self.last_prices[market]
            price_momentum = (current_price - prev_price) / prev_price * 100
        else:
            price_momentum = 0

        self.last_prices[market] = current_price

        # 단기 추세 분석 (최근 3개 가격)
        recent_prices = [h['price'] for h in list(self.price_history[market])[-3:]]
        short_term_trend = 0
        if len(recent_prices) >= 3:
            if recent_prices[-1] > recent_prices[0]:
                short_term_trend = 1  # 상승 추세
            elif recent_prices[-1] < recent_prices[0]:
                short_term_trend = -1  # 하락 추세

        # 공격적 매매 조건
        buy_conditions = [
            change_rate > self.buy_threshold_change,  # 1% 상승
            price_momentum > self.momentum_threshold,  # 0.5% 모멘텀
            short_term_trend >= 0  # 상승 또는 횡보
        ]

        sell_conditions = [
            change_rate < self.sell_threshold_change,  # -1% 하락
            price_momentum < -self.momentum_threshold,  # -0.5% 모멘텀
            short_term_trend <= 0  # 하락 또는 횡보
        ]

        # 2개 이상 조건 만족시 신호
        if sum(buy_conditions) >= 2:
            return 'BUY'
        elif sum(sell_conditions) >= 2 and market in self.positions:
            return 'SELL'
        else:
            return 'HOLD'

    def execute_hybrid_trade(self, market, signal, price, reasons):
        """하이브리드 거래 실행"""
        coin = market.split('-')[1]

        try:
            if signal == 'BUY' and market not in self.positions:
                # 매수 실행
                krw_balance = self.upbit.GetKRWBalance()
                buy_amount = krw_balance * self.max_position_size

                if buy_amount >= 5000:  # 최소 주문 금액
                    if self.trading_mode == 'live':
                        result = self.upbit.BuyMarket(market, buy_amount)
                        if result:
                            quantity = buy_amount / price
                            self.positions[market] = {
                                'type': 'long',
                                'quantity': quantity,
                                'entry_price': price,
                                'timestamp': datetime.now(),
                                'order_uuid': result['uuid'],
                                'reasons': reasons
                            }
                            logger.info(f"🔥 [실거래] 하이브리드 매수: {market} @ {price:,}원")
                    else:
                        # 모의거래
                        quantity = buy_amount / price
                        self.positions[market] = {
                            'type': 'long',
                            'quantity': quantity,
                            'entry_price': price,
                            'timestamp': datetime.now(),
                            'reasons': reasons
                        }
                        logger.info(f"🔥 [모의] 하이브리드 매수: {market} @ {price:,}원")

                    # 적응형 학습에 거래 기록
                    if self.adaptive_learner:
                        self.adaptive_learner.experience_collector.record_trade(
                            coin=coin,
                            action='BUY',
                            price=price,
                            quantity=quantity
                        )

                    # 이유 출력
                    for reason in reasons:
                        logger.info(f"   📊 {reason['source']}: {reason['reason']} (신뢰도: {reason['confidence']:.2f})")

                    return True

            elif signal == 'SELL' and market in self.positions:
                # 매도 실행
                position = self.positions[market]

                if self.trading_mode == 'live':
                    coin_balance = self.upbit.GetCoinBalance(coin)
                    if coin_balance > 0:
                        result = self.upbit.SellMarket(market, coin_balance)
                        if result:
                            pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                            logger.info(f"🔥 [실거래] 하이브리드 매도: {market} @ {price:,}원")
                            logger.info(f"   💰 손익: {pnl_pct:+.2f}%")
                else:
                    # 모의거래
                    pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                    logger.info(f"🔥 [모의] 하이브리드 매도: {market} @ {price:,}원")
                    logger.info(f"   💰 손익: {pnl_pct:+.2f}%")

                # 적응형 학습에 거래 결과 기록
                if self.adaptive_learner:
                    duration = (datetime.now() - position['timestamp']).total_seconds() / 60
                    self.adaptive_learner.learn_from_trade_outcome(
                        coin=coin,
                        trade_action='BUY',  # 원래 매수였던 것
                        entry_price=position['entry_price'],
                        exit_price=price,
                        duration_minutes=duration
                    )

                del self.positions[market]

                # 이유 출력
                for reason in reasons:
                    logger.info(f"   📊 {reason['source']}: {reason['reason']} (신뢰도: {reason['confidence']:.2f})")

                return True

            return False

        except Exception as e:
            logger.error(f"하이브리드 거래 실행 실패: {e}")
            return False

    def check_risk_management(self):
        """리스크 관리 (공격적 설정)"""
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
                    self.execute_hybrid_trade(market, 'SELL', current_price, [
                        {'source': 'risk_management', 'signal': 'SELL', 'confidence': 1.0, 'reason': f'손절매 {pnl_pct:.2f}%'}
                    ])

                # 익절매 체크
                elif pnl_pct >= self.take_profit_percentage:
                    logger.info(f"🎯 익절매 발동: {market} (+{pnl_pct:.2f}%)")
                    self.execute_hybrid_trade(market, 'SELL', current_price, [
                        {'source': 'risk_management', 'signal': 'SELL', 'confidence': 1.0, 'reason': f'익절매 +{pnl_pct:.2f}%'}
                    ])

            except Exception as e:
                logger.error(f"리스크 관리 오류 ({market}): {e}")

    def run_hybrid_training(self):
        """하이브리드 모델 빠른 훈련"""
        logger.info("🎓 하이브리드 패턴 모델 훈련 시작...")

        for market in self.target_coins:
            coin = market.split('-')[1]

            try:
                if len(self.price_history.get(market, [])) < 10:
                    logger.warning(f"{coin}: 훈련 데이터 부족")
                    continue

                # 특성 및 타겟 데이터 생성
                X = []
                y = []

                history = list(self.price_history[market])
                for i in range(5, len(history) - 1):  # 5개 이후부터 예측 가능
                    # 과거 데이터로 특성 생성 (임시로 현재 메소드 사용)
                    features = self.create_hybrid_features(market)
                    if len(features) > 0:
                        # 다음 시점의 변화율을 타겟으로
                        future_price = history[i + 1]['price']
                        current_price = history[i]['price']
                        change_pct = (future_price - current_price) / current_price * 100

                        X.append(features)
                        y.append(change_pct)

                if len(X) < 5:
                    continue

                X = np.array(X)
                y = np.array(y)

                # 모델 생성 및 훈련
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                scaler = StandardScaler()

                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)

                # 저장
                self.pattern_models[coin] = model
                self.pattern_scalers[coin] = scaler

                logger.info(f"✅ {coin} 하이브리드 모델 훈련 완료")

            except Exception as e:
                logger.error(f"{coin} 모델 훈련 실패: {e}")

    def run_hybrid_cycle(self):
        """하이브리드 매매 사이클 실행"""
        logger.info("🚀 하이브리드 마스터 AI 매매 시작!")
        logger.info("=" * 60)
        logger.info("🔥 공격적 설정 + 뉴스 감정 + 패턴 학습 + 적응형 AI")
        logger.info("=" * 60)

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                logger.info(f"\n🔄 하이브리드 매매 사이클 #{cycle_count}")

                # 1. 뉴스 수집 및 분석 (5분마다)
                if cycle_count % 5 == 1:
                    asyncio.run(self.collect_news_and_sentiment())

                # 2. 패턴 모델 재훈련 (10분마다)
                if cycle_count % 10 == 1:
                    self.run_hybrid_training()

                # 3. 현재 잔고 확인
                krw_balance = self.upbit.GetKRWBalance()
                logger.info(f"💰 원화 잔고: {krw_balance:,.0f}원")

                # 4. 각 코인별 하이브리드 신호 확인 및 거래
                for market in self.target_coins:
                    try:
                        signal, price, reasons = self.get_hybrid_signal(market)

                        if signal != 'HOLD':
                            logger.info(f"🎯 {market}: {signal} 신호 (가격: {price:,}원)")
                            self.execute_hybrid_trade(market, signal, price, reasons)
                        else:
                            logger.info(f"⏸️  {market}: HOLD (가격: {price:,}원)")

                    except Exception as e:
                        logger.error(f"{market} 처리 오류: {e}")

                # 5. 리스크 관리
                self.check_risk_management()

                # 6. 적응형 학습 업데이트 (15분마다)
                if cycle_count % 15 == 0 and self.adaptive_learner:
                    self.adaptive_learner.continuous_learning_cycle()
                    self.adaptive_learner.save_learning_state()

                # 7. 현재 포지션 상태 출력
                if self.positions:
                    logger.info("📊 현재 포지션:")
                    for market, pos in self.positions.items():
                        ticker = self.upbit.GetTicker([market])
                        if ticker:
                            current_price = ticker[0].trade_price
                            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                            duration = datetime.now() - pos['timestamp']
                            logger.info(f"   {market}: {pos['quantity']:.8f} ({pnl_pct:+.2f}%, {duration})")
                else:
                    logger.info("📊 현재 포지션: 없음")

                # 8. 20초 대기 (공격적 사이클)
                logger.info("⏱️  20초 대기...")
                time.sleep(20)

        except KeyboardInterrupt:
            logger.info("\\n🛑 사용자에 의한 정지")
        except Exception as e:
            logger.error(f"❌ 치명적 오류: {e}")
        finally:
            # 학습 상태 저장
            if self.adaptive_learner:
                self.adaptive_learner.save_learning_state()
            logger.info("🏁 하이브리드 마스터 AI 매매 종료")

def main():
    """메인 함수"""
    print("🚀 Hybrid Master AI Trader")
    print("=" * 50)
    print("🔥 공격적 매매 + 뉴스 감정 + 패턴 학습 + 적응형 AI")
    print("=" * 50)

    try:
        ai = HybridMasterAI()
        ai.run_hybrid_cycle()
    except Exception as e:
        logger.error(f"하이브리드 AI 실행 실패: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())