#!/usr/bin/env python3
"""
뉴스 기반 코인 예측 AI
- 전 세계 뉴스 수집 및 감정 분석
- 코인 관련 뉴스 필터링
- 감정 점수를 통한 가격 변동 예측
- 1시간 내 학습 완료 최적화
"""

import os
import sys
import time
import logging
import asyncio
import aiohttp
import feedparser
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import pickle
import json

# NLP 라이브러리
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk

# 머신러닝
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from exchange.UpbitAPI import UpbitAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/news_sentiment_ai.log')
    ]
)
logger = logging.getLogger(__name__)

class NewsCollector:
    """전 세계 뉴스 수집기"""

    def __init__(self):
        # 크립토 관련 RSS 피드들
        self.crypto_feeds = [
            'https://cointelegraph.com/rss',
            'https://cryptonews.com/news/feed/',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://decrypt.co/feed',
            'https://bitcoinmagazine.com/.rss/full/',
            'https://cryptobriefing.com/feed/',
            'https://www.crypto-news.net/feed/',
            'https://ambcrypto.com/feed/',
            'https://cryptopotato.com/feed/',
            'https://u.today/rss'
        ]

        # 일반 경제/기술 뉴스
        self.general_feeds = [
            'https://feeds.reuters.com/reuters/businessNews',
            'https://rss.cnn.com/rss/money_news_economy.rss',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://rss.cbc.ca/lineup/business.xml',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html'
        ]

        # 코인별 키워드
        self.coin_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi', 'digital gold'],
            'ETH': ['ethereum', 'eth', 'vitalik', 'smart contract', 'defi'],
            'ADA': ['cardano', 'ada', 'charles hoskinson', 'proof of stake'],
            'DOT': ['polkadot', 'dot', 'parachain', 'substrate']
        }

    async def fetch_feed(self, session, url):
        """RSS 피드 비동기 수집 (SSL 문제 해결)"""
        try:
            # SSL 검증 비활성화
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as ssl_session:
                async with ssl_session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        return feedparser.parse(content)
        except Exception as e:
            logger.error(f"피드 수집 실패 ({url}): {e}")
            # 백업: requests 사용
            try:
                import requests
                requests.packages.urllib3.disable_warnings()
                response = requests.get(url, timeout=10, verify=False)
                if response.status_code == 200:
                    return feedparser.parse(response.content)
            except:
                pass
            return None

    async def collect_news_async(self, max_articles=200):
        """비동기 뉴스 수집 (최적화)"""
        logger.info("🌍 전 세계 뉴스 수집 시작...")
        start_time = time.time()

        all_feeds = self.crypto_feeds + self.general_feeds
        articles = []

        # SSL 검증 비활성화
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_feed(session, url) for url in all_feeds]

            for i, task in enumerate(asyncio.as_completed(tasks)):
                try:
                    feed = await task
                    if feed and hasattr(feed, 'entries'):
                        for entry in feed.entries[:20]:  # 피드당 최대 20개
                            if len(articles) >= max_articles:
                                break

                            article = {
                                'title': entry.get('title', ''),
                                'description': entry.get('description', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'source': all_feeds[i % len(all_feeds)],
                                'timestamp': datetime.now()
                            }
                            articles.append(article)

                        logger.info(f"📰 피드 {i+1}/{len(all_feeds)} 처리 완료 ({len(feed.entries)}개 기사)")

                    if len(articles) >= max_articles:
                        break

                except Exception as e:
                    logger.error(f"피드 처리 오류: {e}")

        elapsed = time.time() - start_time
        logger.info(f"✅ 뉴스 수집 완료: {len(articles)}개 기사 ({elapsed:.1f}초)")

        return articles

    def filter_crypto_news(self, articles):
        """코인 관련 뉴스 필터링"""
        filtered_news = {coin: [] for coin in self.coin_keywords.keys()}

        for article in articles:
            text = (article['title'] + ' ' + article['description']).lower()

            for coin, keywords in self.coin_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        filtered_news[coin].append(article)
                        break

        logger.info("🔍 코인별 뉴스 필터링 완료:")
        for coin, news_list in filtered_news.items():
            logger.info(f"   {coin}: {len(news_list)}개 기사")

        return filtered_news

class SentimentAnalyzer:
    """감정 분석 엔진"""

    def __init__(self):
        # 경량 감정 분석 모델 사용 (1시간 제한을 위해)
        try:
            # FinBERT 대신 더 빠른 모델 사용
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True,
                max_length=512
            )
            logger.info("🧠 고급 감정 분석 모델 로드 완료")
        except Exception as e:
            logger.warning(f"고급 모델 로드 실패, TextBlob 사용: {e}")
            self.sentiment_pipeline = None

        # 크립토 특화 키워드 가중치
        self.crypto_weights = {
            'positive': ['bullish', 'moon', 'pump', 'rally', 'surge', 'adoption', 'breakthrough'],
            'negative': ['bearish', 'crash', 'dump', 'ban', 'hack', 'regulation', 'selloff'],
            'neutral': ['stable', 'sideways', 'consolidation']
        }

    def analyze_sentiment_fast(self, text):
        """빠른 감정 분석 (1시간 제한 고려)"""
        if not text or len(text.strip()) < 10:
            return {'score': 0.0, 'confidence': 0.0}

        try:
            if self.sentiment_pipeline:
                # Transformer 모델 사용 (더 정확함)
                result = self.sentiment_pipeline(text[:500])  # 길이 제한

                label = result[0]['label'].upper()
                confidence = result[0]['score']

                if 'POSITIVE' in label:
                    score = confidence
                elif 'NEGATIVE' in label:
                    score = -confidence
                else:
                    score = 0.0

            else:
                # TextBlob 대안 사용
                blob = TextBlob(text)
                score = blob.sentiment.polarity
                confidence = abs(blob.sentiment.subjectivity)

            # 크립토 특화 키워드 가중치 적용
            text_lower = text.lower()
            for pos_word in self.crypto_weights['positive']:
                if pos_word in text_lower:
                    score += 0.1

            for neg_word in self.crypto_weights['negative']:
                if neg_word in text_lower:
                    score -= 0.1

            # 점수 정규화
            score = max(-1.0, min(1.0, score))

            return {
                'score': float(score),
                'confidence': float(confidence),
                'method': 'transformer' if self.sentiment_pipeline else 'textblob'
            }

        except Exception as e:
            logger.error(f"감정 분석 오류: {e}")
            return {'score': 0.0, 'confidence': 0.0}

    def batch_analyze(self, articles, max_workers=4):
        """배치 감정 분석 (병렬 처리로 최적화)"""
        logger.info(f"🎭 {len(articles)}개 기사 감정 분석 시작...")

        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 텍스트 준비
            texts = []
            for article in articles:
                text = article['title'] + ' ' + article['description']
                texts.append(text[:1000])  # 길이 제한

            # 병렬 처리
            future_to_idx = {
                executor.submit(self.analyze_sentiment_fast, text): i
                for i, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    sentiment = future.result()
                    results.append({
                        'article': articles[idx],
                        'sentiment': sentiment
                    })
                except Exception as e:
                    logger.error(f"기사 {idx} 감정 분석 실패: {e}")
                    results.append({
                        'article': articles[idx],
                        'sentiment': {'score': 0.0, 'confidence': 0.0}
                    })

        elapsed = time.time() - start_time
        logger.info(f"✅ 감정 분석 완료 ({elapsed:.1f}초)")

        return results

class NewsSentimentAI:
    """뉴스 기반 코인 예측 AI (1시간 학습 최적화)"""

    def __init__(self):
        load_dotenv()

        # 컴포넌트 초기화
        self.upbit = UpbitAPI()
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()

        # 적응형 학습 시스템 추가
        try:
            from adaptive_learning_ai import AdaptiveLearningEngine
            self.adaptive_learner = AdaptiveLearningEngine()
            self.adaptive_learner.load_learning_state()
            self.adaptive_learner.initialize_online_models()
            self.adaptive_learning_enabled = True
            logger.info("🧠 적응형 학습 시스템 연동 완료")
        except Exception as e:
            logger.warning(f"적응형 학습 시스템 연동 실패: {e}")
            self.adaptive_learner = None
            self.adaptive_learning_enabled = False

        # 데이터 저장소
        self.news_history = deque(maxlen=1000)
        self.sentiment_history = {}
        self.price_correlation = {}

        # 모델
        self.prediction_models = {}
        self.scalers = {}

        # 1시간 학습 최적화 설정
        self.max_learning_time = 3600  # 1시간
        self.quick_mode = True  # 빠른 학습 모드

        logger.info("🌍 뉴스 기반 코인 예측 AI 초기화 완료")

    def collect_and_analyze_news(self):
        """뉴스 수집 및 분석"""
        start_time = time.time()

        # 1. 뉴스 수집 (비동기, 최대 5분)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        articles = loop.run_until_complete(
            self.news_collector.collect_news_async(max_articles=150)
        )
        loop.close()

        # 2. 코인별 필터링
        filtered_news = self.news_collector.filter_crypto_news(articles)

        # 3. 감정 분석 (병렬 처리, 최대 10분)
        coin_sentiments = {}
        for coin, news_list in filtered_news.items():
            if news_list:
                sentiment_results = self.sentiment_analyzer.batch_analyze(news_list)

                # 평균 감정 점수 계산
                scores = [r['sentiment']['score'] for r in sentiment_results]
                confidences = [r['sentiment']['confidence'] for r in sentiment_results]

                coin_sentiments[coin] = {
                    'avg_sentiment': np.mean(scores) if scores else 0.0,
                    'sentiment_strength': np.std(scores) if len(scores) > 1 else 0.0,
                    'avg_confidence': np.mean(confidences) if confidences else 0.0,
                    'news_count': len(news_list),
                    'timestamp': datetime.now()
                }

        # 4. 히스토리에 저장
        news_data = {
            'timestamp': datetime.now(),
            'articles_total': len(articles),
            'coin_sentiments': coin_sentiments
        }
        self.news_history.append(news_data)

        elapsed = time.time() - start_time
        logger.info(f"📊 뉴스 분석 완료 ({elapsed:.1f}초)")

        return coin_sentiments

    def create_prediction_features(self, coin, hours_back=24):
        """예측용 특성 생성 (최적화)"""
        features = []

        # 1. 뉴스 감정 특성 (최근 24시간)
        recent_news = [n for n in self.news_history if
                      (datetime.now() - n['timestamp']).total_seconds() < hours_back * 3600]

        if recent_news and coin in recent_news[-1]['coin_sentiments']:
            recent_sentiments = []
            news_counts = []

            for news_data in recent_news:
                if coin in news_data['coin_sentiments']:
                    sentiment_info = news_data['coin_sentiments'][coin]
                    recent_sentiments.append(sentiment_info['avg_sentiment'])
                    news_counts.append(sentiment_info['news_count'])

            if recent_sentiments:
                features.extend([
                    np.mean(recent_sentiments),  # 평균 감정
                    np.std(recent_sentiments) if len(recent_sentiments) > 1 else 0,  # 감정 변동성
                    max(recent_sentiments),  # 최고 감정
                    min(recent_sentiments),  # 최저 감정
                    np.sum(news_counts),  # 총 뉴스 수
                    len(recent_sentiments),  # 데이터 포인트 수
                ])
            else:
                features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 6)

        # 2. 가격 기술 지표 (간소화)
        try:
            ticker = self.upbit.GetTicker([f'KRW-{coin}'])
            if ticker:
                price_data = ticker[0]
                features.extend([
                    float(price_data.change_rate),  # 변동률
                    float(price_data.acc_trade_volume_24h),  # 거래량
                    float(price_data.trade_price)  # 현재 가격 (로그)
                ])
            else:
                features.extend([0.0] * 3)
        except:
            features.extend([0.0] * 3)

        return np.array(features)

    def train_quick_models(self, coins=['BTC', 'ETH', 'ADA', 'DOT']):
        """빠른 모델 훈련 (1시간 제한)"""
        logger.info("🚀 빠른 모델 훈련 시작...")
        train_start = time.time()

        for coin in coins:
            try:
                # 시간 체크
                elapsed = time.time() - train_start
                if elapsed > self.max_learning_time * 0.8:  # 80% 시간 초과시 중단
                    logger.warning(f"⏰ 시간 제한으로 {coin} 모델 훈련 건너뜀")
                    continue

                logger.info(f"🎓 {coin} 모델 훈련 중...")

                # 특성 데이터 생성 (최근 데이터만)
                if len(self.news_history) < 10:
                    logger.warning(f"{coin}: 훈련 데이터 부족")
                    continue

                X = []
                y = []

                # 빠른 데이터 준비 (최근 50개만)
                for i in range(min(50, len(self.news_history) - 1)):
                    features = self.create_prediction_features(coin)

                    if len(features) > 0:
                        # 다음 시간의 가격 변화를 예측
                        try:
                            current_ticker = self.upbit.GetTicker([f'KRW-{coin}'])
                            if current_ticker:
                                price_change = float(current_ticker[0].change_rate) * 100
                                X.append(features)
                                y.append(price_change)
                        except:
                            continue

                if len(X) < 5:
                    logger.warning(f"{coin}: 충분한 훈련 데이터 없음")
                    continue

                X = np.array(X)
                y = np.array(y)

                # 빠른 모델 (GradientBoosting 대신 RandomForest)
                model = RandomForestRegressor(
                    n_estimators=50,  # 트리 수 줄임
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1  # 병렬 처리
                )

                # 데이터 정규화
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 모델 훈련
                model.fit(X_scaled, y)

                # 성능 평가 (간소화)
                y_pred = model.predict(X_scaled)
                mae = mean_absolute_error(y, y_pred)

                # 저장
                self.prediction_models[coin] = model
                self.scalers[coin] = scaler

                logger.info(f"✅ {coin} 모델 완료 (MAE: {mae:.2f}%)")

            except Exception as e:
                logger.error(f"{coin} 모델 훈련 실패: {e}")

        total_elapsed = time.time() - train_start
        logger.info(f"🎯 모델 훈련 완료 ({total_elapsed:.1f}초)")

    def predict_price_movement(self, coin, horizon_hours=1):
        """가격 변동 예측 (적응형 학습 포함)"""
        predictions = []

        # 1. 기존 뉴스 기반 예측
        if coin in self.prediction_models:
            try:
                features = self.create_prediction_features(coin)
                if len(features) > 0:
                    features_scaled = self.scalers[coin].transform(features.reshape(1, -1))
                    predicted_change = self.prediction_models[coin].predict(features_scaled)[0]
                    predictions.append({
                        'source': 'news_model',
                        'predicted_change': predicted_change,
                        'confidence': 0.6
                    })
            except Exception as e:
                logger.error(f"{coin} 뉴스 모델 예측 실패: {e}")

        # 2. 적응형 학습 예측
        if self.adaptive_learning_enabled and self.adaptive_learner:
            try:
                adaptive_prediction = self.adaptive_learner.get_enhanced_prediction(coin)
                if adaptive_prediction:
                    predictions.append({
                        'source': 'adaptive_model',
                        'predicted_change': adaptive_prediction['predicted_change_pct'],
                        'confidence': adaptive_prediction['confidence']
                    })
            except Exception as e:
                logger.error(f"{coin} 적응형 모델 예측 실패: {e}")

        if not predictions:
            return None

        # 3. 앙상블 예측 (가중 평균)
        total_weight = sum(p['confidence'] for p in predictions)
        if total_weight == 0:
            return None

        ensemble_prediction = sum(p['predicted_change'] * p['confidence'] for p in predictions) / total_weight
        ensemble_confidence = min(total_weight / len(predictions), 1.0)

        # 현재 가격 정보
        ticker = self.upbit.GetTicker([f'KRW-{coin}'])
        if not ticker:
            return None

        current_price = ticker[0].trade_price
        predicted_price = current_price * (1 + ensemble_prediction / 100)

        prediction_result = {
            'coin': coin,
            'current_price': current_price,
            'predicted_change_pct': ensemble_prediction,
            'predicted_price': predicted_price,
            'horizon_hours': horizon_hours,
            'timestamp': datetime.now(),
            'confidence': ensemble_confidence,
            'ensemble_details': predictions,
            'adaptive_learning': self.adaptive_learning_enabled
        }

        # 적응형 학습에 예측 기록 (나중에 검증용)
        if self.adaptive_learning_enabled:
            self.record_prediction_for_later_verification(coin, ensemble_prediction, ensemble_confidence)

        return prediction_result

    def record_prediction_for_later_verification(self, coin, predicted_change, confidence):
        """예측을 나중에 검증하기 위해 기록"""
        if not hasattr(self, 'pending_predictions'):
            self.pending_predictions = deque(maxlen=1000)

        self.pending_predictions.append({
            'coin': coin,
            'predicted_change': predicted_change,
            'confidence': confidence,
            'prediction_time': datetime.now(),
            'current_price': self.upbit.GetTicker([f'KRW-{coin}'])[0].trade_price if self.upbit.GetTicker([f'KRW-{coin}']) else 0
        })

    def verify_and_learn_from_predictions(self):
        """예측 검증 및 학습"""
        if not hasattr(self, 'pending_predictions') or not self.adaptive_learning_enabled:
            return

        verified_count = 0
        for prediction in list(self.pending_predictions):
            # 30분 후 검증
            time_elapsed = (datetime.now() - prediction['prediction_time']).total_seconds()
            if time_elapsed >= 1800:  # 30분
                coin = prediction['coin']
                try:
                    current_ticker = self.upbit.GetTicker([f'KRW-{coin}'])
                    if current_ticker:
                        current_price = current_ticker[0].trade_price
                        actual_change = (current_price - prediction['current_price']) / prediction['current_price'] * 100

                        # 적응형 학습에 피드백
                        self.adaptive_learner.learn_from_prediction_feedback(
                            coin,
                            prediction['predicted_change'],
                            actual_change,
                            prediction['confidence']
                        )

                        verified_count += 1

                    # 검증된 예측 제거
                    self.pending_predictions.remove(prediction)

                except Exception as e:
                    logger.error(f"예측 검증 실패 ({coin}): {e}")

        if verified_count > 0:
            logger.info(f"🎯 {verified_count}개 예측 검증 및 학습 완료")

    def execute_trade_with_learning(self, coin, signal, price, quantity=None):
        """학습 기능이 포함된 거래 실행"""
        try:
            # 거래 실행 (기존 로직)
            trade_executed = False
            if signal == 'BUY':
                logger.info(f"💰 매수 신호 실행: {coin} @ {price:,}원")
                trade_executed = True
            elif signal == 'SELL':
                logger.info(f"💰 매도 신호 실행: {coin} @ {price:,}원")
                trade_executed = True

            # 적응형 학습에 거래 기록
            if trade_executed and self.adaptive_learning_enabled:
                self.adaptive_learner.experience_collector.record_trade(
                    coin=coin,
                    action=signal,
                    price=price,
                    quantity=quantity or 0
                )

            return trade_executed

        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
            return False

    def continuous_learning_update(self):
        """지속적 학습 업데이트"""
        if not self.adaptive_learning_enabled:
            return

        try:
            # 예측 검증
            self.verify_and_learn_from_predictions()

            # 지속적 학습 사이클
            self.adaptive_learner.continuous_learning_cycle()

            # 학습 상태 저장
            self.adaptive_learner.save_learning_state()

            logger.info("🔄 지속적 학습 업데이트 완료")

        except Exception as e:
            logger.error(f"지속적 학습 업데이트 실패: {e}")

    def generate_trading_signals(self):
        """거래 신호 생성"""
        signals = {}

        for coin in ['BTC', 'ETH', 'ADA', 'DOT']:
            prediction = self.predict_price_movement(coin)
            if not prediction:
                signals[coin] = 'HOLD'
                continue

            predicted_change = prediction['predicted_change_pct']
            confidence = prediction['confidence']

            # 신호 생성 로직 (보수적)
            if predicted_change > 2.0 and confidence > 0.6:
                signals[coin] = 'BUY'
            elif predicted_change < -2.0 and confidence > 0.6:
                signals[coin] = 'SELL'
            else:
                signals[coin] = 'HOLD'

            logger.info(f"🎯 {coin}: {signals[coin]} (예측: {predicted_change:+.2f}%, 신뢰도: {confidence:.2f})")

        return signals

    def save_models(self, filepath='models/news_sentiment_models.pkl'):
        """모델 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            model_data = {
                'models': self.prediction_models,
                'scalers': self.scalers,
                'news_history': list(self.news_history)[-100:],  # 최근 100개만
                'timestamp': datetime.now()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"💾 모델 저장 완료: {filepath}")

        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

    def load_models(self, filepath='models/news_sentiment_models.pkl'):
        """모델 로드"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)

                self.prediction_models = model_data['models']
                self.scalers = model_data['scalers']

                if 'news_history' in model_data:
                    self.news_history.extend(model_data['news_history'])

                logger.info(f"📂 모델 로드 완료: {filepath}")
                return True
            else:
                logger.info("저장된 모델이 없습니다. 새로 훈련합니다.")
                return False

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def run_one_hour_learning(self):
        """1시간 제한 학습"""
        learning_start = time.time()
        logger.info("🔥 1시간 집중 학습 시작!")

        try:
            # 기존 모델 로드 시도
            self.load_models()

            # 1. 뉴스 수집 및 분석 (최대 20분)
            logger.info("📰 1단계: 뉴스 수집 및 감정 분석")
            for cycle in range(3):  # 3번 수집
                elapsed = time.time() - learning_start
                if elapsed > self.max_learning_time * 0.4:  # 40% 시간 초과시 중단
                    logger.warning("⏰ 뉴스 수집 시간 초과, 다음 단계로")
                    break

                logger.info(f"📊 뉴스 수집 사이클 {cycle + 1}/3")
                coin_sentiments = self.collect_and_analyze_news()

                if cycle < 2:  # 마지막이 아니면 잠깐 대기
                    time.sleep(30)

            # 2. 모델 훈련 (최대 30분)
            elapsed = time.time() - learning_start
            remaining_time = self.max_learning_time - elapsed

            if remaining_time > 600:  # 10분 이상 남았을 때만
                logger.info("🎓 2단계: 머신러닝 모델 훈련")
                self.train_quick_models()
            else:
                logger.warning("⏰ 모델 훈련 시간 부족, 기존 모델 사용")

            # 3. 모델 저장
            self.save_models()

            # 4. 초기 예측 생성
            logger.info("🔮 3단계: 초기 예측 생성")
            signals = self.generate_trading_signals()

            total_elapsed = time.time() - learning_start
            logger.info(f"✅ 1시간 학습 완료! ({total_elapsed/60:.1f}분 소요)")
            logger.info("🚀 트레이딩 준비 완료!")

            return True

        except Exception as e:
            logger.error(f"❌ 학습 중 오류: {e}")
            return False

    def run_trading_cycle(self):
        """1시간 트레이딩 사이클"""
        logger.info("💰 1시간 트레이딩 시작!")

        trading_start = time.time()
        cycle_count = 0

        try:
            while time.time() - trading_start < 3600:  # 1시간
                cycle_count += 1
                logger.info(f"🔄 트레이딩 사이클 #{cycle_count}")

                # 1. 새로운 뉴스 수집 (간소화)
                if cycle_count % 3 == 1:  # 3사이클마다
                    coin_sentiments = self.collect_and_analyze_news()

                # 2. 거래 신호 생성
                signals = self.generate_trading_signals()

                # 3. 거래 실행 (학습 포함)
                for coin, signal in signals.items():
                    if signal != 'HOLD':
                        ticker = self.upbit.GetTicker([f'KRW-{coin}'])
                        if ticker:
                            current_price = ticker[0].trade_price
                            self.execute_trade_with_learning(coin, signal, current_price)

                # 4. 지속적 학습 업데이트 (5분마다)
                if cycle_count % 3 == 0:  # 15분마다 (3사이클 * 5분)
                    self.continuous_learning_update()

                # 10분 대기
                logger.info("⏱️ 10분 대기...")
                time.sleep(600)

        except KeyboardInterrupt:
            logger.info("🛑 사용자에 의한 트레이딩 중지")
        except Exception as e:
            logger.error(f"❌ 트레이딩 중 오류: {e}")

        total_elapsed = time.time() - trading_start
        logger.info(f"🏁 트레이딩 완료 ({total_elapsed/60:.1f}분 소요)")

def main():
    """메인 함수 - 1시간 학습 + 1시간 트레이딩"""
    print("🌍 뉴스 기반 코인 예측 AI")
    print("=" * 50)

    try:
        # NLTK 데이터 다운로드 (최초 실행시)
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass

        ai = NewsSentimentAI()

        while True:
            logger.info("🚀 새로운 학습+트레이딩 주기 시작!")

            # 1단계: 1시간 집중 학습
            logger.info("=" * 60)
            logger.info("📚 PHASE 1: 1시간 집중 학습")
            logger.info("=" * 60)

            success = ai.run_one_hour_learning()

            if success:
                # 2단계: 1시간 트레이딩
                logger.info("=" * 60)
                logger.info("💰 PHASE 2: 1시간 트레이딩")
                logger.info("=" * 60)

                ai.run_trading_cycle()
            else:
                logger.error("❌ 학습 실패로 이번 주기 건너뜀")
                time.sleep(3600)  # 1시간 대기

            logger.info("🔄 다음 주기까지 잠시 대기...")
            time.sleep(300)  # 5분 휴식

    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의한 시스템 종료")
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")

if __name__ == "__main__":
    main()