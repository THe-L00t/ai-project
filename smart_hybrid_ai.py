#!/usr/bin/env python3
"""
스마트 하이브리드 AI - 자가 학습 및 성장하는 트레이딩 AI
- 뉴스 기반 감정 분석으로 시장 심리 파악
- 강화학습을 통한 지속적 성능 개선
- 실패에서 배우고 성공을 강화하는 적응형 시스템
- 모든 오류를 해결한 안정적인 구조
"""

import os
import sys
import time
import logging
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque
import pickle
import json
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 머신러닝
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor

# NLP (TextBlob으로 단순화)
from textblob import TextBlob

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from exchange.UpbitAPI import UpbitAPI
from config_loader import get_config
from technical_indicators import TechnicalIndicators

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/smart_hybrid_ai.log')
    ]
)
logger = logging.getLogger(__name__)

class PricePredictionModel:
    """가격 예측 모델"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.prediction_cache = {}
        self.cache_timeout = 60

    def prepare_features(self, candle_data, indicators):
        """예측용 특성 준비 (1분봉 캔들 데이터 사용)"""
        try:
            features = []

            if len(candle_data) >= 15:
                # 1분봉 캔들 데이터에서 종가 추출
                closes = [candle['trade_price'] for candle in candle_data]

                # 가격 변화율 (1, 5, 15분)
                change_1m = (closes[-1] - closes[-2]) / closes[-2] * 100
                change_5m = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
                change_15m = (closes[-1] - closes[-16]) / closes[-16] * 100 if len(closes) >= 16 else 0
                features.extend([change_1m, change_5m, change_15m])

                # 캔들 패턴 특성 추가
                if len(candle_data) >= 3:
                    latest = candle_data[-1]
                    # 캔들 몸통 크기 (시가 대비 종가)
                    body_size = abs(latest['trade_price'] - latest['opening_price']) / latest['opening_price'] * 100
                    # 상승/하락 여부
                    direction = 1 if latest['trade_price'] > latest['opening_price'] else -1
                    # 고저차 (고가 대비 저가)
                    high_low_ratio = (latest['high_price'] - latest['low_price']) / latest['high_price'] * 100

                    features.extend([body_size, direction, high_low_ratio])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0, 0])

            # 기술적 지표
            if indicators:
                features.extend([
                    indicators.get('rsi', 50),
                    indicators.get('macd', {}).get('histogram', 0),
                    indicators.get('bollinger', {}).get('position', 50),
                    indicators.get('volume', {}).get('volume_ratio', 1)
                ])
            else:
                features.extend([50, 0, 50, 1])

            # 가격 변동성 (1분봉 기준)
            if len(candle_data) >= 10:
                closes = [candle['trade_price'] for candle in candle_data[-10:]]
                volatility = np.std(closes) / np.mean(closes) * 100
                features.append(volatility)
            else:
                features.append(0)

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"특성 준비 오류: {e}")
            return np.array([0, 0, 0, 0, 0, 0, 50, 0, 50, 1, 0]).reshape(1, -1)

    def predict_price_change(self, market, candle_data, indicators):
        """가격 변화 예측 (1분봉 캔들 데이터 사용)"""
        try:
            # 캐시 확인
            cache_key = f"{market}_{int(time.time() / self.cache_timeout)}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]

            if not self.is_trained:
                return 0.0, 0.0

            # 특성 준비 (1분봉 캔들 데이터 사용)
            features = self.prepare_features(candle_data, indicators)
            features_scaled = self.scaler.transform(features)

            # 예측
            predicted_change = self.model.predict(features_scaled)[0]

            # 신뢰도 계산 (캔들 패턴 고려)
            base_confidence = min(0.9, abs(predicted_change) / 10.0)

            # 캔들 패턴 신뢰도 보정
            if len(candle_data) >= 3:
                latest = candle_data[-1]
                volume_factor = min(2.0, latest.get('candle_acc_trade_volume', 1) / 1000000)  # 거래량 고려
                confidence = base_confidence * min(1.5, volume_factor)
            else:
                confidence = base_confidence

            confidence = min(0.95, confidence)  # 최대 95%

            # 캐시 저장
            result = (predicted_change, confidence)
            self.prediction_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"가격 예측 오류: {e}")
            return 0.0, 0.0


class SmartNewsCollector:
    """안정적인 뉴스 수집기 (비동기 이슈 해결)"""

    def __init__(self):
        # 안정적인 뉴스 소스들
        self.news_sources = [
            'https://cointelegraph.com/rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://decrypt.co/feed',
            'https://bitcoinmagazine.com/.rss/full/',
            'https://cryptopotato.com/feed/',
        ]

        # 코인 키워드
        self.coin_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi'],
            'ETH': ['ethereum', 'eth', 'vitalik', 'defi'],
            'ADA': ['cardano', 'ada'],
            'DOT': ['polkadot', 'dot', 'kusama']
        }

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; TradingBot/1.0)'})

    def collect_news_safe(self, max_articles=50):
        """안전한 동기식 뉴스 수집"""
        logger.info("📰 뉴스 수집 시작...")
        all_articles = []

        for url in self.news_sources:
            try:
                response = self.session.get(url, timeout=10, verify=False)
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)

                    for entry in feed.entries[:10]:  # 소스당 최대 10개
                        article = {
                            'title': entry.get('title', ''),
                            'description': entry.get('description', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'timestamp': datetime.now()
                        }
                        all_articles.append(article)

                        if len(all_articles) >= max_articles:
                            break

                    logger.info(f"📊 {url.split('//')[1].split('/')[0]} : {len(feed.entries)}개 기사")

            except Exception as e:
                logger.debug(f"뉴스 소스 실패 {url}: {e}")
                continue

            if len(all_articles) >= max_articles:
                break

        logger.info(f"✅ 총 {len(all_articles)}개 뉴스 수집 완료")
        return all_articles

    def filter_crypto_news(self, articles):
        """코인별 뉴스 필터링"""
        filtered = {coin: [] for coin in self.coin_keywords.keys()}

        for article in articles:
            text = (article['title'] + ' ' + article['description']).lower()

            for coin, keywords in self.coin_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        filtered[coin].append(article)
                        break

        for coin, news_list in filtered.items():
            logger.info(f"🔍 {coin}: {len(news_list)}개 관련 뉴스")

        return filtered

class SimpleSentimentAnalyzer:
    """간단하고 안정적인 감정 분석기"""

    def __init__(self):
        # 크립토 특화 감정 키워드
        self.positive_words = [
            'bullish', 'moon', 'pump', 'rally', 'surge', 'adoption', 'breakthrough',
            'partnership', 'upgrade', 'launch', 'positive', 'growth', 'rise', 'gain'
        ]

        self.negative_words = [
            'bearish', 'crash', 'dump', 'ban', 'hack', 'regulation', 'selloff',
            'decline', 'fall', 'drop', 'loss', 'negative', 'concern', 'risk'
        ]

    def analyze_sentiment(self, text):
        """텍스트 감정 분석"""
        if not text or len(text.strip()) < 5:
            return 0.0

        try:
            # TextBlob 기본 분석
            blob = TextBlob(text.lower())
            base_sentiment = blob.sentiment.polarity

            # 크립토 키워드 가중치
            positive_count = sum(1 for word in self.positive_words if word in text.lower())
            negative_count = sum(1 for word in self.negative_words if word in text.lower())

            # 키워드 점수 추가
            keyword_score = (positive_count - negative_count) * 0.1

            # 최종 점수 (-1 ~ 1)
            final_score = base_sentiment + keyword_score
            return max(-1.0, min(1.0, final_score))

        except Exception:
            return 0.0

    def analyze_news_batch(self, articles):
        """뉴스 배치 감정 분석"""
        if not articles:
            return 0.0, 0.0

        scores = []
        for article in articles:
            text = article['title'] + ' ' + article['description']
            score = self.analyze_sentiment(text)
            scores.append(score)

        avg_sentiment = np.mean(scores) if scores else 0.0
        sentiment_strength = np.std(scores) if len(scores) > 1 else 0.0

        return avg_sentiment, sentiment_strength

class ReinforcementLearner:
    """강화학습 모듈 - 거래 결과에서 학습"""

    def __init__(self):
        self.trade_history = deque(maxlen=1000)
        self.learning_buffer = deque(maxlen=500)
        self.model_performance = {}

        # 각 코인별 성공/실패 기록
        self.success_patterns = {}
        self.failure_patterns = {}

    def record_trade_result(self, coin, entry_data, exit_data, profit_pct):
        """거래 결과 기록 및 학습"""
        trade_result = {
            'coin': coin,
            'entry_time': entry_data['timestamp'],
            'exit_time': datetime.now(),
            'entry_price': entry_data['price'],
            'exit_price': exit_data['price'],
            'profit_pct': profit_pct,
            'duration_minutes': (datetime.now() - entry_data['timestamp']).total_seconds() / 60,
            'entry_conditions': entry_data.get('conditions', {}),
            'market_context': entry_data.get('context', {})
        }

        self.trade_history.append(trade_result)

        # 성공/실패 패턴 분류
        if profit_pct > 1.0:  # 1% 이상 수익
            if coin not in self.success_patterns:
                self.success_patterns[coin] = []
            self.success_patterns[coin].append(trade_result)
            logger.info(f"✅ {coin} 성공 패턴 학습: +{profit_pct:.2f}%")

        elif profit_pct < -0.5:  # 0.5% 이상 손실
            if coin not in self.failure_patterns:
                self.failure_patterns[coin] = []
            self.failure_patterns[coin].append(trade_result)
            logger.info(f"❌ {coin} 실패 패턴 학습: {profit_pct:.2f}%")

        # 성과 통계 업데이트
        self.update_performance_stats(coin)

    def update_performance_stats(self, coin):
        """코인별 성과 통계 업데이트"""
        recent_trades = [t for t in self.trade_history if t['coin'] == coin]

        if recent_trades:
            profits = [t['profit_pct'] for t in recent_trades]
            win_rate = len([p for p in profits if p > 0]) / len(profits)
            avg_profit = np.mean(profits)

            self.model_performance[coin] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_trades': len(recent_trades),
                'confidence': min(win_rate + 0.2, 1.0)  # 기본 신뢰도
            }

    def get_trading_confidence(self, coin, current_conditions):
        """현재 조건에 대한 거래 신뢰도"""
        if coin not in self.model_performance:
            return 0.5  # 기본값

        base_confidence = self.model_performance[coin]['confidence']

        # 성공 패턴과 유사한지 확인
        success_boost = 0.0
        if coin in self.success_patterns:
            # 간단한 패턴 매칭 (실제로는 더 복잡한 로직)
            success_boost = 0.1

        # 실패 패턴과 유사한지 확인
        failure_penalty = 0.0
        if coin in self.failure_patterns:
            failure_penalty = 0.1

        final_confidence = base_confidence + success_boost - failure_penalty
        return max(0.1, min(0.9, final_confidence))

class SmartHybridAI:
    """스마트 하이브리드 AI - 모든 오류 해결 버전"""

    def __init__(self):
        load_dotenv()

        # 설정 로드
        self.config = get_config()

        # 가격 예측 모델 초기화
        self.price_predictor = PricePredictionModel()
        self.price_history = {}
        self.candle_history = {}  # 1분봉 캔들 데이터

        # 컴포넌트 초기화
        self.upbit = UpbitAPI()
        self.news_collector = SmartNewsCollector()
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.reinforcement_learner = ReinforcementLearner()
        self.technical_analyzer = TechnicalIndicators()

        # 거래 설정 (AI_SETTINGS.md에서 로드)
        self.trading_mode = os.getenv('TRADING_MODE', 'paper')
        self.max_position_size = self.config.get('MAX_POSITION_SIZE', 0.15)
        self.stop_loss_percentage = self.config.get('STOP_LOSS_PERCENTAGE', 0.7)
        self.take_profit_percentage = self.config.get('TAKE_PROFIT_PERCENTAGE', 1.3)

        # 매매 임계값 (AI_SETTINGS.md에서 로드)
        self.buy_threshold = self.config.get('BUY_THRESHOLD_CHANGE', 3.0)
        self.sell_threshold = self.config.get('SELL_THRESHOLD_CHANGE', -1.0)
        self.confidence_threshold = self.config.get('MIN_CONFIDENCE_THRESHOLD', 0.8)

        # AI 학습 기능 설정
        self.enable_adaptive_learning = self.config.get('ENABLE_ADAPTIVE_LEARNING', True)

        # 고도화된 매수 타이밍 알고리즘 초기화
        try:
            from src.analysis.AdvancedEntrySignalEngine import AdvancedEntrySignalEngine
            self.advanced_entry_engine = AdvancedEntrySignalEngine(self.upbit, self.config)
            self.enable_advanced_entry = True
            logger.info("✅ 고도화된 매수 타이밍 알고리즘 통합 완료")
        except ImportError as e:
            logger.warning(f"⚠️ 고도화된 매수 알고리즘 로드 실패: {e}")
            self.enable_advanced_entry = False
        self.enable_news_sentiment = self.config.get('ENABLE_NEWS_SENTIMENT', True)
        self.enable_pattern_learning = self.config.get('ENABLE_PATTERN_LEARNING', True)

        # 신호 가중치 설정
        self.aggressive_pattern_weight = self.config.get('AGGRESSIVE_PATTERN_WEIGHT', 0.7)
        self.news_sentiment_weight = self.config.get('NEWS_SENTIMENT_WEIGHT', 0.8)
        self.adaptive_learning_weight = self.config.get('ADAPTIVE_LEARNING_WEIGHT', 0.6)
        self.pattern_model_weight = self.config.get('PATTERN_MODEL_WEIGHT', 0.6)

        # 대상 코인
        self.target_coins = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']

        # 데이터 저장소
        self.price_history = {}
        self.sentiment_history = deque(maxlen=100)
        self.positions = {}
        self.last_news_update = datetime.now() - timedelta(hours=1)

        # API 캐싱 시스템 (10초 TTL로 증가)
        self.api_cache = {}
        self.cache_ttl = self.config.get('API_CACHE_TTL', 10)

    def get_position_entry_time(self, position):
        """포지션의 진입 시간을 안전하게 가져오기"""
        return (position.get('entry_time') or
                position.get('timestamp') or
                datetime.now())

    def normalize_position_fields(self):
        """기존 포지션 필드를 표준화 - entry_time으로 통일"""
        for market, position in self.positions.items():
            if 'timestamp' in position and 'entry_time' not in position:
                position['entry_time'] = position['timestamp']
                del position['timestamp']

    def get_cached_ticker(self, markets):
        """캐시된 티커 데이터 조회 또는 새로 가져오기"""
        cache_key = ','.join(sorted(markets)) if isinstance(markets, list) else markets
        current_time = time.time()

        # 캐시에서 확인
        if cache_key in self.api_cache:
            cached_data = self.api_cache[cache_key]
            if current_time - cached_data['timestamp'] < self.cache_ttl:
                logger.debug(f"📋 캐시에서 티커 데이터 사용: {cache_key}")
                return cached_data['data']

        # 캐시 만료 또는 없음 - 새로 조회
        try:
            ticker_data = self.upbit.GetTicker(markets)
            if ticker_data:
                self.api_cache[cache_key] = {
                    'data': ticker_data,
                    'timestamp': current_time
                }
                logger.debug(f"🔄 새로운 티커 데이터 캐시: {cache_key}")
                return ticker_data
        except Exception as e:
            logger.error(f"티커 조회 실패: {e}")

        return None

        # 예측 모델들
        self.prediction_models = {}
        self.scalers = {}

        logger.info("🚀 스마트 하이브리드 AI 초기화 완료")
        logger.info(f"💪 모드: {self.trading_mode}, 포지션: {self.max_position_size*100}%")
        logger.info(f"⚙️  설정: 손절{self.stop_loss_percentage}% | 익절{self.take_profit_percentage}% | 매수{self.buy_threshold}% | 매도{self.sell_threshold}%")
        logger.info("📊 예측 방식: 1분봉 캔들 데이터 + 실시간 보조")

        # 기존 보유 코인 자동 인식
        self.load_existing_positions()

        # 포지션 필드 표준화
        self.normalize_position_fields()

    def collect_price_features(self, market, ticker_data=None):
        """가격 특성 수집 (API 최적화 버전)"""
        try:
            # ticker_data가 제공되면 사용, 없으면 캐시에서 조회
            if ticker_data and market in ticker_data:
                price_data = ticker_data[market]
                # 타입 검증
                if not hasattr(price_data, 'trade_price'):
                    logger.error(f"가격 데이터 타입 오류: {type(price_data)} - {price_data}")
                    return np.array([])
            else:
                ticker = self.get_cached_ticker([market])
                if not ticker:
                    return np.array([])
                price_data = ticker[0]
            coin = market.split('-')[1]

            # 안전한 속성 접근
            try:
                change_rate = getattr(price_data, 'change_rate', 0.0)
                trade_price = getattr(price_data, 'trade_price', 0.0)
                acc_volume = getattr(price_data, 'acc_trade_volume_24h', 0.0)
            except Exception as e:
                logger.error(f"가격 데이터 속성 접근 오류: {e} - {type(price_data)}")
                return np.array([])

            # 기본 특성
            features = [
                float(change_rate) * 100,  # 변동률
                float(trade_price) / 1000000,  # 정규화된 가격
                float(acc_volume) / 1000000000,  # 정규화된 거래량
            ]

            # 가격 히스토리 관리
            if market not in self.price_history:
                self.price_history[market] = deque(maxlen=20)

            self.price_history[market].append({
                'price': float(trade_price),
                'change': float(change_rate) * 100,
                'volume': float(acc_volume),
                'timestamp': datetime.now()
            })

            # 추가 기술적 특성
            if len(self.price_history[market]) >= 5:
                prices = [h['price'] for h in self.price_history[market]]
                changes = [h['change'] for h in self.price_history[market]]

                # 단기 추세
                recent_trend = np.mean(changes[-3:]) if len(changes) >= 3 else 0
                volatility = np.std(changes) if len(changes) >= 2 else 0

                features.extend([recent_trend, volatility])
            else:
                features.extend([0.0, 0.0])

            return np.array(features)

        except Exception as e:
            logger.error(f"가격 특성 수집 실패 ({market}): {e}")
            return np.array([])

    def update_sentiment_data(self):
        """뉴스 감정 데이터 업데이트"""
        try:
            # 30분마다 뉴스 업데이트
            if (datetime.now() - self.last_news_update).total_seconds() < 1800:
                return

            logger.info("📰 뉴스 감정 데이터 업데이트...")

            # 뉴스 수집
            articles = self.news_collector.collect_news_safe()
            filtered_news = self.news_collector.filter_crypto_news(articles)

            # 코인별 감정 분석
            sentiment_data = {}
            for coin, news_list in filtered_news.items():
                if news_list:
                    avg_sentiment, strength = self.sentiment_analyzer.analyze_news_batch(news_list)
                    sentiment_data[coin] = {
                        'sentiment': avg_sentiment,
                        'strength': strength,
                        'news_count': len(news_list)
                    }
                else:
                    sentiment_data[coin] = {'sentiment': 0.0, 'strength': 0.0, 'news_count': 0}

            # 감정 히스토리 저장
            self.sentiment_history.append({
                'timestamp': datetime.now(),
                'data': sentiment_data
            })

            self.last_news_update = datetime.now()
            logger.info("✅ 뉴스 감정 업데이트 완료")

        except Exception as e:
            logger.error(f"감정 데이터 업데이트 실패: {e}")

    def get_technical_analysis(self, market):
        """기술적 지표 분석 (캐시 최적화)"""
        try:
            # 30분봉 캐시 키 (30분마다 갱신)
            current_30min = int(time.time() / 1800)  # 1800초 = 30분
            cache_key = f"{market}_technical_{current_30min}"

            # 캐시 확인
            if cache_key in self.api_cache:
                cached_data = self.api_cache[cache_key]
                if time.time() - cached_data['timestamp'] < 1800:  # 30분 캐시
                    logger.debug(f"📋 기술적 분석 캐시 사용: {market}")
                    return cached_data['data']

            # 30분봉 50개 가져오기 (충분한 데이터 확보)
            candles = self.upbit.GetCandles(market, 'minutes', unit=30, count=50)

            if not candles or len(candles) < 20:
                return None

            # 가격과 거래량 데이터 추출 (최신순 -> 과거순으로 변환)
            prices = []
            volumes = []

            # candles는 최신 -> 과거 순이므로 뒤집어서 과거 -> 최신 순으로 만듦
            for candle in reversed(candles):
                prices.append(float(candle.get('trade_price', 0)))
                volumes.append(float(candle.get('candle_acc_trade_volume', 0)))

            if len(prices) < 20:
                return None

            # 기술적 지표 계산
            rsi = self.technical_analyzer.calculate_rsi(prices, period=9)
            macd_data = self.technical_analyzer.calculate_macd(prices, fast=3, slow=10, signal=16)
            bollinger_data = self.technical_analyzer.calculate_bollinger_bands(prices, period=20, std_dev=2)
            volume_data = self.technical_analyzer.analyze_volume(volumes, prices)

            # 종합 기술적 신호 생성
            technical_signal = self.technical_analyzer.generate_technical_signal(
                rsi, macd_data, bollinger_data, volume_data
            )

            result = {
                'rsi': rsi,
                'macd': macd_data,
                'bollinger': bollinger_data,
                'volume': volume_data,
                'signal': technical_signal
            }

            # 캐시에 저장 (30분 캐시)
            self.api_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            logger.debug(f"🔄 기술적 분석 캐시 저장: {market}")

            return result

        except Exception as e:
            logger.error(f"기술적 분석 실패 ({market}): {e}")
            return None

    def get_current_sentiment(self, coin):
        """현재 감정 점수 조회"""
        if not self.sentiment_history:
            return 0.0, 0.0

        latest = self.sentiment_history[-1]
        coin_data = latest['data'].get(coin, {})
        return coin_data.get('sentiment', 0.0), coin_data.get('strength', 0.0)

    def generate_smart_signal(self, market):
        """스마트 신호 생성 (모든 요소 결합)"""
        coin = market.split('-')[1]

        try:
            # 1. 가격 특성
            price_features = self.collect_price_features(market)
            if len(price_features) == 0:
                return 'HOLD', 0.0, []

            # 2. 현재 감정
            sentiment, sentiment_strength = self.get_current_sentiment(coin)

            # 3. 강화학습 신뢰도
            rl_confidence = self.reinforcement_learner.get_trading_confidence(coin, {})

            # 4. 신호 계산
            signals = []
            reasons = []

            # 가격 기반 신호
            change_rate = price_features[0]
            if change_rate > self.buy_threshold:
                signals.append(('BUY', 0.6))
                reasons.append(f"가격 상승 {change_rate:+.2f}%")
            elif change_rate < self.sell_threshold and market in self.positions:
                signals.append(('SELL', 0.6))
                reasons.append(f"가격 하락 {change_rate:+.2f}%")

            # 감정 기반 신호
            if sentiment > 0.2:
                signals.append(('BUY', 0.5))
                reasons.append(f"긍정 뉴스 {sentiment:.2f}")
            elif sentiment < -0.2:
                signals.append(('SELL', 0.5))
                reasons.append(f"부정 뉴스 {sentiment:.2f}")

            # 추세 기반 신호
            if len(price_features) >= 5:
                trend = price_features[3]  # recent_trend
                if trend > 0.5:
                    signals.append(('BUY', 0.4))
                    reasons.append(f"상승 추세 {trend:.2f}")
                elif trend < -0.5:
                    signals.append(('SELL', 0.4))
                    reasons.append(f"하락 추세 {trend:.2f}")

            # 신호 통합
            if not signals:
                return 'HOLD', 0.0, reasons

            # 가중 투표
            buy_weight = sum(w for s, w in signals if s == 'BUY')
            sell_weight = sum(w for s, w in signals if s == 'SELL')

            # 강화학습 신뢰도 적용
            total_confidence = max(buy_weight, sell_weight) * rl_confidence

            if buy_weight > sell_weight and total_confidence >= self.confidence_threshold:
                return 'BUY', total_confidence, reasons
            elif sell_weight > buy_weight and total_confidence >= self.confidence_threshold:
                return 'SELL', total_confidence, reasons
            else:
                return 'HOLD', total_confidence, reasons

        except Exception as e:
            logger.error(f"신호 생성 실패 ({market}): {e}")
            return 'HOLD', 0.0, []

    def generate_smart_signal_cached(self, market, ticker):
        """스마트 신호 생성 (기술적 지표 + 뉴스 감정 + 강화학습)"""
        coin = market.split('-')[1]

        try:
            # 1. 캐시된 가격 특성
            price_features = self.collect_price_features(market, {market: ticker})
            if len(price_features) == 0:
                return 'HOLD', 0.0, []

            # 2. 현재 감정
            sentiment, sentiment_strength = self.get_current_sentiment(coin)

            # 3. 기술적 지표 분석
            technical_analysis = self.get_technical_analysis(market)

            # 4. 강화학습 신뢰도
            rl_confidence = self.reinforcement_learner.get_trading_confidence(coin, {})

            # 5. 통합 신호 계산 (가중치 적용)
            signals = []
            reasons = []

            # 🚀 고도화된 매수 타이밍 알고리즘 (최우선)
            if self.enable_advanced_entry and hasattr(self, 'advanced_entry_engine'):
                try:
                    # 고도화된 매수 신호 생성
                    market_data = {
                        'trade_price': ticker.trade_price,
                        'timestamp': time.time(),
                        'volume': getattr(ticker, 'acc_trade_volume_24h', 0)
                    }

                    # 동기 버전 사용 (기존 시스템과 호환성)
                    advanced_signal = self.advanced_entry_engine.generate_buy_signal_sync(market, market_data)

                    if advanced_signal and advanced_signal.action == 'BUY':
                        # 고도화된 신호가 매수를 권장하는 경우
                        weighted_conf = advanced_signal.confidence / 100 * 0.8  # 80% 가중치
                        signals.append(('BUY', weighted_conf))

                        advanced_reasons = '; '.join(advanced_signal.reasoning[:2])  # 상위 2개 근거만
                        reasons.append(f"🚀 고도화 알고리즘: {advanced_reasons} (신뢰도: {advanced_signal.confidence:.1f}%)")

                        # 고도화된 알고리즘의 손익 정보 활용
                        if hasattr(self, 'current_entry_info'):
                            self.current_entry_info[market] = {
                                'stop_loss': advanced_signal.stop_loss,
                                'take_profit': advanced_signal.take_profit,
                                'position_size': advanced_signal.position_size,
                                'risk_reward': advanced_signal.risk_reward_ratio
                            }

                        logger.info(f"🚀 {market} 고도화 매수 신호: {advanced_signal.confidence:.1f}% - {advanced_reasons}")

                except Exception as e:
                    logger.warning(f"⚠️ 고도화된 매수 알고리즘 실행 실패: {e}")
                    # 실패시 기존 로직으로 폴백

            # 🔮 기존 가격 예측 기반 신호 (보조 역할)
            if self.enable_pattern_learning:
                # 현재가 추출
                current_price = ticker.trade_price

                # 1분봉 캔들 데이터 수집 (1분마다 갱신)
                current_minute = int(time.time() / 60)
                cache_key = f"{market}_candle_{current_minute}"

                if cache_key not in self.candle_history:
                    # 1분봉 데이터 조회 (최근 50개)
                    candle_data = self.upbit.GetMinuteCandles(market, 50)
                    if candle_data:
                        self.candle_history[cache_key] = candle_data
                        logger.debug(f"📊 {market} 1분봉 데이터 갱신: {len(candle_data)}개")
                    else:
                        # 1분봉 데이터가 없으면 실시간 데이터로 대체
                        if market not in self.price_history:
                            self.price_history[market] = []
                        self.price_history[market].append(current_price)
                        if len(self.price_history[market]) > 50:
                            self.price_history[market] = self.price_history[market][-50:]

                # 가격 예측 실행
                if cache_key in self.candle_history:
                    candle_data = self.candle_history[cache_key]
                    if len(candle_data) >= 15:
                        predicted_change, prediction_confidence = self.price_predictor.predict_price_change(
                            market, candle_data, technical_analysis
                        )
                    else:
                        predicted_change, prediction_confidence = 0.0, 0.0
                else:
                    # 실시간 데이터 폴백
                    if market in self.price_history and len(self.price_history[market]) >= 15:
                        # 실시간 데이터를 캔들 형태로 변환
                        fake_candles = [{'trade_price': p, 'opening_price': p, 'high_price': p, 'low_price': p}
                                       for p in self.price_history[market]]
                        predicted_change, prediction_confidence = self.price_predictor.predict_price_change(
                            market, fake_candles, technical_analysis
                        )
                    else:
                        predicted_change, prediction_confidence = 0.0, 0.0

                # 예측 기반 매수 신호 (기존 로직, 낮은 가중치)
                if predicted_change > 2.0 and prediction_confidence > 0.3:
                    weighted_conf = prediction_confidence * self.pattern_model_weight * 0.5  # 가중치 절반으로 축소
                    signals.append(('BUY', weighted_conf))
                    reasons.append(f"🔮 기존예측: +{predicted_change:.1f}% 상승 예상")

                # 예측 기반 매도 신호
                elif predicted_change < -2.0 and prediction_confidence > 0.3 and market in self.positions:
                    weighted_conf = prediction_confidence * self.pattern_model_weight * 0.5
                    signals.append(('SELL', weighted_conf))
                    reasons.append(f"🔮 기존예측: {predicted_change:.1f}% 하락 예상")

                # 기존 반응형 로직 (최후 보조 수단)
                change_rate = price_features[0]
                if change_rate > self.buy_threshold * 3:  # 임계값 3배로 높임 (극단적 급등시만)
                    weighted_conf = 0.2 * self.pattern_model_weight  # 가중치 더욱 낮춤
                    signals.append(('BUY', weighted_conf))
                    reasons.append(f"긴급: 극단적 급등 {change_rate:+.2f}%")
                elif change_rate < self.sell_threshold and market in self.positions:
                    weighted_conf = 0.3 * self.pattern_model_weight
                    signals.append(('SELL', weighted_conf))
                    reasons.append(f"반응형: 급락 {change_rate:+.2f}%")

            # 감정 기반 신호 (뉴스 분석)
            if self.enable_news_sentiment:
                if sentiment > 0.2:
                    weighted_conf = sentiment_strength * self.news_sentiment_weight
                    signals.append(('BUY', weighted_conf))
                    reasons.append(f"뉴스: 긍정 감정 {sentiment:.2f}")
                elif sentiment < -0.2 and market in self.positions:
                    weighted_conf = sentiment_strength * self.news_sentiment_weight
                    signals.append(('SELL', weighted_conf))
                    reasons.append(f"뉴스: 부정 감정 {sentiment:.2f}")

            # 기술적 지표 신호
            if technical_analysis:
                tech_signal = technical_analysis['signal']
                if tech_signal['signal'] == 'BUY':
                    weighted_conf = tech_signal['confidence'] * self.aggressive_pattern_weight
                    signals.append(('BUY', weighted_conf))
                    reasons.extend([f"기술적: {reason}" for reason in tech_signal['reasons']])
                elif tech_signal['signal'] == 'SELL' and market in self.positions:
                    weighted_conf = tech_signal['confidence'] * self.aggressive_pattern_weight
                    signals.append(('SELL', weighted_conf))
                    reasons.extend([f"기술적: {reason}" for reason in tech_signal['reasons']])

                # 기술적 지표 상세 로그
                rsi = technical_analysis['rsi']
                macd = technical_analysis['macd']
                bollinger = technical_analysis['bollinger']
                volume = technical_analysis['volume']

                logger.info(f"📊 {market} 기술적 지표: RSI={rsi:.1f}, MACD={macd['histogram']:.4f}, 볼린저={bollinger['position']:.1f}%, 거래량={volume['volume_ratio']:.1f}x")

            # 강화학습 신호 (적응형 학습)
            if self.enable_adaptive_learning:
                if rl_confidence > 0.5:
                    weighted_conf = rl_confidence * self.adaptive_learning_weight
                    signals.append(('BUY', weighted_conf))
                    reasons.append(f"강화학습: 신뢰도 {rl_confidence:.2f}")
                elif rl_confidence < 0.3 and market in self.positions:
                    weighted_conf = (1.0 - rl_confidence) * self.adaptive_learning_weight
                    signals.append(('SELL', weighted_conf))
                    reasons.append(f"강화학습: 회피 {rl_confidence:.2f}")

            # 신호가 없으면 HOLD
            if not signals:
                return 'HOLD', 0.0, reasons

            # 가중 투표
            buy_weight = sum(w for s, w in signals if s == 'BUY')
            sell_weight = sum(w for s, w in signals if s == 'SELL')

            # 강화학습 신뢰도 적용
            total_confidence = max(buy_weight, sell_weight) * rl_confidence

            if buy_weight > sell_weight and total_confidence >= self.confidence_threshold:
                return 'BUY', total_confidence, reasons
            elif sell_weight > buy_weight and total_confidence >= self.confidence_threshold:
                return 'SELL', total_confidence, reasons
            else:
                return 'HOLD', total_confidence, reasons

        except Exception as e:
            logger.error(f"신호 생성 실패 ({market}): {e}")
            return 'HOLD', 0.0, []

    def execute_smart_trade(self, market, signal, confidence, reasons, current_price):
        """스마트 거래 실행 - API 최적화 버전"""
        coin = market.split('-')[1]

        try:

            if signal == 'BUY' and market not in self.positions:
                # 매수 실행
                krw_balance = self.upbit.GetKRWBalance()
                buy_amount = krw_balance * self.max_position_size

                if buy_amount >= 5000:
                    logger.info(f"🔥 {market} 매수 실행 (신뢰도: {confidence:.2f})")
                    for reason in reasons:
                        logger.info(f"   💡 {reason}")

                    if self.trading_mode == 'live':
                        # 실제 매수 (API 오류 시 모의거래로 대체)
                        try:
                            result = self.upbit.BuyMarket(market, buy_amount)
                            if result:
                                quantity = buy_amount / current_price
                                logger.info(f"✅ 실제 매수 성공: {quantity:.8f} @ {current_price:,}원")
                            else:
                                raise Exception("매수 API 실패")
                        except:
                            logger.warning("API 오류로 모의거래 모드로 전환")
                            quantity = buy_amount / current_price
                    else:
                        # 모의 매수
                        quantity = buy_amount / current_price
                        logger.info(f"📝 모의 매수: {quantity:.8f} @ {current_price:,}원")

                    # 포지션 기록
                    self.positions[market] = {
                        'type': 'long',
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'conditions': {'confidence': confidence, 'reasons': reasons},
                        'context': {'sentiment': self.get_current_sentiment(coin)}
                    }

                    return True

            elif signal == 'SELL' and market in self.positions:
                # 매도 실행
                position = self.positions[market]

                logger.info(f"🔥 {market} 매도 실행 (신뢰도: {confidence:.2f})")
                for reason in reasons:
                    logger.info(f"   💡 {reason}")

                # 수량 추출 (quantity 또는 amount 키 지원)
                sell_quantity = position.get('quantity') or position.get('amount')
                if not sell_quantity:
                    logger.error(f"❌ 매도 수량 정보 없음 - 포지션 유지")
                    return False

                # 최소 주문 금액 검증 (5,000원)
                estimated_value = sell_quantity * current_price
                min_order_amount = 5000

                if estimated_value < min_order_amount:
                    logger.warning(f"⚠️  {market} 매도 금액 부족: {estimated_value:,.0f}원 < {min_order_amount:,}원")
                    logger.warning(f"   소량 보유로 매도 불가 - 포지션 관리에서 제외")

                    # 소액 포지션은 포지션 목록에서 제거 (더 이상 관리하지 않음)
                    if estimated_value < 100:  # 100원 미만은 완전히 제거
                        logger.info(f"🗑️  {market} 소액 포지션 제거: {estimated_value:,.0f}원")
                        del self.positions[market]

                    return False

                sell_success = False
                if self.trading_mode == 'live':
                    try:
                        result = self.upbit.SellMarket(market, sell_quantity)
                        if result:
                            sell_success = True
                            logger.info(f"✅ 실제 매도 성공: {sell_quantity:.8f} (약 {estimated_value:,.0f}원)")
                        else:
                            logger.error(f"❌ 매도 API 실패 - 포지션 유지")
                            return False
                    except Exception as e:
                        error_msg = str(e)
                        if "under_min_total" in error_msg:
                            logger.warning(f"⚠️  {market} 최소 주문 금액 미달 - 포지션 유지")
                        else:
                            logger.error(f"❌ 매도 API 오류: {e} - 포지션 유지")
                        return False
                else:
                    # 모의거래 모드
                    sell_success = True
                    logger.info(f"📝 모의 매도: {sell_quantity:.8f} (약 {estimated_value:,.0f}원)")

                if sell_success:
                    # 수익률 계산
                    profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100

                    logger.info(f"✅ 매도 완료: 수익률 {profit_pct:+.2f}%")

                    # 강화학습에 결과 기록
                    entry_data_for_learning = {
                        'timestamp': self.get_position_entry_time(position),
                        'price': position['entry_price'],
                        'conditions': position.get('conditions', {})
                    }
                    self.reinforcement_learner.record_trade_result(
                        coin, entry_data_for_learning, {'price': current_price}, profit_pct
                    )

                    # 포지션 삭제 (매도 성공시에만)
                    del self.positions[market]
                    return True

                return False

            return False

        except Exception as e:
            logger.error(f"거래 실행 실패: {e}")
            return False

    def check_risk_management(self):
        """리스크 관리"""
        for market, position in list(self.positions.items()):
            try:
                ticker = self.get_cached_ticker([market])
                if not ticker:
                    continue

                current_price = ticker[0].trade_price
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100

                # 손절매
                if profit_pct <= -self.stop_loss_percentage:
                    logger.warning(f"🛑 {market} 손절매 발동: {profit_pct:.2f}%")
                    self.execute_smart_trade(market, 'SELL', 1.0, ['손절매'], current_price)

                # 익절매
                elif profit_pct >= self.take_profit_percentage:
                    logger.info(f"🎯 {market} 익절매 발동: +{profit_pct:.2f}%")
                    self.execute_smart_trade(market, 'SELL', 1.0, ['익절매'], current_price)

            except Exception as e:
                logger.error(f"리스크 관리 오류 ({market}): {e}")

    def check_risk_management_cached(self, ticker_data):
        """리스크 관리 (캐시된 시세 사용)"""
        for market, position in list(self.positions.items()):
            try:
                if market not in ticker_data:
                    continue

                current_price = ticker_data[market].trade_price
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100

                # 손절매
                if profit_pct <= -self.stop_loss_percentage:
                    logger.warning(f"🛑 {market} 손절매 발동: {profit_pct:.2f}%")
                    self.execute_smart_trade(market, 'SELL', 1.0, ['손절매'], current_price)

                # 익절매
                elif profit_pct >= self.take_profit_percentage:
                    logger.info(f"🎯 {market} 익절매 발동: +{profit_pct:.2f}%")
                    self.execute_smart_trade(market, 'SELL', 1.0, ['익절매'], current_price)

            except Exception as e:
                logger.error(f"리스크 관리 오류 ({market}): {e}")

    def save_learning_state(self):
        """학습 상태 저장"""
        try:
            os.makedirs('models', exist_ok=True)

            state = {
                'trade_history': list(self.reinforcement_learner.trade_history),
                'model_performance': self.reinforcement_learner.model_performance,
                'success_patterns': self.reinforcement_learner.success_patterns,
                'failure_patterns': self.reinforcement_learner.failure_patterns,
                'sentiment_history': list(self.sentiment_history)[-50:],  # 최근 50개만
                'timestamp': datetime.now()
            }

            with open('models/smart_hybrid_state.pkl', 'wb') as f:
                pickle.dump(state, f)

            logger.info("💾 학습 상태 저장 완료")

        except Exception as e:
            logger.error(f"학습 상태 저장 실패: {e}")

    def load_learning_state(self):
        """학습 상태 로드"""
        try:
            if os.path.exists('models/smart_hybrid_state.pkl'):
                with open('models/smart_hybrid_state.pkl', 'rb') as f:
                    state = pickle.load(f)

                self.reinforcement_learner.trade_history.extend(state.get('trade_history', []))
                self.reinforcement_learner.model_performance = state.get('model_performance', {})
                self.reinforcement_learner.success_patterns = state.get('success_patterns', {})
                self.reinforcement_learner.failure_patterns = state.get('failure_patterns', {})
                self.sentiment_history.extend(state.get('sentiment_history', []))

                logger.info("📂 학습 상태 로드 완료")

                # 성과 리포트
                total_trades = len(self.reinforcement_learner.trade_history)
                if total_trades > 0:
                    profits = [t['profit_pct'] for t in self.reinforcement_learner.trade_history]
                    win_rate = len([p for p in profits if p > 0]) / len(profits)
                    avg_profit = np.mean(profits)

                    logger.info(f"📈 학습 이력: {total_trades}건 거래, 승률 {win_rate:.1%}, 평균 {avg_profit:+.2f}%")

        except Exception as e:
            logger.info(f"이전 학습 상태 없음 또는 로드 실패: {e}")

    def load_existing_positions(self):
        """기존 보유 코인을 포지션으로 자동 등록 - API 최적화 버전"""
        try:
            accounts = self.upbit.GetAccountInfo()
            if not accounts:
                logger.warning("❌ 계정 정보를 가져올 수 없습니다")
                return

            # 보유 중인 대상 코인들 찾기
            held_markets = []
            account_data = {}

            for account in accounts:
                currency = account['currency']
                balance = float(account['balance'])

                # KRW는 제외하고, 잔고가 있는 코인만 처리
                if currency != 'KRW' and balance > 0:
                    market = f'KRW-{currency}'
                    avg_buy_price = float(account.get('avg_buy_price', 0))

                    # 🔥 모든 보유 코인을 포지션으로 등록 (가치 기준 완화)
                    # 대상 코인이거나 평균매수가가 있는 경우 (실제 구매한 코인)
                    if market in self.target_coins or avg_buy_price > 0:
                        held_markets.append(market)
                        account_data[market] = {
                            'balance': balance,
                            'avg_buy_price': avg_buy_price
                        }
                        logger.info(f"🔍 발견된 보유 코인: {market} ({balance:.8f}개, 평균매수가: {avg_buy_price:,.0f}원)")

            if not held_markets:
                logger.info("📝 기존 보유 코인이 없거나 대상 코인 아님")
                # 모든 계정 정보 로그 출력 (디버깅용)
                logger.info("💰 전체 계정 정보:")
                for account in accounts:
                    currency = account['currency']
                    balance = float(account['balance'])
                    if balance > 0:
                        logger.info(f"  {currency}: {balance:.8f}")
                return

            # 배치로 현재 가격 조회 (API 최적화 + 캐싱)
            try:
                tickers = self.get_cached_ticker(held_markets)
                if not tickers:
                    logger.warning("⚠️  보유 코인 시세 조회 실패")
                    return

                ticker_data = {ticker.market: ticker for ticker in tickers}

                loaded_positions = 0
                total_value = 0

                for market in held_markets:
                    try:
                        account_info = account_data[market]
                        balance = account_info['balance']
                        avg_buy_price = account_info['avg_buy_price']

                        if market in ticker_data and avg_buy_price > 0:
                            current_price = ticker_data[market].trade_price
                            position_value = balance * current_price

                            # 실제 수익률 계산
                            profit_pct = (current_price - avg_buy_price) / avg_buy_price * 100

                            # 최소 금액 검증 (5,000원 미만은 포지션 관리에서 제외)
                            if position_value < 5000:
                                logger.warning(f"⚠️  {market} 소액 포지션 스킵: {position_value:,.0f}원 < 5,000원")
                                if position_value < 100:  # 100원 미만은 완전 무시
                                    logger.info(f"🗑️  {market} 극소액 보유량 무시: {position_value:,.0f}원")
                                continue

                            # 포지션으로 등록 (실제 평균 매수가 사용)
                            self.positions[market] = {
                                'type': 'long',
                                'quantity': balance,  # quantity 키로 통일
                                'entry_price': avg_buy_price,  # 실제 평균 매수가 사용
                                'entry_time': datetime.now() - timedelta(days=1),  # 기존 보유로 가정
                                'conditions': {'confidence': 0.5, 'reasons': ['기존 보유']},
                                'context': {'source': 'existing'}  # 기존 보유 코인 표시
                            }

                            loaded_positions += 1
                            total_value += position_value

                            logger.info(f"📦 기존 포지션 등록: {market} ({balance:.8f}개, {avg_buy_price:,.0f}→{current_price:,.0f}원, {profit_pct:+.2f}%)")

                    except Exception as e:
                        logger.warning(f"⚠️  {market} 포지션 등록 실패: {e}")

                if loaded_positions > 0:
                    logger.info(f"✅ 기존 보유 코인 {loaded_positions}개 포지션 등록 완료 (총 {total_value:,.0f}원)")

                    # 🔥 보유 코인들을 대상 코인 목록에 자동 추가
                    original_target_count = len(self.target_coins)
                    for market in held_markets:
                        if market not in self.target_coins:
                            self.target_coins.append(market)
                            logger.info(f"🎯 대상 코인 추가: {market}")

                    new_target_count = len(self.target_coins)
                    if new_target_count > original_target_count:
                        logger.info(f"🎯 대상 코인 목록 확장: {original_target_count}개 → {new_target_count}개")

            except Exception as e:
                logger.error(f"보유 코인 시세 조회 실패: {e}")

        except Exception as e:
            logger.error(f"기존 포지션 로드 실패: {e}")

    def run_smart_cycle(self):
        """스마트 하이브리드 사이클 실행"""
        logger.info("🧠 스마트 하이브리드 AI 시작!")
        logger.info("=" * 60)
        logger.info("🔥 자가학습 + 뉴스분석 + 강화학습 통합 AI")
        logger.info("=" * 60)

        # 이전 학습 상태 로드
        self.load_learning_state()

        # 🔥 기존 보유 코인 포지션 강제 로딩
        logger.info("📦 기존 보유 코인 포지션 로딩 중...")
        self.load_existing_positions()

        cycle_count = 0
        last_save_time = datetime.now()

        try:
            while True:
                cycle_count += 1
                logger.info(f"\n🔄 스마트 사이클 #{cycle_count}")

                # 0. 포지션 필드 정규화 (안전성 확보)
                self.normalize_position_fields()

                # 0.3. 캔들 히스토리 캐시 정리 (메모리 최적화)
                current_minute = int(time.time() / 60)
                expired_keys = [key for key in self.candle_history.keys()
                              if int(key.split('_')[-1]) < current_minute - 5]  # 5분 이전 데이터 삭제
                for key in expired_keys:
                    del self.candle_history[key]
                if expired_keys:
                    logger.debug(f"🧹 만료된 캔들 캐시 정리: {len(expired_keys)}개")

                # 0.5. 첫 번째 사이클에서 포지션 재로딩 확인
                if cycle_count == 1 and not self.positions:
                    logger.info("🔄 첫 사이클에서 포지션이 비어있음, 재로딩 시도...")
                    self.load_existing_positions()

                # 1. 감정 데이터 업데이트 (30분마다)
                self.update_sentiment_data()

                # 2. 모든 코인 시세 한 번에 조회 (API 최적화 + 캐싱)
                try:
                    all_tickers = self.get_cached_ticker(self.target_coins)
                    ticker_data = {ticker.market: ticker for ticker in all_tickers} if all_tickers else {}

                    if not ticker_data:
                        logger.warning("⚠️  시세 데이터 조회 실패, 다음 사이클에서 재시도")
                        time.sleep(5)  # 5초 대기 후 다음 사이클
                        continue

                except Exception as e:
                    logger.error(f"시세 조회 실패: {e}")
                    time.sleep(5)
                    continue

                # 3. 각 코인별 스마트 신호 및 거래 (캐시된 시세 사용)
                for market in self.target_coins:
                    try:
                        if market not in ticker_data:
                            logger.warning(f"⚠️  {market} 시세 데이터 없음")
                            continue

                        ticker = ticker_data[market]
                        signal, confidence, reasons = self.generate_smart_signal_cached(market, ticker)

                        if signal != 'HOLD':
                            logger.info(f"🎯 {market}: {signal} (신뢰도: {confidence:.2f})")
                            self.execute_smart_trade(market, signal, confidence, reasons, ticker.trade_price)
                        else:
                            price = ticker.trade_price
                            logger.info(f"⏸️  {market}: HOLD (가격: {price:,}원)")

                    except Exception as e:
                        logger.error(f"{market} 처리 오류: {e}")

                # 4. 리스크 관리 (캐시된 시세 사용)
                self.check_risk_management_cached(ticker_data)

                # 5. 현재 포지션 상태 (캐시된 시세 사용)
                if self.positions:
                    logger.info("📊 현재 포지션:")
                    total_position_value = 0
                    for market, pos in self.positions.items():
                        try:
                            if market in ticker_data:
                                current_price = ticker_data[market].trade_price
                                profit_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100

                                # 수량 추출 (quantity 또는 amount 키 지원)
                                quantity = pos.get('quantity') or pos.get('amount', 0)
                                position_value = quantity * current_price
                                total_position_value += position_value

                                duration = datetime.now() - pos['entry_time']
                                duration_str = f"{duration.days}d {duration.seconds//3600}h" if duration.days > 0 else f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"

                                source_indicator = "🔄" if pos.get('context', {}).get('source') == 'existing' else "🆕"
                                logger.info(f"   {source_indicator} {market}: {quantity:.8f}개 ({profit_pct:+.2f}%, {position_value:,.0f}원, {duration_str})")
                            else:
                                logger.warning(f"   {market}: 시세 데이터 없음")
                        except Exception as e:
                            logger.warning(f"   {market}: 포지션 상태 확인 실패 - {e}")

                    logger.info(f"💰 총 포지션 가치: {total_position_value:,.0f}원")
                else:
                    logger.info("📊 현재 포지션: 없음")

                # 5. 학습 상태 저장 (10분마다)
                if (datetime.now() - last_save_time).total_seconds() > 600:
                    self.save_learning_state()
                    last_save_time = datetime.now()

                # 6. 대기 (단타 최적화 - 15초 간격)
                cycle_interval = self.config.get('TRADING_CYCLE_SECONDS', 15)
                logger.info(f"⚡ {cycle_interval}초 대기 (단타 모드)...")
                time.sleep(cycle_interval)

        except KeyboardInterrupt:
            logger.info("\n🛑 사용자에 의한 정지")
        except Exception as e:
            logger.error(f"❌ 시스템 오류: {e}")
        finally:
            self.save_learning_state()
            logger.info("🏁 스마트 하이브리드 AI 종료")

def main():
    """메인 함수"""
    print("🧠 Smart Hybrid AI - 자가 학습 트레이딩 AI")
    print("=" * 60)
    print("📰 뉴스 분석 + 🔄 강화학습 + 📊 기술적 분석")
    print("=" * 60)

    try:
        ai = SmartHybridAI()
        ai.run_smart_cycle()
    except Exception as e:
        logger.error(f"AI 실행 실패: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())