#!/usr/bin/env python3
"""
데이터 저장소
- SQLite 데이터베이스를 사용한 데이터 저장
- 시장 데이터, 뉴스 데이터, 거래 데이터 관리
- 데이터 조회 및 분석 기능
"""

import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import json
from contextlib import contextmanager
import threading


class DataStorage:
    """
    데이터 저장소 클래스
    SQLite를 사용하여 모든 데이터를 안전하고 효율적으로 저장
    """

    def __init__(self, db_path: str = "data/trading_data.db"):
        """
        초기화

        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

        # 디렉토리 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 데이터베이스 초기화
        self._InitializeDatabase()

        self.logger.info(f"데이터 저장소 초기화: {db_path}")

    @contextmanager
    def _GetConnection(self):
        """
        데이터베이스 연결 컨텍스트 매니저
        스레드 안전성을 위해 락 사용
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
            try:
                yield conn
            finally:
                conn.close()

    def _InitializeDatabase(self):
        """데이터베이스 테이블 초기화"""
        with self._GetConnection() as conn:
            cursor = conn.cursor()

            # 시세 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ticker_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market TEXT NOT NULL,
                    trade_price REAL NOT NULL,
                    trade_volume REAL NOT NULL,
                    change TEXT,
                    change_rate REAL,
                    acc_trade_volume_24h REAL,
                    acc_trade_price_24h REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 캔들 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candle_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market TEXT NOT NULL,
                    period TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    trade_value REAL NOT NULL,

                    -- 기술적 지표
                    sma_5 REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    rsi_14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    bollinger_upper REAL,
                    bollinger_lower REAL,
                    atr_14 REAL,
                    volume_sma_20 REAL,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, market, period)
                )
            ''')

            # 뉴스 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT NOT NULL,
                    url TEXT UNIQUE,
                    published_date TEXT NOT NULL,
                    language TEXT NOT NULL,

                    -- 감정 분석 결과
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    confidence REAL,

                    -- 코인 관련도
                    relevant_coins TEXT,  -- JSON 배열
                    relevance_score REAL,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 거래 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,  -- buy/sell
                    type TEXT NOT NULL,  -- market/limit
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    total_amount REAL NOT NULL,
                    fee REAL NOT NULL,

                    -- 주문 정보
                    order_uuid TEXT,
                    order_state TEXT,

                    -- AI 결정 정보
                    ai_confidence REAL,
                    ai_signal_type TEXT,
                    ai_reasoning TEXT,

                    -- 성능 지표
                    profit_loss REAL,
                    profit_loss_rate REAL,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 모델 성능 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,

                    -- 성능 지표
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,

                    -- 거래 성능
                    total_return REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,

                    -- 추가 메타데이터
                    metadata TEXT,  -- JSON

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_market_time ON ticker_data(market, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_candle_market_period_time ON candle_data(market, period, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_published_date ON news_data(published_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_market_time ON trade_data(market, timestamp)')

            conn.commit()
            self.logger.info("데이터베이스 테이블 초기화 완료")

    # ==========================================================================
    # 시세 데이터 저장
    # ==========================================================================

    def SaveTickerData(self, ticker_data: Dict):
        """
        실시간 시세 데이터 저장

        Args:
            ticker_data: 시세 데이터 딕셔너리
        """
        try:
            with self._GetConnection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ticker_data (
                        timestamp, market, trade_price, trade_volume,
                        change, change_rate, acc_trade_volume_24h, acc_trade_price_24h
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker_data['timestamp'].isoformat(),
                    ticker_data['market'],
                    ticker_data['trade_price'],
                    ticker_data['trade_volume'],
                    ticker_data.get('change'),
                    ticker_data.get('change_rate'),
                    ticker_data.get('acc_trade_volume_24h'),
                    ticker_data.get('acc_trade_price_24h')
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"시세 데이터 저장 오류: {e}")

    def SaveCandleData(self, candle_data, period: str):
        """
        캔들 데이터 저장

        Args:
            candle_data: MarketDataPoint 객체 또는 딕셔너리
            period: 데이터 기간 (minutes_1, minutes_5 등)
        """
        try:
            with self._GetConnection() as conn:
                cursor = conn.cursor()

                # 데이터 형식 정규화
                if hasattr(candle_data, 'timestamp'):  # MarketDataPoint 객체
                    data = asdict(candle_data)
                else:  # 딕셔너리
                    data = candle_data

                cursor.execute('''
                    INSERT OR REPLACE INTO candle_data (
                        timestamp, market, period, open_price, high_price, low_price, close_price,
                        volume, trade_value, sma_5, sma_20, sma_50, ema_12, ema_26,
                        rsi_14, macd, macd_signal, bollinger_upper, bollinger_lower,
                        atr_14, volume_sma_20
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['timestamp'].isoformat() if isinstance(data['timestamp'], datetime) else data['timestamp'],
                    data['market'],
                    period,
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close'],
                    data['volume'],
                    data['trade_value'],
                    data.get('sma_5'),
                    data.get('sma_20'),
                    data.get('sma_50'),
                    data.get('ema_12'),
                    data.get('ema_26'),
                    data.get('rsi_14'),
                    data.get('macd'),
                    data.get('macd_signal'),
                    data.get('bollinger_upper'),
                    data.get('bollinger_lower'),
                    data.get('atr_14'),
                    data.get('volume_sma_20')
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"캔들 데이터 저장 오류: {e}")

    # ==========================================================================
    # 뉴스 데이터 저장
    # ==========================================================================

    def SaveNewsData(self, news_data: Dict):
        """
        뉴스 데이터 저장

        Args:
            news_data: 뉴스 데이터 딕셔너리
        """
        try:
            with self._GetConnection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO news_data (
                        title, content, source, url, published_date, language,
                        sentiment_score, sentiment_label, confidence,
                        relevant_coins, relevance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    news_data['title'],
                    news_data.get('content'),
                    news_data['source'],
                    news_data.get('url'),
                    news_data['published_date'].isoformat() if isinstance(news_data['published_date'], datetime) else news_data['published_date'],
                    news_data['language'],
                    news_data.get('sentiment_score'),
                    news_data.get('sentiment_label'),
                    news_data.get('confidence'),
                    json.dumps(news_data.get('relevant_coins', [])),
                    news_data.get('relevance_score')
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"뉴스 데이터 저장 오류: {e}")

    # ==========================================================================
    # 거래 데이터 저장
    # ==========================================================================

    def SaveTradeData(self, trade_data: Dict):
        """
        거래 데이터 저장

        Args:
            trade_data: 거래 데이터 딕셔너리
        """
        try:
            with self._GetConnection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trade_data (
                        timestamp, market, side, type, price, volume, total_amount, fee,
                        order_uuid, order_state, ai_confidence, ai_signal_type, ai_reasoning,
                        profit_loss, profit_loss_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['timestamp'].isoformat() if isinstance(trade_data['timestamp'], datetime) else trade_data['timestamp'],
                    trade_data['market'],
                    trade_data['side'],
                    trade_data['type'],
                    trade_data['price'],
                    trade_data['volume'],
                    trade_data['total_amount'],
                    trade_data['fee'],
                    trade_data.get('order_uuid'),
                    trade_data.get('order_state'),
                    trade_data.get('ai_confidence'),
                    trade_data.get('ai_signal_type'),
                    trade_data.get('ai_reasoning'),
                    trade_data.get('profit_loss'),
                    trade_data.get('profit_loss_rate')
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"거래 데이터 저장 오류: {e}")

    # ==========================================================================
    # 데이터 조회
    # ==========================================================================

    def GetCandleData(self, market: str, period: str, count: int = 100) -> List[Dict]:
        """
        캔들 데이터 조회

        Args:
            market: 마켓 코드
            period: 데이터 기간
            count: 데이터 개수

        Returns:
            캔들 데이터 리스트
        """
        try:
            with self._GetConnection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM candle_data
                    WHERE market = ? AND period = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (market, period, count))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"캔들 데이터 조회 오류: {e}")
            return []

    def GetTickerData(self, market: str, hours: int = 24) -> List[Dict]:
        """
        시세 데이터 조회

        Args:
            market: 마켓 코드
            hours: 조회 시간 범위 (시간)

        Returns:
            시세 데이터 리스트
        """
        try:
            start_time = datetime.now() - timedelta(hours=hours)

            with self._GetConnection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM ticker_data
                    WHERE market = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (market, start_time.isoformat()))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"시세 데이터 조회 오류: {e}")
            return []

    def GetNewsData(self, hours: int = 24, limit: int = 100,
                   relevant_coin: str = None) -> List[Dict]:
        """
        뉴스 데이터 조회

        Args:
            hours: 조회 시간 범위
            limit: 최대 결과 수
            relevant_coin: 특정 코인 관련 뉴스만 조회

        Returns:
            뉴스 데이터 리스트
        """
        try:
            start_time = datetime.now() - timedelta(hours=hours)

            with self._GetConnection() as conn:
                cursor = conn.cursor()

                if relevant_coin:
                    cursor.execute('''
                        SELECT * FROM news_data
                        WHERE published_date >= ?
                        AND (relevant_coins LIKE ? OR title LIKE ? OR content LIKE ?)
                        ORDER BY published_date DESC
                        LIMIT ?
                    ''', (
                        start_time.isoformat(),
                        f'%{relevant_coin}%',
                        f'%{relevant_coin}%',
                        f'%{relevant_coin}%',
                        limit
                    ))
                else:
                    cursor.execute('''
                        SELECT * FROM news_data
                        WHERE published_date >= ?
                        ORDER BY published_date DESC
                        LIMIT ?
                    ''', (start_time.isoformat(), limit))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"뉴스 데이터 조회 오류: {e}")
            return []

    def GetTradeHistory(self, market: str = None, days: int = 30) -> List[Dict]:
        """
        거래 내역 조회

        Args:
            market: 마켓 코드 (없으면 전체)
            days: 조회 기간 (일)

        Returns:
            거래 내역 리스트
        """
        try:
            start_time = datetime.now() - timedelta(days=days)

            with self._GetConnection() as conn:
                cursor = conn.cursor()

                if market:
                    cursor.execute('''
                        SELECT * FROM trade_data
                        WHERE market = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (market, start_time.isoformat()))
                else:
                    cursor.execute('''
                        SELECT * FROM trade_data
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (start_time.isoformat(),))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"거래 내역 조회 오류: {e}")
            return []

    def Close(self):
        """리소스 정리"""
        self.logger.info("데이터 저장소 종료")

    def __del__(self):
        """소멸자"""
        try:
            self.Close()
        except:
            pass