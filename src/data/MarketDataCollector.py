#!/usr/bin/env python3
"""
시장 데이터 수집기
- 업비트 API를 통한 실시간 시세 데이터 수집
- 기술적 지표 계산
- 데이터 저장 및 관리
"""

import time
import logging
import schedule
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import talib as ta

from ..exchange.UpbitAPI import UpbitAPI
from .DataStorage import DataStorage


@dataclass
class MarketDataPoint:
    """시장 데이터 포인트"""
    timestamp: datetime
    market: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_value: float
    
    # 기술적 지표
    sma_5: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    volume_sma_20: Optional[float] = None


class MarketDataCollector:
    """
    시장 데이터 수집기
    업비트 API를 사용하여 실시간 시장 데이터를 수집하고 분석
    """

    def __init__(self, upbit_api: UpbitAPI, data_storage: DataStorage, 
                 target_markets: List[str] = None):
        """
        초기화
        
        Args:
            upbit_api: 업비트 API 인스턴스
            data_storage: 데이터 저장소 인스턴스
            target_markets: 대상 마켓 리스트
        """
        self.upbit_api = upbit_api
        self.data_storage = data_storage
        self.target_markets = target_markets or [
            'KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-SOL'
        ]
        
        self.logger = logging.getLogger(__name__)
        self.is_collecting = False
        self.collection_thread = None
        
        # 데이터 수집 주기 (분)
        self.collection_intervals = {
            'ticker': 1,     # 1분마다 현재가
            'candle_1m': 1,  # 1분마다 1분봉 데이터
            'candle_5m': 5,  # 5분마다 5분봉 데이터
            'candle_1h': 60, # 1시간마다 1시간봉 데이터
        }
        
        # 기술적 지표 계산을 위한 데이터 버퍼
        self.price_buffers = {market: [] for market in self.target_markets}
        self.volume_buffers = {market: [] for market in self.target_markets}
        self.buffer_size = 200  # 최대 200개 데이터 보관
        
        # 웹소켓 콜백 설정
        self.realtime_callbacks = []
        
        self.logger.info(f"시장 데이터 수집기 초기화: {self.target_markets}")

    def AddRealtimeCallback(self, callback: Callable):
        """
        실시간 데이터 수신 콜백 추가
        
        Args:
            callback: 데이터 수신시 호출될 콜백 함수
        """
        self.realtime_callbacks.append(callback)

    def _CollectTicker(self):
        """현재 시세 데이터 수집"""
        try:
            tickers = self.upbit_api.GetTicker(self.target_markets)
            
            for ticker in tickers:
                # 데이터 저장
                ticker_data = {
                    'timestamp': datetime.now(),
                    'market': ticker.market,
                    'trade_price': ticker.trade_price,
                    'trade_volume': ticker.trade_volume,
                    'change': ticker.change,
                    'change_rate': ticker.change_rate,
                    'acc_trade_volume_24h': ticker.acc_trade_volume_24h,
                    'acc_trade_price_24h': ticker.acc_trade_price_24h
                }
                
                self.data_storage.SaveTickerData(ticker_data)
                
                # 실시간 콜백 실행
                for callback in self.realtime_callbacks:
                    callback('ticker', ticker_data)
                    
        except Exception as e:
            self.logger.error(f"티커 데이터 수집 오류: {e}")

    def _CollectCandles(self, period: str, unit: int = 1):
        """
        캔들 데이터 수집
        
        Args:
            period: 캔들 기간 ('minutes', 'days')
            unit: 단위 (1, 5, 60 등)
        """
        try:
            for market in self.target_markets:
                candles = self.upbit_api.GetCandles(
                    market=market, 
                    period=period, 
                    unit=unit, 
                    count=self.buffer_size
                )
                
                if not candles:
                    continue
                    
                # 데이터 처리 및 저장
                processed_candles = []
                
                for candle in candles:
                    candle_data = MarketDataPoint(
                        timestamp=datetime.fromisoformat(candle['candle_date_time_utc'].replace('Z', '+00:00')),
                        market=candle['market'],
                        open=float(candle['opening_price']),
                        high=float(candle['high_price']),
                        low=float(candle['low_price']),
                        close=float(candle['trade_price']),
                        volume=float(candle['candle_acc_trade_volume']),
                        trade_value=float(candle['candle_acc_trade_price'])
                    )
                    processed_candles.append(candle_data)
                
                # 기술적 지표 계산
                processed_candles = self._CalculateTechnicalIndicators(processed_candles)
                
                # 데이터베이스에 저장
                for candle in processed_candles:
                    self.data_storage.SaveCandleData(candle, f"{period}_{unit}")
                
                # 버퍼 업데이트
                self._UpdateBuffers(market, processed_candles)
                
        except Exception as e:
            self.logger.error(f"캔들 데이터 수집 오류: {e}")

    def _UpdateBuffers(self, market: str, candles: List[MarketDataPoint]):
        """
        가격/볼륨 버퍼 업데이트
        
        Args:
            market: 마켓 코드
            candles: 캔들 데이터 리스트
        """
        if market not in self.price_buffers:
            self.price_buffers[market] = []
            self.volume_buffers[market] = []
        
        # 최신 데이터로 버퍼 업데이트
        for candle in candles:
            self.price_buffers[market].append(candle.close)
            self.volume_buffers[market].append(candle.volume)
        
        # 버퍼 크기 제한
        if len(self.price_buffers[market]) > self.buffer_size:
            self.price_buffers[market] = self.price_buffers[market][-self.buffer_size:]
            self.volume_buffers[market] = self.volume_buffers[market][-self.buffer_size:]

    def _CalculateTechnicalIndicators(self, candles: List[MarketDataPoint]) -> List[MarketDataPoint]:
        """
        기술적 지표 계산
        
        Args:
            candles: 캔들 데이터 리스트
            
        Returns:
            기술적 지표가 추가된 캔들 데이터
        """
        if len(candles) < 50:  # 최소 데이터 개수 부족
            return candles
        
        try:
            # NumPy 배열로 변환
            closes = np.array([c.close for c in candles])
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            volumes = np.array([c.volume for c in candles])
            
            # 이동평균
            sma_5 = ta.SMA(closes, timeperiod=5)
            sma_20 = ta.SMA(closes, timeperiod=20)
            sma_50 = ta.SMA(closes, timeperiod=50)
            
            # 지수이동평균
            ema_12 = ta.EMA(closes, timeperiod=12)
            ema_26 = ta.EMA(closes, timeperiod=26)
            
            # RSI
            rsi_14 = ta.RSI(closes, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(closes)
            
            # 볼린저 밴드
            bollinger_upper, bollinger_middle, bollinger_lower = ta.BBANDS(closes)
            
            # ATR
            atr_14 = ta.ATR(highs, lows, closes, timeperiod=14)
            
            # 볼륨 이동평균
            volume_sma_20 = ta.SMA(volumes, timeperiod=20)
            
            # 결과를 객체에 대입
            for i, candle in enumerate(candles):
                candle.sma_5 = float(sma_5[i]) if not np.isnan(sma_5[i]) else None
                candle.sma_20 = float(sma_20[i]) if not np.isnan(sma_20[i]) else None
                candle.sma_50 = float(sma_50[i]) if not np.isnan(sma_50[i]) else None
                candle.ema_12 = float(ema_12[i]) if not np.isnan(ema_12[i]) else None
                candle.ema_26 = float(ema_26[i]) if not np.isnan(ema_26[i]) else None
                candle.rsi_14 = float(rsi_14[i]) if not np.isnan(rsi_14[i]) else None
                candle.macd = float(macd[i]) if not np.isnan(macd[i]) else None
                candle.macd_signal = float(macd_signal[i]) if not np.isnan(macd_signal[i]) else None
                candle.bollinger_upper = float(bollinger_upper[i]) if not np.isnan(bollinger_upper[i]) else None
                candle.bollinger_lower = float(bollinger_lower[i]) if not np.isnan(bollinger_lower[i]) else None
                candle.atr_14 = float(atr_14[i]) if not np.isnan(atr_14[i]) else None
                candle.volume_sma_20 = float(volume_sma_20[i]) if not np.isnan(volume_sma_20[i]) else None
            
            return candles
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 오류: {e}")
            return candles

    def GetLatestData(self, market: str, period: str = "minutes_1", count: int = 100) -> List[MarketDataPoint]:
        """
        최신 시장 데이터 조회
        
        Args:
            market: 마켓 코드
            period: 데이터 기간
            count: 데이터 개수
            
        Returns:
            시장 데이터 리스트
        """
        return self.data_storage.GetCandleData(market, period, count)

    def GetCurrentPrice(self, market: str) -> Optional[float]:
        """
        현재 가격 조회
        
        Args:
            market: 마켓 코드
            
        Returns:
            현재 가격
        """
        try:
            ticker = self.upbit_api.GetTicker([market])
            return ticker[0].trade_price if ticker else None
        except Exception as e:
            self.logger.error(f"현재 가격 조회 오류: {e}")
            return None

    def GetMarketSummary(self, market: str) -> Optional[Dict]:
        """
        마켓 요약 정보 조회
        
        Args:
            market: 마켓 코드
            
        Returns:
            마켓 요약 정보
        """
        try:
            ticker = self.upbit_api.GetTicker([market])
            if not ticker:
                return None
            
            t = ticker[0]
            return {
                'market': t.market,
                'current_price': t.trade_price,
                'change_rate': t.change_rate,
                'volume_24h': t.acc_trade_volume_24h,
                'trade_value_24h': t.acc_trade_price_24h,
                'high_52w': t.highest_52_week_price,
                'low_52w': t.lowest_52_week_price,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"마켓 요약 조회 오류: {e}")
            return None

    def StartCollection(self):
        """데이터 수집 시작"""
        if self.is_collecting:
            self.logger.warning("데이터 수집이 이미 시작되었습니다")
            return
        
        self.logger.info("시장 데이터 수집 시작")
        
        # 수집 스케줄 설정
        schedule.every(self.collection_intervals['ticker']).minutes.do(self._CollectTicker)
        schedule.every(self.collection_intervals['candle_1m']).minutes.do(
            lambda: self._CollectCandles('minutes', 1)
        )
        schedule.every(self.collection_intervals['candle_5m']).minutes.do(
            lambda: self._CollectCandles('minutes', 5)
        )
        schedule.every(self.collection_intervals['candle_1h']).minutes.do(
            lambda: self._CollectCandles('minutes', 60)
        )
        
        # 실시간 웹소켓 연결
        self._StartRealtimeCollection()
        
        self.is_collecting = True
        
        # 수집 스레드 시작
        self.collection_thread = threading.Thread(target=self._CollectionLoop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("시장 데이터 수집 시작 완료")

    def _StartRealtimeCollection(self):
        """실시간 데이터 수집 시작"""
        try:
            def on_realtime_data(data):
                """실시간 데이터 수신 처리"""
                try:
                    processed_data = {
                        'timestamp': datetime.now(),
                        'market': data.get('code', ''),
                        'trade_price': data.get('trade_price', 0),
                        'change_rate': data.get('signed_change_rate', 0),
                        'volume': data.get('acc_trade_volume_24h', 0)
                    }
                    
                    # 실시간 콜백 실행
                    for callback in self.realtime_callbacks:
                        callback('realtime', processed_data)
                        
                except Exception as e:
                    self.logger.error(f"실시간 데이터 처리 오류: {e}")
            
            # 웹소켓 연결
            self.upbit_api.ConnectWebSocket(on_message=on_realtime_data)
            self.upbit_api.SubscribeTicker(self.target_markets)
            
            # 비블로킹 웹소켓 실행
            ws_thread = threading.Thread(
                target=self.upbit_api.StartWebSocket, 
                daemon=True
            )
            ws_thread.start()
            
        except Exception as e:
            self.logger.error(f"실시간 데이터 수집 시작 오류: {e}")

    def _CollectionLoop(self):
        """데이터 수집 메인 루프"""
        while self.is_collecting:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"데이터 수집 루프 오류: {e}")
                time.sleep(5)

    def StopCollection(self):
        """데이터 수집 중지"""
        if not self.is_collecting:
            return
        
        self.logger.info("시장 데이터 수집 중지")
        
        self.is_collecting = False
        
        # 웹소켓 연결 종료
        self.upbit_api.StopWebSocket()
        
        # 스케줄 클리어
        schedule.clear()
        
        # 스레드 종료 대기
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        self.logger.info("시장 데이터 수집 중지 완료")

    def GetCollectionStatus(self) -> Dict:
        """
        데이터 수집 상태 조회
        
        Returns:
            수집 상태 정보
        """
        return {
            'is_collecting': self.is_collecting,
            'target_markets': self.target_markets,
            'collection_intervals': self.collection_intervals,
            'buffer_sizes': {market: len(self.price_buffers[market]) 
                           for market in self.target_markets},
            'realtime_callbacks_count': len(self.realtime_callbacks)
        }

    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            self.StopCollection()
        except:
            pass