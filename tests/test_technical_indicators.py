#!/usr/bin/env python3
"""
기술적 지표 테스트
RSI, MACD, 볼린저 밴드, 거래량 분석 테스트
"""

import os
import sys
from dotenv import load_dotenv

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from exchange.UpbitAPI import UpbitAPI
from technical_indicators import TechnicalIndicators

def test_technical_indicators():
    """기술적 지표 테스트"""
    load_dotenv()

    upbit = UpbitAPI()
    tech_analyzer = TechnicalIndicators()

    # 비트코인 30분봉 데이터 가져오기
    market = 'KRW-BTC'
    print(f"🔍 {market} 기술적 지표 분석 테스트")
    print("=" * 50)

    try:
        # 캔들 데이터 조회
        candles = upbit.GetCandles(market, 'minutes', unit=30, count=50)

        if not candles:
            print("❌ 캔들 데이터를 가져올 수 없습니다.")
            return

        print(f"📊 {len(candles)}개 캔들 데이터 수집 완료")

        # 가격과 거래량 데이터 추출
        prices = []
        volumes = []

        for candle in reversed(candles):  # 과거 -> 최신 순으로 변환
            prices.append(float(candle.get('trade_price', 0)))
            volumes.append(float(candle.get('candle_acc_trade_volume', 0)))

        print(f"💰 최신 가격: {prices[-1]:,}원")
        print(f"📈 가격 범위: {min(prices):,} ~ {max(prices):,}원")

        # 기술적 지표 계산
        print("\n🔧 기술적 지표 계산:")

        # RSI(9)
        rsi = tech_analyzer.calculate_rsi(prices, period=9)
        print(f"   RSI(9): {rsi:.1f}")

        # MACD(3-10-16)
        macd_data = tech_analyzer.calculate_macd(prices, fast=3, slow=10, signal=16)
        print(f"   MACD: {macd_data['macd']:.4f}")
        print(f"   Signal: {macd_data['signal']:.4f}")
        print(f"   Histogram: {macd_data['histogram']:.4f}")

        # 볼린저 밴드
        bollinger_data = tech_analyzer.calculate_bollinger_bands(prices, period=20, std_dev=2)
        print(f"   볼린저 상단: {bollinger_data['upper']:,.0f}원")
        print(f"   볼린저 중간: {bollinger_data['middle']:,.0f}원")
        print(f"   볼린저 하단: {bollinger_data['lower']:,.0f}원")
        print(f"   현재 위치: {bollinger_data['position']:.1f}%")

        # 거래량 분석
        volume_data = tech_analyzer.analyze_volume(volumes, prices)
        print(f"   평균 거래량: {volume_data['avg_volume']:.2f}")
        print(f"   거래량 비율: {volume_data['volume_ratio']:.2f}x")
        print(f"   거래량 추세: {volume_data['volume_price_trend']}")

        # 종합 기술적 신호
        print("\n🎯 종합 기술적 신호:")
        technical_signal = tech_analyzer.generate_technical_signal(
            rsi, macd_data, bollinger_data, volume_data
        )

        print(f"   신호: {technical_signal['signal']}")
        print(f"   신뢰도: {technical_signal['confidence']:.2f}")
        print(f"   이유:")
        for reason in technical_signal['reasons']:
            print(f"     - {reason}")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_technical_indicators()