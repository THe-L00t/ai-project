#!/usr/bin/env python3
"""
실시간 모니터링 대시보드 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.dashboard.app import app, socketio

if __name__ == '__main__':
    print("🚀 CoinTradingAI 대시보드 시작...")
    print("📊 브라우저에서 http://localhost:5000 접속")
    print("⚠️  Ctrl+C로 종료")

    socketio.run(
        app,
        debug=False,  # 프로덕션에서는 False
        host='0.0.0.0',  # 모든 인터페이스에서 접근 가능
        port=5000
    )