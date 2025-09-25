"""
실시간 모니터링 대시보드
Flask 기반 웹 인터페이스로 거래 상황을 실시간 모니터링합니다.
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import asyncio
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.trading.LiveTradingExecutor import LiveTradingExecutor
from src.trading.PositionManager import PositionManager
from src.trading.RiskManager import RiskManager
from src.data.DataStorage import DataStorage
from src.exchange.UpbitAPI import UpbitAPI


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardManager:
    """대시보드 관리자"""

    def __init__(self):
        self.trading_executor = None
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
        self.storage = DataStorage()
        self.upbit = None

        # 상태 데이터
        self.status = {
            'is_running': False,
            'last_update': None,
            'total_balance': 0,
            'daily_pnl': 0,
            'positions': {},
            'trades_today': 0,
            'risk_level': 'LOW'
        }

        self.price_data = {}
        self.performance_data = []

    def initialize_trading(self, access_key: str, secret_key: str):
        """거래 시스템 초기화"""
        try:
            self.upbit = UpbitAPI(access_key, secret_key)
            self.trading_executor = LiveTradingExecutor(access_key, secret_key)
            return True
        except Exception as e:
            print(f"거래 시스템 초기화 실패: {e}")
            return False

    async def update_data(self):
        """데이터 업데이트"""
        try:
            if not self.upbit:
                return

            # 잔고 조회
            krw_balance = await self.upbit.get_balance('KRW')

            # 현재 시세 조회
            symbols = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT']
            current_prices = {}

            for symbol in symbols:
                price = await self.upbit.get_current_price(symbol)
                if price:
                    current_prices[symbol] = price

            self.price_data = current_prices

            # 포지션 정보 업데이트
            if self.trading_executor:
                executor_status = self.trading_executor.get_status()
                self.status.update(executor_status)

            # 총 자산 계산
            total_asset_value = krw_balance
            for symbol, position in self.status.get('position_details', {}).items():
                current_price = current_prices.get(symbol, position.get('entry_price', 0))
                position_value = position.get('quantity', 0) * current_price
                total_asset_value += position_value

            self.status['total_balance'] = total_asset_value
            self.status['last_update'] = datetime.now().isoformat()

            # 성과 데이터 추가
            self.performance_data.append({
                'timestamp': datetime.now().isoformat(),
                'balance': total_asset_value,
                'krw_balance': krw_balance,
                'positions': len(self.status.get('position_details', {}))
            })

            # 최근 24시간 데이터만 보관
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_data = [
                data for data in self.performance_data
                if datetime.fromisoformat(data['timestamp']) > cutoff_time
            ]

        except Exception as e:
            print(f"데이터 업데이트 중 오류: {e}")

dashboard_manager = DashboardManager()


@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """시스템 상태 조회"""
    return jsonify(dashboard_manager.status)


@app.route('/api/prices')
def get_prices():
    """현재 가격 조회"""
    return jsonify(dashboard_manager.price_data)


@app.route('/api/performance')
def get_performance():
    """성과 데이터 조회"""
    return jsonify(dashboard_manager.performance_data)


@app.route('/api/positions')
def get_positions():
    """포지션 정보 조회"""
    if dashboard_manager.trading_executor:
        positions = dashboard_manager.trading_executor.positions
        current_prices = dashboard_manager.price_data

        position_data = []
        for symbol, position in positions.items():
            current_price = current_prices.get(symbol, position.get('entry_price', 0))

            position_info = {
                'symbol': symbol,
                'entry_price': position.get('entry_price', 0),
                'current_price': current_price,
                'quantity': position.get('quantity', 0),
                'pnl': (current_price - position.get('entry_price', 0)) * position.get('quantity', 0),
                'pnl_rate': ((current_price - position.get('entry_price', 0)) / position.get('entry_price', 1)) * 100,
                'entry_time': position.get('entry_time', datetime.now()).isoformat() if hasattr(position.get('entry_time', datetime.now()), 'isoformat') else str(position.get('entry_time', '')),
                'stop_loss': position.get('stop_loss', 0),
                'take_profit': position.get('take_profit', 0)
            }
            position_data.append(position_info)

        return jsonify(position_data)

    return jsonify([])


@app.route('/api/trades')
def get_trades():
    """거래 내역 조회"""
    if dashboard_manager.trading_executor:
        trades = dashboard_manager.trading_executor.trade_history[-20:]  # 최근 20개
        return jsonify(trades)
    return jsonify([])


@app.route('/api/risk')
def get_risk():
    """리스크 정보 조회"""
    risk_summary = dashboard_manager.risk_manager.get_risk_summary()
    return jsonify(risk_summary)


@app.route('/api/start_trading', methods=['POST'])
def start_trading():
    """거래 시작"""
    try:
        data = request.get_json()
        access_key = data.get('access_key')
        secret_key = data.get('secret_key')

        if not access_key or not secret_key:
            return jsonify({'success': False, 'message': 'API 키가 필요합니다.'})

        # 거래 시스템 초기화
        if dashboard_manager.initialize_trading(access_key, secret_key):
            # 비동기 거래 시작 (실제로는 별도 스레드에서 실행해야 함)
            dashboard_manager.status['is_running'] = True
            return jsonify({'success': True, 'message': '거래가 시작되었습니다.'})
        else:
            return jsonify({'success': False, 'message': '거래 시스템 초기화 실패'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'오류: {str(e)}'})


@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    """거래 중지"""
    try:
        if dashboard_manager.trading_executor:
            dashboard_manager.trading_executor.stop_trading()
            dashboard_manager.status['is_running'] = False
            return jsonify({'success': True, 'message': '거래가 중지되었습니다.'})
        else:
            return jsonify({'success': False, 'message': '활성화된 거래가 없습니다.'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'오류: {str(e)}'})


@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """비상 정지"""
    try:
        if dashboard_manager.trading_executor:
            asyncio.run(dashboard_manager.trading_executor._emergency_stop_all())
            return jsonify({'success': True, 'message': '비상 정지 실행됨'})
        else:
            return jsonify({'success': False, 'message': '활성화된 거래가 없습니다.'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'오류: {str(e)}'})


@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    print('클라이언트 연결됨')
    emit('status', dashboard_manager.status)


@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    print('클라이언트 연결 해제됨')


def background_thread():
    """백그라운드 데이터 업데이트 스레드"""
    while True:
        try:
            if dashboard_manager.upbit:
                # 비동기 함수를 동기 방식으로 실행
                asyncio.run(dashboard_manager.update_data())

                # 클라이언트에 실시간 데이터 전송
                socketio.emit('status_update', dashboard_manager.status)
                socketio.emit('price_update', dashboard_manager.price_data)

        except Exception as e:
            print(f"백그라운드 업데이트 오류: {e}")

        time.sleep(5)  # 5초마다 업데이트


# 백그라운드 스레드 시작
thread = threading.Thread(target=background_thread)
thread.daemon = True
thread.start()


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)