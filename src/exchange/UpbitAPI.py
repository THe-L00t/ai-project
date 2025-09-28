#!/usr/bin/env python3
"""
업비트 API 연동 모듈
- 업비트 REST API 및 WebSocket 연동
- 주문, 조회, 입출금 기능
- 실시간 시세 데이터 수집
"""

import os
import time
import hmac
import hashlib
import uuid
import jwt
import requests
import websocket
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlencode
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """주문 타입"""
    BUY = "bid"
    SELL = "ask"


class OrderSide(Enum):
    """주문 방식"""
    LIMIT = "limit"
    MARKET = "price"


@dataclass
class MarketInfo:
    """마켓 정보"""
    market: str
    korean_name: str
    english_name: str


@dataclass
class TickerInfo:
    """현재 시세 정보"""
    market: str
    trade_price: float
    trade_volume: float
    trade_date: str
    trade_time: str
    change: str
    change_price: float
    change_rate: float
    prev_closing_price: float
    acc_trade_volume_24h: float
    acc_trade_price_24h: float
    highest_52_week_price: float
    highest_52_week_date: str
    lowest_52_week_price: float
    lowest_52_week_date: str


@dataclass
class OrderInfo:
    """주문 정보"""
    uuid: str
    side: str
    ord_type: str
    price: Optional[float]
    volume: Optional[float]
    remaining_volume: float
    reserved_fee: float
    remaining_fee: float
    paid_fee: float
    locked: float
    executed_volume: float
    trade_count: int
    market: str
    created_at: str
    updated_at: str
    state: str


class UpbitAPI:
    """
    업비트 API 클라이언트
    안정적인 거래소 연동을 위한 포괄적 기능 제공
    """

    def __init__(self, access_key: str = None, secret_key: str = None):
        """
        업비트 API 초기화
        
        Args:
            access_key: 업비트 액세스 키
            secret_key: 업비트 시크릿 키
        """
        self.access_key = access_key or os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('UPBIT_SECRET_KEY')
        
        # API 엔드포인트
        self.base_url = "https://api.upbit.com/v1"
        self.ws_url = "wss://api.upbit.com/websocket/v1"
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CoinTradingAI/1.0',
            'Content-Type': 'application/json'
        })
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 웹소켓 연결
        self.ws_connection = None
        self.ws_callbacks = {}
        
        # API 호출 제한 관리
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # 1분
        self.max_requests_per_minute = 100
        
        self.logger.info("업비트 API 클라이언트 초기화 완료")

    def _GenerateJWT(self, query_params: Dict = None) -> str:
        """
        JWT 토큰 생성
        
        Args:
            query_params: 쿼리 파라미터
            
        Returns:
            JWT 토큰 문자열
        """
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
        }
        
        if query_params:
            query_string = urlencode(query_params, doseq=True, safe='')
            m = hashlib.sha512()
            m.update(query_string.encode())
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def _CheckRateLimit(self):
        """API 호출 제한 확인"""
        current_time = time.time()
        
        # 1분이 지났으면 카운터 리셋
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = current_time
        
        # 요청 제한 확인
        if self.request_count >= self.max_requests_per_minute:
            wait_time = self.rate_limit_window - (current_time - self.last_request_time)
            if wait_time > 0:
                self.logger.warning(f"API 요청 제한 도달. {wait_time:.1f}초 대기")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1

    def _MakeRequest(self, method: str, endpoint: str, params: Dict = None, 
                    auth_required: bool = False) -> Dict:
        """
        API 요청 실행
        
        Args:
            method: HTTP 메서드 (GET, POST, DELETE)
            endpoint: API 엔드포인트
            params: 요청 파라미터
            auth_required: 인증 필요 여부
            
        Returns:
            API 응답 데이터
        """
        self._CheckRateLimit()
        
        url = f"{self.base_url}{endpoint}"
        headers = self.session.headers.copy()
        
        # 인증이 필요한 경우 JWT 토큰 추가
        if auth_required:
            if not self.access_key or not self.secret_key:
                raise ValueError("API 키가 설정되지 않았습니다")
            
            jwt_token = self._GenerateJWT(params)
            headers['Authorization'] = f'Bearer {jwt_token}'
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, json=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")

            # 성공 상태 코드 확인 (200, 201, 204 모두 성공)
            if response.status_code not in [200, 201, 204]:
                error_content = response.text
                self.logger.error(f"API 오류 응답 ({response.status_code}): {error_content}")
            else:
                self.logger.debug(f"API 성공 응답 ({response.status_code})")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API 요청 실패: {e}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"응답 내용: {e.response.text}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 디코딩 실패: {e}")
            raise

    # ==========================================================================
    # 마켓 정보 API
    # ==========================================================================
    
    def GetAllMarkets(self) -> List[MarketInfo]:
        """
        업비트에서 거래 가능한 마켓 목록 조회
        
        Returns:
            마켓 정보 리스트
        """
        try:
            response = self._MakeRequest('GET', '/market/all')
            return [MarketInfo(
                market=item['market'],
                korean_name=item['korean_name'],
                english_name=item['english_name']
            ) for item in response]
        except Exception as e:
            self.logger.error(f"마켓 목록 조회 실패: {e}")
            return []

    def GetTicker(self, markets: List[str]) -> List[TickerInfo]:
        """
        현재 시세 정보 조회
        
        Args:
            markets: 마켓 코드 리스트 (예: ['KRW-BTC', 'KRW-ETH'])
            
        Returns:
            현재 시세 정보 리스트
        """
        try:
            params = {'markets': ','.join(markets)}
            response = self._MakeRequest('GET', '/ticker', params)
            
            return [TickerInfo(
                market=item['market'],
                trade_price=float(item['trade_price']),
                trade_volume=float(item['trade_volume']),
                trade_date=item['trade_date'],
                trade_time=item['trade_time'],
                change=item['change'],
                change_price=float(item['change_price']),
                change_rate=float(item['change_rate']),
                prev_closing_price=float(item['prev_closing_price']),
                acc_trade_volume_24h=float(item['acc_trade_volume_24h']),
                acc_trade_price_24h=float(item['acc_trade_price_24h']),
                highest_52_week_price=float(item['highest_52_week_price']),
                highest_52_week_date=item['highest_52_week_date'],
                lowest_52_week_price=float(item['lowest_52_week_price']),
                lowest_52_week_date=item['lowest_52_week_date']
            ) for item in response]
            
        except Exception as e:
            self.logger.error(f"현재 시세 조회 실패: {e}")
            return []

    def GetCandles(self, market: str, period: str = 'minutes', 
                  unit: int = 1, count: int = 200) -> List[Dict]:
        """
        캔들 차트 데이터 조회
        
        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            period: 캔들 기간 ('minutes', 'days', 'weeks', 'months')
            unit: 단위 (분봉: 1,3,5,15,10,30,60,240, 일봉: 1)
            count: 캔들 개수 (최대 200)
            
        Returns:
            캔들 데이터 리스트
        """
        try:
            if period == 'minutes':
                endpoint = f'/candles/minutes/{unit}'
            else:
                endpoint = f'/candles/{period}'
            
            params = {
                'market': market,
                'count': min(count, 200)
            }
            
            response = self._MakeRequest('GET', endpoint, params)
            return response
            
        except Exception as e:
            self.logger.error(f"캔들 데이터 조회 실패: {e}")
            return []

    def GetMinuteCandles(self, market: str, count: int = 50) -> List[Dict]:
        """
        1분봉 캔들 데이터 조회 (편의 메서드)

        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            count: 캔들 개수 (최대 200)

        Returns:
            1분봉 캔들 데이터 리스트
        """
        return self.GetCandles(market, 'minutes', 1, count)

    # ==========================================================================
    # 계정 정보 API
    # ==========================================================================
    
    def GetAccountInfo(self) -> List[Dict]:
        """
        계정 정보 조회
        
        Returns:
            계정별 보유 자산 정보
        """
        try:
            response = self._MakeRequest('GET', '/accounts', auth_required=True)
            return response
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {e}")
            return []

    def GetBalance(self, currency: str = None) -> Dict:
        """
        잔고 조회
        
        Args:
            currency: 특정 화폐 조회 (없으면 전체)
            
        Returns:
            잔고 정보
        """
        try:
            accounts = self.GetAccountInfo()
            
            if currency:
                for account in accounts:
                    if account['currency'] == currency:
                        return {
                            'currency': account['currency'],
                            'balance': float(account['balance']),
                            'locked': float(account['locked']),
                            'avg_buy_price': float(account['avg_buy_price']),
                            'avg_buy_price_modified': account['avg_buy_price_modified'],
                            'unit_currency': account['unit_currency']
                        }
                return {}
            else:
                return {account['currency']: {
                    'balance': float(account['balance']),
                    'locked': float(account['locked']),
                    'avg_buy_price': float(account['avg_buy_price'])
                } for account in accounts}
                
        except Exception as e:
            self.logger.error(f"잔고 조회 실패: {e}")
            return {}

    # ==========================================================================
    # 주문 API
    # ==========================================================================
    
    def PlaceOrder(self, market: str, side: str, ord_type: str, 
                  volume: float = None, price: float = None) -> Optional[Dict]:
        """
        주문하기
        
        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            side: 주문 종류 ('bid': 매수, 'ask': 매도)
            ord_type: 주문 타입 ('limit': 지정가, 'price': 시장가매수, 'market': 시장가매도)
            volume: 주문 수량 (지정가, 시장가 매도시 필수)
            price: 주문 가격 (지정가, 시장가 매수시 필수)
            
        Returns:
            주문 결과
        """
        try:
            params = {
                'market': market,
                'side': side,
                'ord_type': ord_type
            }

            if volume is not None:
                params['volume'] = str(volume)
            if price is not None:
                params['price'] = str(price)

            response = self._MakeRequest('POST', '/orders', params, auth_required=True)

            if response:
                self.logger.info(f"주문 성공: {market} {side} {ord_type} {volume} {price}")
                self.logger.info(f"주문 UUID: {response.get('uuid', 'N/A')}")
                return response
            else:
                self.logger.error("주문 응답이 비어있음")
                return None

        except Exception as e:
            self.logger.error(f"주문 실패: {e}")
            return None

    def CancelOrder(self, uuid: str) -> Optional[Dict]:
        """
        주문 취소
        
        Args:
            uuid: 취소할 주문 UUID
            
        Returns:
            취소 결과
        """
        try:
            params = {'uuid': uuid}
            response = self._MakeRequest('DELETE', '/order', params, auth_required=True)
            
            self.logger.info(f"주문 취소 성공: {uuid}")
            return response
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {e}")
            return None

    def GetOrder(self, uuid: str) -> Optional[OrderInfo]:
        """
        개별 주문 조회
        
        Args:
            uuid: 주문 UUID
            
        Returns:
            주문 정보
        """
        try:
            params = {'uuid': uuid}
            response = self._MakeRequest('GET', '/order', params, auth_required=True)
            
            return OrderInfo(
                uuid=response['uuid'],
                side=response['side'],
                ord_type=response['ord_type'],
                price=float(response['price']) if response['price'] else None,
                volume=float(response['volume']) if response['volume'] else None,
                remaining_volume=float(response['remaining_volume']),
                reserved_fee=float(response['reserved_fee']),
                remaining_fee=float(response['remaining_fee']),
                paid_fee=float(response['paid_fee']),
                locked=float(response['locked']),
                executed_volume=float(response['executed_volume']),
                trade_count=response['trade_count'],
                market=response['market'],
                created_at=response['created_at'],
                updated_at=response['updated_at'],
                state=response['state']
            )
            
        except Exception as e:
            self.logger.error(f"주문 조회 실패: {e}")
            return None

    def GetOpenOrders(self, market: str = None) -> List[OrderInfo]:
        """
        미체결 주문 조회
        
        Args:
            market: 특정 마켓 필터 (없으면 전체)
            
        Returns:
            미체결 주문 리스트
        """
        try:
            params = {'state': 'wait'}
            if market:
                params['market'] = market
                
            response = self._MakeRequest('GET', '/orders', params, auth_required=True)
            
            return [OrderInfo(
                uuid=order['uuid'],
                side=order['side'],
                ord_type=order['ord_type'],
                price=float(order['price']) if order['price'] else None,
                volume=float(order['volume']) if order['volume'] else None,
                remaining_volume=float(order['remaining_volume']),
                reserved_fee=float(order['reserved_fee']),
                remaining_fee=float(order['remaining_fee']),
                paid_fee=float(order['paid_fee']),
                locked=float(order['locked']),
                executed_volume=float(order['executed_volume']),
                trade_count=order['trade_count'],
                market=order['market'],
                created_at=order['created_at'],
                updated_at=order['updated_at'],
                state=order['state']
            ) for order in response]
            
        except Exception as e:
            self.logger.error(f"미체결 주문 조회 실패: {e}")
            return []

    # ==========================================================================
    # 편의 메서드
    # ==========================================================================
    
    def BuyMarket(self, market: str, amount: float) -> Optional[Dict]:
        """
        시장가 매수
        
        Args:
            market: 마켓 코드
            amount: 매수 금액 (원)
            
        Returns:
            주문 결과
        """
        return self.PlaceOrder(market, 'bid', 'price', price=amount)

    def BuyLimit(self, market: str, volume: float, price: float) -> Optional[Dict]:
        """
        지정가 매수
        
        Args:
            market: 마켓 코드
            volume: 매수 수량
            price: 매수 가격
            
        Returns:
            주문 결과
        """
        return self.PlaceOrder(market, 'bid', 'limit', volume=volume, price=price)

    def SellMarket(self, market: str, volume: float) -> Optional[Dict]:
        """
        시장가 매도
        
        Args:
            market: 마켓 코드
            volume: 매도 수량
            
        Returns:
            주문 결과
        """
        return self.PlaceOrder(market, 'ask', 'market', volume=volume)

    def SellLimit(self, market: str, volume: float, price: float) -> Optional[Dict]:
        """
        지정가 매도
        
        Args:
            market: 마켓 코드
            volume: 매도 수량
            price: 매도 가격
            
        Returns:
            주문 결과
        """
        return self.PlaceOrder(market, 'ask', 'limit', volume=volume, price=price)

    def GetKRWBalance(self) -> float:
        """
        원화 잔고 조회
        
        Returns:
            원화 잔고
        """
        balance_info = self.GetBalance('KRW')
        return balance_info.get('balance', 0.0) if balance_info else 0.0

    def GetCoinBalance(self, coin: str) -> float:
        """
        특정 코인 잔고 조회
        
        Args:
            coin: 코인 심볼 (예: 'BTC', 'ETH')
            
        Returns:
            코인 잔고
        """
        balance_info = self.GetBalance(coin)
        return balance_info.get('balance', 0.0) if balance_info else 0.0

    # ==========================================================================
    # WebSocket 연결
    # ==========================================================================
    
    def ConnectWebSocket(self, on_message: Callable = None, on_error: Callable = None):
        """
        웹소켓 연결 설정
        
        Args:
            on_message: 메시지 수신 콜백
            on_error: 에러 콜백
        """
        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                if on_message:
                    on_message(data)
                
                # 등록된 콜백 실행
                market = data.get('code')
                if market in self.ws_callbacks:
                    for callback in self.ws_callbacks[market]:
                        callback(data)
                        
            except Exception as e:
                self.logger.error(f"웹소켓 메시지 처리 오류: {e}")

        def on_ws_error(ws, error):
            self.logger.error(f"웹소켓 오류: {error}")
            if on_error:
                on_error(error)

        def on_ws_close(ws, close_status_code, close_msg):
            self.logger.info("웹소켓 연결 종료")

        def on_ws_open(ws):
            self.logger.info("웹소켓 연결 성공")

        websocket.enableTrace(True)
        self.ws_connection = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close,
            on_open=on_ws_open
        )

    def SubscribeTicker(self, markets: List[str], callback: Callable = None):
        """
        실시간 시세 구독
        
        Args:
            markets: 구독할 마켓 리스트
            callback: 데이터 수신 콜백
        """
        if not self.ws_connection:
            raise RuntimeError("웹소켓 연결이 설정되지 않았습니다")
        
        # 콜백 등록
        if callback:
            for market in markets:
                if market not in self.ws_callbacks:
                    self.ws_callbacks[market] = []
                self.ws_callbacks[market].append(callback)
        
        # 구독 메시지 전송
        subscribe_message = {
            "ticket": str(uuid.uuid4()),
            "type": "ticker",
            "codes": markets,
            "isOnlySnapshot": False,
            "isOnlyRealtime": True
        }
        
        self.ws_connection.send(json.dumps([subscribe_message]))
        self.logger.info(f"실시간 시세 구독: {markets}")

    def StartWebSocket(self):
        """
        웹소켓 실행 (블로킹)
        """
        if not self.ws_connection:
            raise RuntimeError("웹소켓 연결이 설정되지 않았습니다")
        
        self.ws_connection.run_forever()

    def StopWebSocket(self):
        """
        웹소켓 연결 종료
        """
        if self.ws_connection:
            self.ws_connection.close()
            self.ws_connection = None
            self.ws_callbacks.clear()
            self.logger.info("웹소켓 연결 종료")

    # ==========================================================================
    # 지갑 연결 및 입출금 API
    # ==========================================================================

    def GetCoinAddresses(self) -> List[Dict]:
        """
        코인 출금 주소 목록 조회

        Returns:
            출금 주소 정보 리스트
        """
        try:
            response = self._MakeRequest('GET', '/deposits/coin_addresses', auth_required=True)
            return response
        except Exception as e:
            self.logger.error(f"출금 주소 목록 조회 실패: {e}")
            return []

    def GetCoinAddress(self, currency: str, net_type: str = None) -> Optional[Dict]:
        """
        개별 코인 출금 주소 조회

        Args:
            currency: 화폐 심볼 (예: 'BTC', 'ETH')
            net_type: 네트워크 타입 (선택사항)

        Returns:
            출금 주소 정보
        """
        try:
            params = {'currency': currency}
            if net_type:
                params['net_type'] = net_type

            response = self._MakeRequest('GET', '/deposits/coin_address', params, auth_required=True)
            return response
        except Exception as e:
            self.logger.error(f"출금 주소 조회 실패: {e}")
            return None

    def CreateCoinAddress(self, currency: str, net_type: str = None) -> Optional[Dict]:
        """
        코인 출금 주소 생성 요청

        Args:
            currency: 화폐 심볼 (예: 'BTC', 'ETH')
            net_type: 네트워크 타입 (선택사항)

        Returns:
            출금 주소 생성 결과
        """
        try:
            params = {'currency': currency}
            if net_type:
                params['net_type'] = net_type

            response = self._MakeRequest('POST', '/deposits/generate_coin_address', params, auth_required=True)
            self.logger.info(f"{currency} 출금 주소 생성 요청 완료")
            return response
        except Exception as e:
            self.logger.error(f"출금 주소 생성 실패: {e}")
            return None

    def WithdrawCoin(self, currency: str, amount: float, address: str,
                    secondary_address: str = None, transaction_type: str = 'default') -> Optional[Dict]:
        """
        코인 출금하기

        Args:
            currency: 화폐 심볼 (예: 'BTC', 'ETH')
            amount: 출금 수량
            address: 출금 주소
            secondary_address: 2차 주소 (필요시)
            transaction_type: 출금 유형 ('default' 또는 'internal')

        Returns:
            출금 결과
        """
        try:
            params = {
                'currency': currency,
                'amount': str(amount),
                'address': address,
                'transaction_type': transaction_type
            }

            if secondary_address:
                params['secondary_address'] = secondary_address

            response = self._MakeRequest('POST', '/withdraws/coin', params, auth_required=True)

            self.logger.info(f"코인 출금 요청 완료: {currency} {amount} → {address}")
            return response
        except Exception as e:
            self.logger.error(f"코인 출금 실패: {e}")
            return None

    def WithdrawKRW(self, amount: int) -> Optional[Dict]:
        """
        원화 출금하기

        Args:
            amount: 출금 금액 (원)

        Returns:
            출금 결과
        """
        try:
            params = {'amount': str(amount)}
            response = self._MakeRequest('POST', '/withdraws/krw', params, auth_required=True)

            self.logger.info(f"원화 출금 요청 완료: {amount:,}원")
            return response
        except Exception as e:
            self.logger.error(f"원화 출금 실패: {e}")
            return None

    def GetDepositHistory(self, currency: str = None, state: str = None,
                         uuids: List[str] = None, page: int = 1, limit: int = 100) -> List[Dict]:
        """
        입금 내역 조회

        Args:
            currency: 특정 화폐 (선택사항)
            state: 입금 상태 ('WAITING', 'PROCESSING', 'DONE', 'CANCELLED', 'REJECTED')
            uuids: 특정 입금 UUID 리스트 (선택사항)
            page: 페이지 번호
            limit: 조회 개수 (최대 100)

        Returns:
            입금 내역 리스트
        """
        try:
            params = {'page': page, 'limit': min(limit, 100)}

            if currency:
                params['currency'] = currency
            if state:
                params['state'] = state
            if uuids:
                params['uuids'] = uuids

            response = self._MakeRequest('GET', '/deposits', params, auth_required=True)
            return response
        except Exception as e:
            self.logger.error(f"입금 내역 조회 실패: {e}")
            return []

    def GetWithdrawHistory(self, currency: str = None, state: str = None,
                          uuids: List[str] = None, page: int = 1, limit: int = 100) -> List[Dict]:
        """
        출금 내역 조회

        Args:
            currency: 특정 화폐 (선택사항)
            state: 출금 상태 ('WAITING', 'PROCESSING', 'DONE', 'CANCELLED', 'REJECTED')
            uuids: 특정 출금 UUID 리스트 (선택사항)
            page: 페이지 번호
            limit: 조회 개수 (최대 100)

        Returns:
            출금 내역 리스트
        """
        try:
            params = {'page': page, 'limit': min(limit, 100)}

            if currency:
                params['currency'] = currency
            if state:
                params['state'] = state
            if uuids:
                params['uuids'] = uuids

            response = self._MakeRequest('GET', '/withdraws', params, auth_required=True)
            return response
        except Exception as e:
            self.logger.error(f"출금 내역 조회 실패: {e}")
            return []

    def GetWithdrawLimit(self, currency: str) -> Optional[Dict]:
        """
        출금 한도 조회

        Args:
            currency: 화폐 심볼 (예: 'BTC', 'ETH')

        Returns:
            출금 한도 정보
        """
        try:
            params = {'currency': currency}
            response = self._MakeRequest('GET', '/withdraws/limit', params, auth_required=True)
            return response
        except Exception as e:
            self.logger.error(f"출금 한도 조회 실패: {e}")
            return None

    # ==========================================================================
    # 지갑 연결 편의 메서드
    # ==========================================================================

    def SetupWallet(self, currency: str, net_type: str = None) -> Optional[str]:
        """
        지갑 연결 설정 (입금 주소 생성/조회)

        Args:
            currency: 화폐 심볼 (예: 'BTC', 'ETH')
            net_type: 네트워크 타입 (선택사항)

        Returns:
            입금 주소
        """
        try:
            # 기존 주소 확인
            address_info = self.GetCoinAddress(currency, net_type)

            if address_info and address_info.get('deposit_address'):
                address = address_info['deposit_address']
                self.logger.info(f"{currency} 기존 입금 주소 사용: {address}")
                return address
            else:
                # 새 주소 생성
                result = self.CreateCoinAddress(currency, net_type)
                if result:
                    self.logger.info(f"{currency} 새 입금 주소 생성 요청 완료")
                    return "주소 생성 중... 잠시 후 다시 조회해주세요."
                return None

        except Exception as e:
            self.logger.error(f"지갑 설정 실패: {e}")
            return None

    def TransferToExternalWallet(self, currency: str, amount: float, address: str,
                                memo: str = None) -> Optional[Dict]:
        """
        외부 지갑으로 코인 전송

        Args:
            currency: 화폐 심볼
            amount: 전송 수량
            address: 수신 주소
            memo: 메모 (필요시)

        Returns:
            전송 결과
        """
        try:
            # 출금 한도 확인
            limit_info = self.GetWithdrawLimit(currency)
            if limit_info:
                available = float(limit_info.get('limit_available', 0))
                if amount > available:
                    self.logger.error(f"출금 한도 초과: 요청 {amount}, 가능 {available}")
                    return None

            # 잔고 확인
            balance = self.GetCoinBalance(currency)
            if amount > balance:
                self.logger.error(f"잔고 부족: 요청 {amount}, 보유 {balance}")
                return None

            # 출금 실행
            return self.WithdrawCoin(currency, amount, address, memo)

        except Exception as e:
            self.logger.error(f"외부 지갑 전송 실패: {e}")
            return None

    def __del__(self):
        """소멸자 - 웹소켓 정리"""
        try:
            self.StopWebSocket()
        except:
            pass