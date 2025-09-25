"""
알림 시스템
텔레그램, 이메일, 웹 알림을 통해 중요한 이벤트를 전송합니다.
"""

import asyncio
import logging
import smtplib
import requests
from datetime import datetime
from typing import Dict, List, Optional
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Python 3.13 호환성을 위한 대체 import
    MimeText = None
    MimeMultipart = None


class NotificationSystem:
    """알림 시스템"""

    def __init__(self, config: Dict = None):
        """
        초기화

        Args:
            config: 알림 설정
                - telegram: {bot_token, chat_id}
                - email: {smtp_server, smtp_port, username, password, to_emails}
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 텔레그램 설정
        self.telegram_bot_token = self.config.get('telegram', {}).get('bot_token')
        self.telegram_chat_id = self.config.get('telegram', {}).get('chat_id')

        # 이메일 설정
        email_config = self.config.get('email', {})
        self.smtp_server = email_config.get('smtp_server')
        self.smtp_port = email_config.get('smtp_port', 587)
        self.email_username = email_config.get('username')
        self.email_password = email_config.get('password')
        self.to_emails = email_config.get('to_emails', [])

        # 알림 레벨
        self.notification_levels = {
            'INFO': '📘',
            'WARNING': '⚠️',
            'ERROR': '🚨',
            'SUCCESS': '✅',
            'TRADE': '💰',
            'RISK': '🛡️'
        }

    async def send_notification(self, message: str, level: str = 'INFO',
                              channels: List[str] = None):
        """알림 전송"""
        if channels is None:
            channels = ['telegram', 'email']

        emoji = self.notification_levels.get(level, '📌')
        formatted_message = f"{emoji} {message}"

        tasks = []

        if 'telegram' in channels and self.telegram_bot_token:
            tasks.append(self._send_telegram(formatted_message))

        if 'email' in channels and self.smtp_server:
            tasks.append(self._send_email(f"CoinTradingAI Alert", formatted_message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_trade_notification(self, trade_data: Dict):
        """거래 알림"""
        symbol = trade_data.get('symbol', 'Unknown')
        action = trade_data.get('action', 'Unknown')
        price = trade_data.get('price', 0)
        quantity = trade_data.get('quantity', 0)
        profit_rate = trade_data.get('profit_rate', 0)

        if action == 'BUY':
            message = f"""
🔥 매수 실행
코인: {symbol}
가격: {price:,.0f} 원
수량: {quantity:.8f}
시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        elif action == 'SELL':
            profit_emoji = '📈' if profit_rate > 0 else '📉'
            message = f"""
{profit_emoji} 매도 실행
코인: {symbol}
가격: {price:,.0f} 원
수량: {quantity:.8f}
수익률: {profit_rate:.2f}%
시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            message = f"거래 실행: {action} - {symbol}"

        await self.send_notification(message, 'TRADE')

    async def send_risk_alert(self, risk_data: Dict):
        """리스크 알림"""
        risk_level = risk_data.get('level', 'UNKNOWN')
        risk_factors = risk_data.get('factors', [])
        recommendations = risk_data.get('recommendations', [])

        message = f"""
🛡️ 리스크 경고
레벨: {risk_level}
요인: {', '.join(risk_factors)}
권장사항: {', '.join(recommendations)}
시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        level = 'ERROR' if risk_level == 'CRITICAL' else 'WARNING'
        await self.send_notification(message, level)

    async def send_performance_summary(self, summary_data: Dict):
        """성과 요약 알림 (일일)"""
        total_balance = summary_data.get('total_balance', 0)
        daily_pnl = summary_data.get('daily_pnl', 0)
        daily_return = summary_data.get('daily_return', 0)
        total_trades = summary_data.get('total_trades', 0)
        win_rate = summary_data.get('win_rate', 0)

        return_emoji = '📈' if daily_return > 0 else '📉' if daily_return < 0 else '➡️'

        message = f"""
{return_emoji} 일일 성과 요약
총 자산: {total_balance:,.0f} 원
일일 손익: {daily_pnl:,.0f} 원
일일 수익률: {daily_return:.2f}%
총 거래 수: {total_trades}
승률: {win_rate:.1f}%
날짜: {datetime.now().strftime('%Y-%m-%d')}
"""

        level = 'SUCCESS' if daily_return > 0 else 'WARNING' if daily_return < -5 else 'INFO'
        await self.send_notification(message, level)

    async def send_system_alert(self, alert_type: str, message: str):
        """시스템 알림"""
        system_messages = {
            'STARTUP': '🚀 시스템 시작됨',
            'SHUTDOWN': '🛑 시스템 종료됨',
            'ERROR': '❌ 시스템 오류',
            'EMERGENCY_STOP': '🚨 비상 정지 실행',
            'CONNECTION_LOST': '📡 연결 끊김',
            'CONNECTION_RESTORED': '✅ 연결 복구'
        }

        system_emoji = system_messages.get(alert_type, '📌')
        full_message = f"{system_emoji}\n{message}\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        level = 'ERROR' if alert_type in ['ERROR', 'EMERGENCY_STOP'] else 'INFO'
        await self.send_notification(full_message, level)

    async def _send_telegram(self, message: str):
        """텔레그램 메시지 전송"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                return False

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            self.logger.info("텔레그램 알림 전송 성공")
            return True

        except Exception as e:
            self.logger.error(f"텔레그램 알림 전송 실패: {e}")
            return False

    async def _send_email(self, subject: str, message: str):
        """이메일 전송"""
        try:
            if not self.smtp_server or not self.to_emails:
                return False

            msg = MimeMultipart()
            msg['From'] = self.email_username
            msg['Subject'] = subject

            # HTML 형식으로 메시지 포맷
            html_message = message.replace('\n', '<br>')
            msg.attach(MimeText(html_message, 'html'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.email_username and self.email_password:
                    server.login(self.email_username, self.email_password)

                for to_email in self.to_emails:
                    msg['To'] = to_email
                    text = msg.as_string()
                    server.sendmail(self.email_username, to_email, text)
                    del msg['To']

            self.logger.info("이메일 알림 전송 성공")
            return True

        except Exception as e:
            self.logger.error(f"이메일 알림 전송 실패: {e}")
            return False

    def test_notifications(self):
        """알림 테스트"""
        test_message = "알림 시스템 테스트 메시지입니다."
        asyncio.run(self.send_notification(test_message, 'INFO'))


# 설정 예시
NOTIFICATION_CONFIG_EXAMPLE = {
    'telegram': {
        'bot_token': 'YOUR_BOT_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    },
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',  # Gmail App Password
        'to_emails': ['recipient@gmail.com']
    }
}


if __name__ == '__main__':
    # 테스트 실행
    notification_system = NotificationSystem(NOTIFICATION_CONFIG_EXAMPLE)
    notification_system.test_notifications()