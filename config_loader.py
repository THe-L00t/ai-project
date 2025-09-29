#!/usr/bin/env python3
"""
AI 설정 파일 로더
AI_SETTINGS.md 파일에서 설정을 읽어와 적용합니다.
"""

import os
import re
import yaml
import logging

logger = logging.getLogger(__name__)

class AIConfigLoader:
    """AI 설정 파일 로더"""

    def __init__(self, config_file='AI_SETTINGS.md'):
        self.config_file = config_file
        self.settings = {}
        self.load_settings()

    def load_settings(self):
        """설정 파일 로드"""
        try:
            # 1. AI_SETTINGS.md 로드
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.parse_settings(content)
            else:
                logger.warning(f"설정 파일 없음: {self.config_file}")

            # 2. trading_config.yaml 로드 (새 설정 추가)
            yaml_config_path = 'config/trading_config.yaml'
            if os.path.exists(yaml_config_path):
                with open(yaml_config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                self.parse_yaml_settings(yaml_config)
                logger.info("✅ trading_config.yaml 설정 추가 로드")

            if not self.settings:
                logger.warning("설정이 비어있음, 기본값 사용")
                self.load_defaults()
                return

            self.validate_settings()
            logger.info(f"✅ 설정 파일 로드 완료: {len(self.settings)}개 설정")

        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}, 기본값 사용")
            self.load_defaults()

    def parse_settings(self, content):
        """설정 파싱"""
        # 패턴: KEY = VALUE 형태 찾기
        pattern = r'^([A-Z_]+)\s*=\s*(.+)$'

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue

            # 코드 블록 내의 설정 찾기
            if '```' in line:
                continue

            match = re.match(pattern, line)
            if match:
                key, value = match.groups()
                self.settings[key] = self.convert_value(value.strip())

    def convert_value(self, value):
        """값 타입 변환"""
        # 문자열 정리
        value = value.strip('`"\'')

        # 불린 값
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False

        # 숫자 값
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # 리스트 값 (쉼표로 구분)
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        # 문자열 값
        return value

    def parse_yaml_settings(self, yaml_config):
        """YAML 설정 파싱"""
        try:
            # ai_integration 섹션에서 설정 추출
            ai_integration = yaml_config.get('ai_integration', {})

            if 'trading_cycle_seconds' in ai_integration:
                self.settings['TRADING_CYCLE_SECONDS'] = ai_integration['trading_cycle_seconds']

            if 'api_cache_ttl' in ai_integration:
                self.settings['API_CACHE_TTL'] = ai_integration['api_cache_ttl']

            logger.info(f"YAML에서 추가된 설정: TRADING_CYCLE_SECONDS={ai_integration.get('trading_cycle_seconds')}, API_CACHE_TTL={ai_integration.get('api_cache_ttl')}")

        except Exception as e:
            logger.error(f"YAML 설정 파싱 실패: {e}")

    def validate_settings(self):
        """설정값 검증 (사용자 설정값을 그대로 사용)"""
        # 사용자가 설정한 값을 그대로 사용 - 검증 제거
        logger.info(f"✅ 사용자 설정값 그대로 적용: {len(self.settings)}개 설정")

        # 설정값 출력
        for key, value in self.settings.items():
            if key in ['MAX_POSITION_SIZE', 'STOP_LOSS_PERCENTAGE', 'TAKE_PROFIT_PERCENTAGE',
                      'BUY_THRESHOLD_CHANGE', 'SELL_THRESHOLD_CHANGE']:
                logger.info(f"   {key}: {value}")

        # validation 완전 제거 - 사용자 설정값 우선

    def load_defaults(self):
        """기본 설정값 로드"""
        self.settings = {
            # 기본 매매 설정
            'MAX_POSITION_SIZE': 0.15,
            'STOP_LOSS_PERCENTAGE': 3.0,
            'TAKE_PROFIT_PERCENTAGE': 15.0,

            # 반응 민감도
            'BUY_THRESHOLD_CHANGE': 1.0,
            'SELL_THRESHOLD_CHANGE': -1.0,
            'MOMENTUM_THRESHOLD': 0.5,

            # 사이클 설정
            'TRADING_CYCLE_SECONDS': 20,
            'NEWS_COLLECTION_INTERVAL': 5,
            'MODEL_RETRAIN_INTERVAL': 10,

            # 대상 코인
            'TARGET_COINS': ['KRW-BTC', 'KRW-ETH', 'KRW-ADA', 'KRW-DOT'],
            'MAX_NEWS_ARTICLES': 100,

            # AI 설정
            'ENABLE_ADAPTIVE_LEARNING': True,
            'ENABLE_NEWS_SENTIMENT': True,
            'ENABLE_PATTERN_LEARNING': True,
            'MIN_CONFIDENCE_THRESHOLD': 0.3,

            # 신호 가중치
            'AGGRESSIVE_PATTERN_WEIGHT': 0.7,
            'NEWS_SENTIMENT_WEIGHT': 0.8,
            'ADAPTIVE_LEARNING_WEIGHT': 0.6,
            'PATTERN_MODEL_WEIGHT': 0.6,

            # 리스크 관리
            'MAX_CONCURRENT_POSITIONS': 2,
            'MAX_DAILY_TRADES': 20,
            'DAILY_LOSS_LIMIT': 5.0,

            # 프리셋
            'PRESET': 'AGGRESSIVE'
        }

    def apply_preset(self):
        """프리셋 적용 - 사용자 설정값 우선"""
        # 프리셋 무시하고 사용자 설정값만 사용
        preset = self.settings.get('PRESET', 'USER_CUSTOM')
        logger.info(f"📋 사용자 맞춤 설정 적용 (프리셋: {preset})")
        logger.info("✅ AI_SETTINGS.md 파일의 설정값을 그대로 사용합니다")

    def get(self, key, default=None):
        """설정값 가져오기"""
        return self.settings.get(key, default)

    def get_all(self):
        """모든 설정값 반환"""
        return self.settings.copy()

    def save_current_settings(self):
        """현재 설정을 파일에 저장"""
        try:
            # 현재 설정을 바탕으로 AI_SETTINGS.md 업데이트
            content = self.generate_settings_content()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("💾 현재 설정 저장 완료")
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")

    def generate_settings_content(self):
        """설정 파일 내용 생성"""
        # 기본 템플릿 유지하면서 값만 업데이트
        # 실제 구현에서는 템플릿을 읽고 값만 교체
        return f"""# 🚀 하이브리드 마스터 AI 설정 파일

## 현재 적용된 설정값

### 기본 매매 설정
```
MAX_POSITION_SIZE = {self.settings['MAX_POSITION_SIZE']}
STOP_LOSS_PERCENTAGE = {self.settings['STOP_LOSS_PERCENTAGE']}
TAKE_PROFIT_PERCENTAGE = {self.settings['TAKE_PROFIT_PERCENTAGE']}
```

### 반응 민감도
```
BUY_THRESHOLD_CHANGE = {self.settings['BUY_THRESHOLD_CHANGE']}
SELL_THRESHOLD_CHANGE = {self.settings['SELL_THRESHOLD_CHANGE']}
MOMENTUM_THRESHOLD = {self.settings['MOMENTUM_THRESHOLD']}
```

[... 나머지 설정들 ...]

*자동 생성된 설정 파일 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

# 싱글톤 인스턴스
_config_instance = None

def get_config():
    """설정 인스턴스 가져오기"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AIConfigLoader()
        _config_instance.apply_preset()
    return _config_instance

def reload_config():
    """설정 다시 로드"""
    global _config_instance
    _config_instance = None
    return get_config()

if __name__ == "__main__":
    # 테스트
    config = get_config()
    print("로드된 설정:")
    for key, value in config.get_all().items():
        print(f"  {key}: {value}")