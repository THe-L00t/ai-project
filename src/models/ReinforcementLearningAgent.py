#!/usr/bin/env python3
"""
강화학습 거래 에이전트
- PPO 알고리즘 기반 암호화폐 자동 거래
- 동적 포트폴리오 관리
- 리스크 관리와 수익 최적화
"""

import os
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

# Stable Baselines3 imports (will be installed via requirements.txt)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@dataclass
class ActionResult:
    """거래 액션 결과"""
    action: int
    action_name: str
    confidence: float
    expected_reward: float
    portfolio_change: Dict[str, float]
    reasoning: str


@dataclass
class AgentPerformance:
    """에이전트 성능 지표"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_profit_per_trade: float
    volatility: float


class TradingEnvironment(gym.Env):
    """
    암호화폐 거래 환경
    강화학습 에이전트가 학습할 수 있는 거래 시뮬레이션 환경
    """

    def __init__(self, market_data: pd.DataFrame, initial_balance: float = 100000,
                 transaction_cost: float = 0.001, max_position: float = 0.3):
        """
        환경 초기화

        Args:
            market_data: 시장 데이터
            initial_balance: 초기 자본
            transaction_cost: 거래 비용
            max_position: 최대 포지션 비율
        """
        super().__init__()

        self.market_data = market_data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        self.logger = logging.getLogger(__name__)

        # 상태 공간: [시장 피처, 포트폴리오 상태, 시간 피처]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(50,), dtype=np.float32  # 50차원 상태 공간
        )

        # 행동 공간: [HOLD, BUY_WEAK, BUY_STRONG, SELL_WEAK, SELL_STRONG]
        self.action_space = spaces.Discrete(5)

        # 환경 상태
        self.reset()

    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)

        self.current_step = 50  # 충분한 히스토리 확보
        self.balance = self.initial_balance
        self.position = 0.0  # 현재 코인 보유량
        self.position_value = 0.0
        self.total_value = self.initial_balance

        # 성과 추적
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        self.actions_taken = []

        # 리스크 관리
        self.max_value_seen = self.initial_balance
        self.current_drawdown = 0.0

        return self._get_observation(), {}

    def step(self, action):
        """한 스텝 실행"""
        if self.current_step >= len(self.market_data) - 1:
            return self._get_observation(), 0, True, True, {}

        # 현재 시장 상태
        current_price = self.market_data.iloc[self.current_step]['close_price']
        next_price = self.market_data.iloc[self.current_step + 1]['close_price']

        # 액션 실행
        reward = self._execute_action(action, current_price)

        # 다음 스텝으로 이동
        self.current_step += 1

        # 포트폴리오 가치 업데이트
        if self.current_step < len(self.market_data):
            new_price = self.market_data.iloc[self.current_step]['close_price']
            self.position_value = self.position * new_price
            self.total_value = self.balance + self.position_value

            self.portfolio_values.append(self.total_value)

            # 드로다운 계산
            if self.total_value > self.max_value_seen:
                self.max_value_seen = self.total_value
            self.current_drawdown = (self.max_value_seen - self.total_value) / self.max_value_seen

        # 종료 조건 체크
        done = (self.current_step >= len(self.market_data) - 1 or
                self.total_value <= self.initial_balance * 0.5)  # 50% 손실시 종료

        truncated = False

        return self._get_observation(), reward, done, truncated, {}

    def _execute_action(self, action: int, current_price: float) -> float:
        """액션 실행 및 보상 계산"""
        reward = 0.0
        action_executed = False

        # 액션 타입별 실행
        if action == 0:  # HOLD
            reward = 0.0

        elif action == 1:  # BUY_WEAK (현재 자본의 10%)
            buy_ratio = 0.1
            reward = self._execute_buy(buy_ratio, current_price)
            action_executed = True

        elif action == 2:  # BUY_STRONG (현재 자본의 30%)
            buy_ratio = 0.3
            reward = self._execute_buy(buy_ratio, current_price)
            action_executed = True

        elif action == 3:  # SELL_WEAK (현재 포지션의 30%)
            sell_ratio = 0.3
            reward = self._execute_sell(sell_ratio, current_price)
            action_executed = True

        elif action == 4:  # SELL_STRONG (현재 포지션의 70%)
            sell_ratio = 0.7
            reward = self._execute_sell(sell_ratio, current_price)
            action_executed = True

        # 액션 기록
        self.actions_taken.append(action)

        # 리스크 페널티 추가
        reward -= self.current_drawdown * 10  # 드로다운 페널티

        # 거래 비용 적용
        if action_executed:
            reward -= self.transaction_cost * 100  # 거래 비용 페널티

        return reward

    def _execute_buy(self, buy_ratio: float, price: float) -> float:
        """매수 실행"""
        if buy_ratio <= 0 or self.balance <= 0:
            return -1.0  # 잘못된 매수 시도에 대한 페널티

        # 실제 매수 가능한 금액 계산
        max_buy_amount = min(
            self.balance * buy_ratio,
            self.balance * (self.max_position - abs(self.position * price) / self.total_value)
        )

        if max_buy_amount <= 0:
            return -0.5  # 매수 불가능에 대한 작은 페널티

        # 거래 비용 고려
        actual_buy_amount = max_buy_amount * (1 - self.transaction_cost)
        coins_bought = actual_buy_amount / price

        # 포트폴리오 업데이트
        self.balance -= max_buy_amount
        self.position += coins_bought

        # 거래 기록
        self.trade_history.append({
            'step': self.current_step,
            'action': 'buy',
            'amount': max_buy_amount,
            'price': price,
            'coins': coins_bought
        })

        return 1.0  # 매수 성공 보상

    def _execute_sell(self, sell_ratio: float, price: float) -> float:
        """매도 실행"""
        if sell_ratio <= 0 or self.position <= 0:
            return -1.0  # 잘못된 매도 시도에 대한 페널티

        # 매도할 코인 수량
        coins_to_sell = min(self.position * sell_ratio, self.position)
        sell_amount = coins_to_sell * price * (1 - self.transaction_cost)

        # 포트폴리오 업데이트
        self.position -= coins_to_sell
        self.balance += sell_amount

        # 거래 기록
        self.trade_history.append({
            'step': self.current_step,
            'action': 'sell',
            'amount': sell_amount,
            'price': price,
            'coins': coins_to_sell
        })

        return 1.0  # 매도 성공 보상

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관측값 생성"""
        if self.current_step >= len(self.market_data):
            return np.zeros(50, dtype=np.float32)

        try:
            current_data = self.market_data.iloc[self.current_step]

            # 시장 데이터 피처 (30차원)
            market_features = []
            market_columns = ['sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14',
                            'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
                            'atr_14', 'volume_sma_20', 'close_price', 'volume']

            for col in market_columns:
                if col in current_data:
                    value = current_data[col]
                    market_features.append(float(value) if not pd.isna(value) else 0.0)
                else:
                    market_features.append(0.0)

            # 가격 변화율 추가 (5일, 10일, 20일)
            if self.current_step >= 20:
                for days in [5, 10, 20]:
                    if self.current_step >= days:
                        past_price = self.market_data.iloc[self.current_step - days]['close_price']
                        current_price = current_data['close_price']
                        change_rate = (current_price - past_price) / past_price if past_price > 0 else 0.0
                        market_features.append(change_rate)
                    else:
                        market_features.append(0.0)

            # 부족한 차원 채우기
            while len(market_features) < 30:
                market_features.append(0.0)
            market_features = market_features[:30]

            # 포트폴리오 상태 피처 (15차원)
            current_price = current_data['close_price']
            portfolio_features = [
                self.balance / self.initial_balance,  # 현금 비율
                (self.position * current_price) / self.initial_balance,  # 포지션 비율
                self.total_value / self.initial_balance,  # 총 자산 비율
                self.current_drawdown,  # 현재 드로다운
                len(self.trade_history) / 100,  # 거래 횟수 (정규화)
                min(self.position / (self.initial_balance / current_price), 1.0),  # 포지션 크기
                1.0 if self.position > 0 else 0.0,  # 포지션 보유 여부
                self.balance / self.total_value if self.total_value > 0 else 1.0,  # 현금 비중
                (self.position * current_price) / self.total_value if self.total_value > 0 else 0.0,  # 코인 비중
            ]

            # 최근 5개 액션 원핫 인코딩
            recent_actions = self.actions_taken[-5:] if len(self.actions_taken) >= 5 else [0] * 5
            for action in recent_actions:
                action_onehot = [0.0] * 5
                if 0 <= action < 5:
                    action_onehot[action] = 1.0
                portfolio_features.extend(action_onehot)

            # 부족한 차원 채우기
            while len(portfolio_features) < 15:
                portfolio_features.append(0.0)
            portfolio_features = portfolio_features[:15]

            # 시간 피처 (5차원)
            time_features = [
                self.current_step / len(self.market_data),  # 진행 비율
                (self.current_step % 24) / 24,  # 시간 (일주기)
                (self.current_step % 168) / 168,  # 요일 (주주기)
                np.sin(2 * np.pi * self.current_step / 24),  # 일주기 sin
                np.cos(2 * np.pi * self.current_step / 24),  # 일주기 cos
            ]

            # 전체 관측값 결합
            observation = np.array(market_features + portfolio_features + time_features, dtype=np.float32)

            # NaN 값 처리
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

            # 정규화 ([-1, 1] 범위로)
            observation = np.clip(observation, -10, 10)

            return observation

        except Exception as e:
            self.logger.error(f"관측값 생성 오류: {e}")
            return np.zeros(50, dtype=np.float32)

    def get_portfolio_metrics(self) -> Dict:
        """포트폴리오 성과 지표 계산"""
        if len(self.portfolio_values) < 2:
            return {}

        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        total_return = (self.total_value - self.initial_balance) / self.initial_balance

        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            volatility = 0

        max_drawdown = max([
            (max(values[:i+1]) - values[i]) / max(values[:i+1])
            for i in range(1, len(values))
        ]) if len(values) > 1 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'total_trades': len(self.trade_history),
            'final_value': self.total_value
        }


class ReinforcementLearningAgent:
    """
    강화학습 기반 거래 에이전트
    PPO 알고리즘을 사용하여 최적의 거래 전략 학습
    """

    def __init__(self, config: Dict = None):
        """
        초기화

        Args:
            config: 에이전트 설정
        """
        self.logger = logging.getLogger(__name__)

        if not SB3_AVAILABLE:
            self.logger.error("Stable Baselines3가 설치되지 않았습니다. pip install stable-baselines3")
            raise ImportError("Stable Baselines3 required")

        # 기본 설정
        default_config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'total_timesteps': 100000,
            'eval_freq': 5000
        }

        self.config = {**default_config, **(config or {})}

        # 모델과 환경
        self.model = None
        self.env = None
        self.is_trained = False

        # 성능 추적
        self.training_history = []
        self.evaluation_results = []

        # 모델 저장 경로
        self.model_dir = "models/"
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger.info("강화학습 에이전트 초기화 완료")

    def setup_environment(self, market_data: pd.DataFrame, **env_kwargs):
        """거래 환경 설정"""
        try:
            self.env = TradingEnvironment(market_data, **env_kwargs)
            self.logger.info("거래 환경 설정 완료")
        except Exception as e:
            self.logger.error(f"환경 설정 오류: {e}")

    def train(self, market_data: pd.DataFrame, **env_kwargs) -> bool:
        """
        모델 학습

        Args:
            market_data: 학습용 시장 데이터
            **env_kwargs: 환경 설정 인수

        Returns:
            학습 성공 여부
        """
        try:
            self.logger.info("강화학습 모델 학습 시작")

            # 환경 설정
            if self.env is None:
                self.setup_environment(market_data, **env_kwargs)

            # 벡터화된 환경 생성
            vec_env = DummyVecEnv([lambda: self.env])

            # PPO 모델 초기화
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=self.config['learning_rate'],
                gamma=self.config['gamma'],
                clip_range=self.config['clip_range'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                verbose=1
            )

            # 학습 진행
            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=self._create_training_callback()
            )

            self.is_trained = True
            self.logger.info("강화학습 모델 학습 완료")

            # 모델 저장
            self.save_model()

            return True

        except Exception as e:
            self.logger.error(f"모델 학습 오류: {e}")
            return False

    def _create_training_callback(self):
        """학습 중 성능 모니터링 콜백"""
        class TrainingCallback(BaseCallback):
            def __init__(self, parent_agent):
                super().__init__()
                self.parent_agent = parent_agent
                self.eval_count = 0

            def _on_step(self) -> bool:
                # 주기적 성능 평가
                if self.num_timesteps % self.parent_agent.config['eval_freq'] == 0:
                    self.eval_count += 1

                    # 현재 환경 성과 지표 수집
                    if hasattr(self.training_env.envs[0], 'get_portfolio_metrics'):
                        metrics = self.training_env.envs[0].get_portfolio_metrics()
                        metrics['timestep'] = self.num_timesteps
                        self.parent_agent.training_history.append(metrics)

                return True

        return TrainingCallback(self)

    def predict(self, market_data: pd.DataFrame, deterministic: bool = True) -> ActionResult:
        """
        예측 수행

        Args:
            market_data: 현재 시장 데이터
            deterministic: 결정적 예측 여부

        Returns:
            액션 결과
        """
        try:
            if not self.is_trained or self.model is None:
                self.logger.error("모델이 학습되지 않았습니다")
                return ActionResult(
                    action=0, action_name="HOLD", confidence=0.0,
                    expected_reward=0.0, portfolio_change={},
                    reasoning="모델 미학습"
                )

            # 임시 환경 생성 (예측용)
            temp_env = TradingEnvironment(market_data.tail(100))  # 최근 100개 데이터 사용
            obs, _ = temp_env.reset()

            # 예측 실행
            action, _states = self.model.predict(obs, deterministic=deterministic)
            action = int(action)

            # 액션 이름 매핑
            action_names = ['HOLD', 'BUY_WEAK', 'BUY_STRONG', 'SELL_WEAK', 'SELL_STRONG']
            action_name = action_names[action] if 0 <= action < 5 else 'UNKNOWN'

            # 신뢰도 계산 (정책의 확률 분포 기반)
            action_probs = self.model.policy.get_distribution(obs).distribution.probs.detach().numpy()
            confidence = float(action_probs[action])

            # 예상 보상 (간단한 추정)
            expected_reward = self._estimate_reward(action, market_data.tail(10))

            # 추론 근거 생성
            reasoning = self._generate_reasoning(action, market_data.tail(5))

            return ActionResult(
                action=action,
                action_name=action_name,
                confidence=confidence,
                expected_reward=expected_reward,
                portfolio_change={},  # 실제 실행 시 계산
                reasoning=reasoning
            )

        except Exception as e:
            self.logger.error(f"예측 오류: {e}")
            return ActionResult(
                action=0, action_name="HOLD", confidence=0.0,
                expected_reward=0.0, portfolio_change={},
                reasoning=f"예측 오류: {str(e)}"
            )

    def _estimate_reward(self, action: int, recent_data: pd.DataFrame) -> float:
        """액션의 예상 보상 추정"""
        try:
            if len(recent_data) < 5:
                return 0.0

            # 최근 가격 트렌드 분석
            prices = recent_data['close_price'].values
            price_change = (prices[-1] - prices[0]) / prices[0]

            # 액션별 예상 보상
            if action == 0:  # HOLD
                return 0.0
            elif action in [1, 2]:  # BUY
                return price_change * 10  # 상승시 긍정적 보상
            elif action in [3, 4]:  # SELL
                return -price_change * 10  # 하락시 긍정적 보상

            return 0.0

        except Exception:
            return 0.0

    def _generate_reasoning(self, action: int, recent_data: pd.DataFrame) -> str:
        """액션 선택 근거 생성"""
        try:
            action_names = ['보유', '약매수', '강매수', '약매도', '강매도']
            action_name = action_names[action] if 0 <= action < 5 else '알수없음'

            reasoning_parts = [f"액션: {action_name}"]

            if len(recent_data) > 0:
                latest = recent_data.iloc[-1]

                # RSI 분석
                if 'rsi_14' in latest:
                    rsi = latest['rsi_14']
                    if rsi > 70:
                        reasoning_parts.append("RSI과매수")
                    elif rsi < 30:
                        reasoning_parts.append("RSI과매도")

                # MACD 분석
                if 'macd' in latest and 'macd_signal' in latest:
                    if latest['macd'] > latest['macd_signal']:
                        reasoning_parts.append("MACD상승신호")
                    else:
                        reasoning_parts.append("MACD하락신호")

            return " | ".join(reasoning_parts)

        except Exception:
            return "분석불가"

    def evaluate(self, test_data: pd.DataFrame, **env_kwargs) -> AgentPerformance:
        """
        모델 성능 평가

        Args:
            test_data: 테스트 데이터
            **env_kwargs: 환경 설정

        Returns:
            성능 지표
        """
        try:
            if not self.is_trained:
                self.logger.error("모델이 학습되지 않았습니다")
                return AgentPerformance(0, 0, 0, 0, 0, 0, 0, 0)

            # 테스트 환경 생성
            test_env = TradingEnvironment(test_data, **env_kwargs)
            obs, _ = test_env.reset()
            done = False

            total_reward = 0
            step_count = 0

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                total_reward += reward
                step_count += 1

                if truncated:
                    break

            # 성능 지표 계산
            metrics = test_env.get_portfolio_metrics()

            # 수익성 거래 분석
            profitable_trades = sum(1 for trade in test_env.trade_history
                                  if self._is_profitable_trade(trade, test_env))

            return AgentPerformance(
                total_return=metrics.get('total_return', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                win_rate=profitable_trades / max(len(test_env.trade_history), 1),
                total_trades=metrics.get('total_trades', 0),
                profitable_trades=profitable_trades,
                avg_profit_per_trade=total_reward / max(step_count, 1),
                volatility=metrics.get('volatility', 0)
            )

        except Exception as e:
            self.logger.error(f"성능 평가 오류: {e}")
            return AgentPerformance(0, 0, 0, 0, 0, 0, 0, 0)

    def _is_profitable_trade(self, trade: Dict, env: TradingEnvironment) -> bool:
        """거래의 수익성 판단"""
        # 간단한 구현: 매수 후 가격 상승, 매도 후 가격 하락을 수익으로 간주
        try:
            if trade['step'] + 1 < len(env.market_data):
                current_price = trade['price']
                future_price = env.market_data.iloc[trade['step'] + 1]['close_price']

                if trade['action'] == 'buy':
                    return future_price > current_price
                else:  # sell
                    return future_price < current_price

            return False
        except:
            return False

    def save_model(self, model_name: str = None):
        """모델 저장"""
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("저장할 학습된 모델이 없습니다")
                return

            if not model_name:
                model_name = f"rl_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            model_path = os.path.join(self.model_dir, f"{model_name}.zip")
            self.model.save(model_path)

            # 메타데이터 저장
            metadata = {
                'config': self.config,
                'training_history': self.training_history,
                'evaluation_results': self.evaluation_results,
                'is_trained': self.is_trained,
                'model_name': model_name,
                'saved_at': datetime.now().isoformat()
            }

            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"모델 저장 완료: {model_path}")

        except Exception as e:
            self.logger.error(f"모델 저장 오류: {e}")

    def load_model(self, model_path: str) -> bool:
        """모델 불러오기"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
                return False

            self.model = PPO.load(model_path)
            self.is_trained = True

            # 메타데이터 불러오기
            metadata_path = model_path.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.config = metadata.get('config', self.config)
                    self.training_history = metadata.get('training_history', [])
                    self.evaluation_results = metadata.get('evaluation_results', [])

            self.logger.info(f"모델 로드 완료: {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"모델 로드 오류: {e}")
            return False

    def get_training_progress(self) -> List[Dict]:
        """학습 진행 상황 반환"""
        return self.training_history.copy()

    def get_agent_info(self) -> Dict:
        """에이전트 정보 반환"""
        return {
            'is_trained': self.is_trained,
            'config': self.config,
            'training_steps': len(self.training_history),
            'evaluation_count': len(self.evaluation_results),
            'model_type': 'PPO'
        }