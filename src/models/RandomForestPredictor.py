#!/usr/bin/env python3
"""
Random Forest 예측 모델
- 시장 데이터 기반 가격 예측
- 기술적 지표를 활용한 머신러닝 모델
- 안정적이고 해석 가능한 예측 결과 제공
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib


@dataclass
class PredictionResult:
    """예측 결과"""
    market: str
    timestamp: datetime
    prediction_type: str  # 'price', 'direction', 'signal'
    predicted_value: float
    confidence: float
    probability_distribution: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None


@dataclass
class ModelPerformance:
    """모델 성능 지표"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None


class FeatureEngineer:
    """
    피처 엔지니어링 클래스
    시장 데이터를 머신러닝에 적합한 형태로 변환
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def CreateFeatures(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        시장 데이터로부터 피처 생성

        Args:
            market_data: 시장 데이터 DataFrame

        Returns:
            피처가 추가된 DataFrame
        """
        try:
            df = market_data.copy()

            # 가격 변화율 기반 피처
            df['price_change_1'] = df['close_price'].pct_change(1)
            df['price_change_3'] = df['close_price'].pct_change(3)
            df['price_change_5'] = df['close_price'].pct_change(5)

            # 변동성 지표
            df['price_volatility_5'] = df['close_price'].rolling(5).std()
            df['price_volatility_20'] = df['close_price'].rolling(20).std()

            # 거래량 기반 피처
            df['volume_change_1'] = df['volume'].pct_change(1)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

            # RSI 기반 피처
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_normalized'] = (df['rsi_14'] - 50) / 50

            # MACD 기반 피처
            df['macd_signal_positive'] = (df['macd'] > df['macd_signal']).astype(int)
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # 볼린저 밴드 기반 피처
            df['bb_position'] = (df['close_price'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
            df['bb_squeeze'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['close_price']

            # 이동평균선 관련 피처
            df['sma_5_20_ratio'] = df['sma_5'] / df['sma_20']
            df['price_sma_5_ratio'] = df['close_price'] / df['sma_5']
            df['price_sma_20_ratio'] = df['close_price'] / df['sma_20']
            df['price_sma_50_ratio'] = df['close_price'] / df['sma_50']

            # 시간 기반 피처
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['korean_trading_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)

            # NaN 값 처리
            df = df.fillna(method='forward').fillna(method='backward')

            return df

        except Exception as e:
            self.logger.error(f"피처 생성 오류: {e}")
            return market_data

    def SelectFeatures(self, df: pd.DataFrame) -> List[str]:
        """
        학습에 사용할 피처 선택

        Args:
            df: 피처가 포함된 DataFrame

        Returns:
            선택된 피처명 리스트
        """
        # 기본 기술적 지표
        technical_features = [
            'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'atr_14',
            'bollinger_upper', 'bollinger_lower', 'volume_sma_20'
        ]

        # 생성된 피처
        engineered_features = [
            'price_change_1', 'price_change_3', 'price_change_5',
            'price_volatility_5', 'price_volatility_20',
            'volume_change_1', 'volume_ratio',
            'rsi_oversold', 'rsi_overbought', 'rsi_normalized',
            'macd_signal_positive', 'macd_histogram',
            'bb_position', 'bb_squeeze',
            'sma_5_20_ratio', 'price_sma_5_ratio', 'price_sma_20_ratio', 'price_sma_50_ratio',
            'hour', 'day_of_week', 'is_weekend', 'korean_trading_hours'
        ]

        # 실제 존재하는 피처만 선택
        available_features = []
        for feature in technical_features + engineered_features:
            if feature in df.columns:
                available_features.append(feature)

        return available_features

    def PrepareData(self, df: pd.DataFrame, target_column: str = 'close_price',
                   prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 준비

        Args:
            df: 입력 데이터
            target_column: 타겟 컬럼명
            prediction_horizon: 예측 시점 (몇 스텝 후 예측할지)

        Returns:
            X, y 데이터
        """
        try:
            # 피처 선택
            features = self.SelectFeatures(df)

            # 피처 데이터 준비
            X = df[features].values

            # 타겟 데이터 준비 (미래 가격)
            if target_column in df.columns:
                y = df[target_column].shift(-prediction_horizon).values
            else:
                # 가격 변화 방향 예측 (상승/하락)
                price_change = df['close_price'].pct_change(prediction_horizon).shift(-prediction_horizon)
                y = (price_change > 0).astype(int).values

            # NaN 값 제거
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]

            return X, y

        except Exception as e:
            self.logger.error(f"데이터 준비 오류: {e}")
            return np.array([]), np.array([])


class RandomForestPredictor:
    """
    Random Forest 기반 암호화폐 가격 예측 모델
    안정적이고 해석 가능한 예측 결과 제공
    """

    def __init__(self, model_config: Dict = None):
        """
        초기화

        Args:
            model_config: 모델 설정 딕셔너리
        """
        self.logger = logging.getLogger(__name__)

        # 기본 설정
        default_config = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }

        self.config = {**default_config, **(model_config or {})}

        # 모델 초기화
        self.price_model = RandomForestRegressor(**self.config)
        self.direction_model = RandomForestClassifier(**self.config)

        # 피처 엔지니어
        self.feature_engineer = FeatureEngineer()

        # 모델 상태
        self.is_trained = False
        self.feature_names = []
        self.last_training_data_size = 0

        # 성능 지표
        self.performance_metrics = {}

        # 모델 저장 경로
        self.model_dir = "models/"
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger.info("Random Forest 예측 모델 초기화 완료")

    def Train(self, market_data: pd.DataFrame, retrain: bool = False) -> bool:
        """
        모델 학습

        Args:
            market_data: 학습 데이터
            retrain: 재학습 여부

        Returns:
            학습 성공 여부
        """
        try:
            self.logger.info("Random Forest 모델 학습 시작")

            # 데이터 크기 확인
            if len(market_data) < 100:
                self.logger.warning("학습 데이터가 부족합니다 (최소 100개 필요)")
                return False

            # 재학습 필요성 판단
            if self.is_trained and not retrain and len(market_data) <= self.last_training_data_size * 1.1:
                self.logger.info("재학습이 필요하지 않습니다")
                return True

            # 피처 생성
            df = self.feature_engineer.CreateFeatures(market_data)

            # 가격 예측용 데이터 준비
            X_price, y_price = self.feature_engineer.PrepareData(df, 'close_price', 1)

            # 방향 예측용 데이터 준비
            X_direction, y_direction = self.feature_engineer.PrepareData(df, 'direction', 1)

            if len(X_price) == 0 or len(X_direction) == 0:
                self.logger.error("학습 데이터 준비 실패")
                return False

            # 피처명 저장
            self.feature_names = self.feature_engineer.SelectFeatures(df)

            # 시계열 교차 검증
            tscv = TimeSeriesSplit(n_splits=5)

            # 가격 예측 모델 학습
            self.logger.info("가격 예측 모델 학습 중...")
            self.price_model.fit(X_price, y_price)

            # 가격 모델 성능 평가
            price_scores = cross_val_score(self.price_model, X_price, y_price, cv=tscv, scoring='neg_mean_squared_error')
            price_r2_scores = cross_val_score(self.price_model, X_price, y_price, cv=tscv, scoring='r2')

            # 방향 예측 모델 학습
            self.logger.info("방향 예측 모델 학습 중...")
            self.direction_model.fit(X_direction, y_direction)

            # 방향 모델 성능 평가
            direction_scores = cross_val_score(self.direction_model, X_direction, y_direction, cv=tscv, scoring='accuracy')

            # 성능 지표 계산
            self.performance_metrics = {
                'price_model': {
                    'mse': -price_scores.mean(),
                    'rmse': np.sqrt(-price_scores.mean()),
                    'r2_score': price_r2_scores.mean(),
                    'cross_val_std': price_scores.std()
                },
                'direction_model': {
                    'accuracy': direction_scores.mean(),
                    'cross_val_std': direction_scores.std()
                },
                'feature_importance': dict(zip(
                    self.feature_names,
                    self.price_model.feature_importances_
                )),
                'training_samples': len(X_price),
                'training_date': datetime.now().isoformat()
            }

            self.is_trained = True
            self.last_training_data_size = len(market_data)

            # 모델 저장
            self.SaveModel()

            self.logger.info(f"모델 학습 완료 - 가격 예측 R²: {price_r2_scores.mean():.3f}, 방향 예측 정확도: {direction_scores.mean():.3f}")
            return True

        except Exception as e:
            self.logger.error(f"모델 학습 오류: {e}")
            return False

    def Predict(self, current_data: pd.DataFrame, prediction_type: str = 'both') -> List[PredictionResult]:
        """
        예측 수행

        Args:
            current_data: 현재 시장 데이터
            prediction_type: 예측 타입 ('price', 'direction', 'both')

        Returns:
            예측 결과 리스트
        """
        try:
            if not self.is_trained:
                self.logger.error("모델이 학습되지 않았습니다")
                return []

            # 피처 생성
            df = self.feature_engineer.CreateFeatures(current_data)

            # 최신 데이터 선택
            latest_data = df.iloc[-1:][self.feature_names]
            X = latest_data.values

            if len(X) == 0 or np.isnan(X).any():
                self.logger.error("예측용 데이터가 유효하지 않습니다")
                return []

            results = []
            current_time = datetime.now()
            market = current_data['market'].iloc[-1] if 'market' in current_data.columns else 'UNKNOWN'

            # 가격 예측
            if prediction_type in ['price', 'both']:
                try:
                    price_pred = self.price_model.predict(X)[0]

                    # 신뢰도 계산 (트리들의 예측 분산 기반)
                    tree_predictions = np.array([tree.predict(X)[0] for tree in self.price_model.estimators_])
                    price_confidence = 1 / (1 + np.std(tree_predictions))

                    results.append(PredictionResult(
                        market=market,
                        timestamp=current_time,
                        prediction_type='price',
                        predicted_value=price_pred,
                        confidence=price_confidence,
                        reasoning=self._GeneratePriceReasoning(latest_data)
                    ))

                except Exception as e:
                    self.logger.error(f"가격 예측 오류: {e}")

            # 방향 예측
            if prediction_type in ['direction', 'both']:
                try:
                    direction_pred = self.direction_model.predict(X)[0]
                    direction_proba = self.direction_model.predict_proba(X)[0]

                    # 확률 분포
                    prob_dist = {
                        'down': direction_proba[0],
                        'up': direction_proba[1]
                    }

                    direction_confidence = max(direction_proba)

                    results.append(PredictionResult(
                        market=market,
                        timestamp=current_time,
                        prediction_type='direction',
                        predicted_value=float(direction_pred),
                        confidence=direction_confidence,
                        probability_distribution=prob_dist,
                        reasoning=self._GenerateDirectionReasoning(latest_data, prob_dist)
                    ))

                except Exception as e:
                    self.logger.error(f"방향 예측 오류: {e}")

            return results

        except Exception as e:
            self.logger.error(f"예측 오류: {e}")
            return []

    def _GeneratePriceReasoning(self, data: pd.DataFrame) -> str:
        """가격 예측 근거 생성"""
        try:
            reasoning_parts = []

            # 주요 피처들의 중요도와 현재 값 분석
            feature_importance = self.performance_metrics.get('feature_importance', {})

            # 상위 5개 중요 피처 분석
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            for feature, importance in top_features:
                if feature in data.columns:
                    value = data[feature].iloc[0]
                    reasoning_parts.append(f"{feature}: {value:.4f} (중요도: {importance:.3f})")

            return " | ".join(reasoning_parts)

        except Exception:
            return "피처 분석 실패"

    def _GenerateDirectionReasoning(self, data: pd.DataFrame, prob_dist: Dict) -> str:
        """방향 예측 근거 생성"""
        try:
            reasoning_parts = []

            # 확률 정보
            reasoning_parts.append(f"상승확률: {prob_dist['up']:.1%}")
            reasoning_parts.append(f"하락확률: {prob_dist['down']:.1%}")

            # RSI 분석
            if 'rsi_14' in data.columns:
                rsi = data['rsi_14'].iloc[0]
                if rsi > 70:
                    reasoning_parts.append("RSI 과매수")
                elif rsi < 30:
                    reasoning_parts.append("RSI 과매도")
                else:
                    reasoning_parts.append(f"RSI 중립({rsi:.1f})")

            # MACD 분석
            if 'macd_signal_positive' in data.columns:
                macd_positive = data['macd_signal_positive'].iloc[0]
                reasoning_parts.append("MACD 상승" if macd_positive else "MACD 하락")

            return " | ".join(reasoning_parts)

        except Exception:
            return "방향 분석 실패"

    def GetPerformanceMetrics(self) -> ModelPerformance:
        """
        모델 성능 지표 반환

        Returns:
            성능 지표 객체
        """
        if not self.performance_metrics:
            return ModelPerformance()

        price_metrics = self.performance_metrics.get('price_model', {})
        direction_metrics = self.performance_metrics.get('direction_model', {})

        return ModelPerformance(
            accuracy=direction_metrics.get('accuracy'),
            mse=price_metrics.get('mse'),
            rmse=price_metrics.get('rmse'),
            r2_score=price_metrics.get('r2_score'),
            feature_importance=self.performance_metrics.get('feature_importance')
        )

    def SaveModel(self, model_name: str = None):
        """
        모델 저장

        Args:
            model_name: 모델 저장명
        """
        try:
            if not model_name:
                model_name = f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")

            # 모델과 메타데이터 저장
            model_data = {
                'price_model': self.price_model,
                'direction_model': self.direction_model,
                'feature_names': self.feature_names,
                'config': self.config,
                'performance_metrics': self.performance_metrics,
                'is_trained': self.is_trained,
                'last_training_data_size': self.last_training_data_size
            }

            joblib.dump(model_data, model_path)
            self.logger.info(f"모델 저장 완료: {model_path}")

        except Exception as e:
            self.logger.error(f"모델 저장 오류: {e}")

    def LoadModel(self, model_path: str) -> bool:
        """
        모델 불러오기

        Args:
            model_path: 모델 파일 경로

        Returns:
            로드 성공 여부
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
                return False

            model_data = joblib.load(model_path)

            self.price_model = model_data['price_model']
            self.direction_model = model_data['direction_model']
            self.feature_names = model_data['feature_names']
            self.config = model_data['config']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = model_data['is_trained']
            self.last_training_data_size = model_data['last_training_data_size']

            self.logger.info(f"모델 로드 완료: {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"모델 로드 오류: {e}")
            return False

    def GetModelInfo(self) -> Dict:
        """
        모델 정보 반환

        Returns:
            모델 정보 딕셔너리
        """
        return {
            'is_trained': self.is_trained,
            'config': self.config,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_samples': self.last_training_data_size,
            'performance_metrics': self.performance_metrics
        }