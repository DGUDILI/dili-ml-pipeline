from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseStacking(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
    ) -> None:
        """베이스 모델 + 메타 모델 학습 후 save_dir에 저장."""
        pass

    @abstractmethod
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        save_dir: str,
        verbose: bool = True,
    ) -> dict:
        """저장된 모델 로드 후 평가.
        {"auc", "threshold", "acc", "mcc", "f1", "prec", "sens", "spec"} 반환.
        verbose=False 이면 출력 없이 메트릭만 반환.
        """
        pass
