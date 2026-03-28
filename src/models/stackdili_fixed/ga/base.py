from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseGA(ABC):
    # 서브클래스에서 재정의하면 해당 파일을 Feature.csv 복원 소스로 사용
    feature_raw_csv: Optional[str] = None

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """피처 선택 실행. 선택된 컬럼명 리스트를 반환."""
        pass
