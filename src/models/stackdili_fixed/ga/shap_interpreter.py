"""
SHAP Post-hoc 해석기

DGUDILI 학습 완료 후, 원본 FP 218개 feature 중
어떤 feature가 예측에 얼마나 기여했는지 XGBoost SHAP로 해석.

사용법:
    from models.stackdili_fixed.ga.shap_interpreter import SHAPInterpreter

    interp = SHAPInterpreter()
    interp.fit(X_train_fp, y_train)          # XGBoost 학습
    interp.plot(save_path="shap_summary.png") # SHAP bar plot 저장
    top_features = interp.top_k(k=16)         # 상위 16개 feature 이름 반환
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


class SHAPInterpreter:
    """
    XGBoost 기반 SHAP feature importance 계산기.

    NOTE: 이 클래스는 DGUDILI의 feature selection 도구가 아닌
          학습 완료 후 원본 FP feature의 기여도를 해석하는 post-hoc 도구.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.model       = None
        self.shap_values = None
        self.feature_names = None
        self.mean_abs_shap = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SHAPInterpreter":
        """
        XGBoost를 FP(218) feature로 학습하고 SHAP 값 계산.

        Parameters
        ----------
        X : pd.DataFrame - FP feature matrix (n_samples, n_features)
        y : np.ndarray  - Label (0/1)
        """
        from xgboost import XGBClassifier
        import shap

        self.feature_names = X.columns.tolist()

        print("[SHAP] XGBoost 학습 중...")
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_seed,
            verbosity=0,
        )
        self.model.fit(X.values, y)

        print("[SHAP] SHAP 값 계산 중 (TreeExplainer)...")
        explainer        = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X.values)   # (n_samples, n_features)

        # 각 feature의 global importance: mean(|SHAP|)
        self.mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        print(f"[SHAP] 완료. feature 수: {len(self.feature_names)}")
        return self

    def top_k(self, k: int = 16) -> list:
        """SHAP 기준 상위 k개 feature 이름 반환."""
        if self.mean_abs_shap is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        sorted_idx = np.argsort(self.mean_abs_shap)[::-1]
        return [self.feature_names[i] for i in sorted_idx[:k]]

    def importance_df(self) -> pd.DataFrame:
        """feature별 mean |SHAP| DataFrame 반환 (내림차순 정렬)."""
        if self.mean_abs_shap is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        df = pd.DataFrame({
            "feature":       self.feature_names,
            "mean_abs_shap": self.mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def plot(self, save_path: str = None, top_n: int = 20):
        """
        SHAP bar plot 저장 (또는 화면 출력).

        Parameters
        ----------
        save_path : str | None - 저장 경로 (.png). None이면 plt.show()
        top_n     : int        - 표시할 상위 feature 수
        """
        if self.mean_abs_shap is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        try:
            import matplotlib
            matplotlib.use("Agg" if save_path else "TkAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[SHAP] matplotlib 미설치 - plot 건너뜀.")
            return

        df = self.importance_df().head(top_n)

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="steelblue")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"SHAP Feature Importance - Top {top_n} FP Features")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[SHAP] 저장: {save_path}")
        else:
            plt.show()
        plt.close()

    def save_csv(self, save_path: str):
        """feature 중요도를 CSV로 저장."""
        df = self.importance_df()
        df.to_csv(save_path, index=False)
        print(f"[SHAP] CSV 저장: {save_path}")
