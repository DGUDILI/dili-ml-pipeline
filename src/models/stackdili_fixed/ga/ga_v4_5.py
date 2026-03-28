import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier

from models.stackdili_fixed.ga.base import BaseGA


class GAv4_5(BaseGA):
    """XGBoost + SHAP 기반 피처 선택 (v4.5).

    G4의 XGBoost L1/L2 파라미터 탐색 구조를 유지하되,
    feature 선택 기준을 gain 기반 feature_importances_ 에서
    XGBoost 네이티브 SHAP(pred_contribs)의 평균 절대 기여도로 교체합니다.

    - SHAP은 게임이론 Shapley 값 기반으로 일관성(consistency)이 보장됩니다.
    - gain importance의 고분산/상관 피처 편향 문제를 해결합니다.
    - 샘플이 줄어드는 clean 모드에서도 안정적인 feature 선택이 가능합니다.
    - 누적 SHAP 커버리지(shap_coverage) 기준으로 선택 피처 수를 제어합니다.

    RDKit 확장 피처셋(Feature_raw_rdkit.csv)을 전용으로 사용합니다.
    """

    feature_raw_csv = "Feature_raw_rdkit.csv"

    def __init__(
        self,
        reg_alphas:        list  = [0.01, 0.05, 0.1, 0.5],
        reg_lambdas:       list  = [0.1, 0.5, 1.0, 5.0],
        learning_rates:    list  = [0.05, 0.1],   # LR 탐색 추가 — 피처 증가 시 낮은 LR 유리
        max_depth:         int   = 7,
        subsample:         float = 0.8,
        colsample_bytree:  float = 0.8,
        shap_coverage:     float = 0.95,
        cv_folds:          int   = 5,
        cv_repeats:        int   = 3,
        n_estimators:      int   = 300,            # CV용
        n_estimators_final: int  = 500,            # 최종 재학습용 (더 많은 트리)
        min_features:      int   = 50,
        random_seed:       int   = 42,
        n_jobs:            int   = -1,
    ):
        self.reg_alphas         = reg_alphas
        self.reg_lambdas        = reg_lambdas
        self.learning_rates     = learning_rates
        self.max_depth          = max_depth
        self.subsample          = subsample
        self.colsample_bytree   = colsample_bytree
        self.shap_coverage      = shap_coverage
        self.cv_folds           = cv_folds
        self.cv_repeats         = cv_repeats
        self.n_estimators       = n_estimators
        self.n_estimators_final = n_estimators_final
        self.min_features       = min_features
        self.random_seed        = random_seed
        self.n_jobs             = n_jobs

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _make_xgb(self, reg_alpha, reg_lambda, scale_pos_weight,
                  learning_rate=None, n_estimators=None):
        return XGBClassifier(
            n_estimators=n_estimators if n_estimators is not None else self.n_estimators,
            learning_rate=learning_rate if learning_rate is not None else self.learning_rates[0],
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=0.5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

    def _search_best_params(self, X_vals, y_vals, scale_pos_weight):
        """(reg_alpha, reg_lambda, learning_rate) 후보 × RepeatedStratifiedKFold CV → 최적 조합 반환."""
        skf = RepeatedStratifiedKFold(
            n_splits=self.cv_folds, n_repeats=self.cv_repeats, random_state=self.random_seed
        )

        best_mcc    = -np.inf
        best_params = (self.reg_alphas[0], self.reg_lambdas[0], self.learning_rates[0])

        for lr in self.learning_rates:
            for alpha in self.reg_alphas:
                for lam in self.reg_lambdas:
                    mcc_scores = []
                    model = self._make_xgb(alpha, lam, scale_pos_weight, learning_rate=lr)

                    for tr_idx, val_idx in skf.split(X_vals, y_vals):
                        model.fit(X_vals[tr_idx], y_vals[tr_idx])
                        y_pred = model.predict(X_vals[val_idx])
                        mcc_scores.append(matthews_corrcoef(y_vals[val_idx], y_pred))

                    mean_mcc = float(np.mean(mcc_scores))
                    print(
                        f"  lr={lr:.2f}  alpha={alpha:5.2f}  lambda={lam:5.2f}"
                        f" → MCC={mean_mcc:.4f}"
                    )

                    if mean_mcc > best_mcc:
                        best_mcc    = mean_mcc
                        best_params = (alpha, lam, lr)

        return best_params, best_mcc

    def _shap_select(self, model, X_vals, n_features: int):
        """XGBoost 네이티브 SHAP 기반 feature 선택.

        shap 라이브러리 대신 xgboost의 pred_contribs=True를 사용합니다.
        XGBoost 2.x / SHAP 버전 호환 문제를 우회하며 추가 의존성이 없습니다.

        Returns:
            selected_idx : 커버리지 기준 선택된 인덱스 리스트
            sorted_idx   : |SHAP| 내림차순 전체 인덱스 (fallback 용)
        """
        booster     = model.get_booster()
        dmat        = xgboost.DMatrix(X_vals)
        # pred_contribs=True → shape: (n_samples, n_features + 1)
        # 마지막 열은 bias(기저값)이므로 제외
        contribs    = booster.predict(dmat, pred_contribs=True)
        shap_values = contribs[:, :-1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (n_features,)

        # 내림차순 정렬 후 누적 커버리지 기준 cutoff
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        cumsum     = np.cumsum(mean_abs_shap[sorted_idx])
        total      = cumsum[-1]

        if total == 0:
            # SHAP 값이 모두 0인 엣지 케이스
            return sorted_idx[:self.min_features].tolist(), sorted_idx

        cutoff       = int(np.searchsorted(cumsum, self.shap_coverage * total))
        selected_idx = sorted_idx[:cutoff + 1].tolist()

        print(
            f"[XGB-SHAP] 누적 SHAP {self.shap_coverage*100:.0f}% 커버: "
            f"{len(selected_idx)}개 / {n_features}개 피처"
        )
        return selected_idx, sorted_idx

    # ------------------------------------------------------------------
    # BaseGA 인터페이스
    # ------------------------------------------------------------------

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        X_vals = X.values
        y_vals = y.values

        # 1. 불균형 보정 가중치
        n_neg = int((y_vals == 0).sum())
        n_pos = int((y_vals == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"[XGB-SHAP] scale_pos_weight={scale_pos_weight:.4f} (불균형 보정)")

        # 2. CV 기반 최적 (reg_alpha, reg_lambda) 탐색
        n_combos = len(self.learning_rates) * len(self.reg_alphas) * len(self.reg_lambdas)
        print(
            f"[XGB-SHAP] 파라미터 탐색 중..."
            f" ({n_combos}조합 × {self.cv_folds}-fold × {self.cv_repeats}반복 CV)"
        )
        (best_alpha, best_lambda, best_lr), best_mcc = self._search_best_params(
            X_vals, y_vals, scale_pos_weight
        )
        print(
            f"[XGB-SHAP] 최적: lr={best_lr}, reg_alpha={best_alpha}, reg_lambda={best_lambda}"
            f" (MCC={best_mcc:.4f})"
        )

        # 3. 전체 데이터로 재학습 (n_estimators_final=500)
        print(f"[XGB-SHAP] 전체 데이터로 재학습 중... (n_estimators={self.n_estimators_final})")
        final_model = self._make_xgb(
            best_alpha, best_lambda, scale_pos_weight,
            learning_rate=best_lr, n_estimators=self.n_estimators_final
        )
        final_model.fit(X_vals, y_vals)

        # 4. SHAP 기반 feature 선택
        print("[XGB-SHAP] SHAP 값 계산 중...")
        selected_idx, sorted_idx = self._shap_select(final_model, X_vals, X.shape[1])

        # 5. min_features fallback
        if len(selected_idx) < self.min_features:
            print(
                f"[XGB-SHAP] 선택 피처 {len(selected_idx)}개 < "
                f"min_features({self.min_features}), "
                f"SHAP 상위 {self.min_features}개로 fallback"
            )
            selected_idx = sorted_idx[:self.min_features].tolist()

        selected_cols = X.columns[selected_idx].tolist()
        print(f"[XGB-SHAP] 최종 선택: {len(selected_cols)}개 피처")
        return selected_cols
