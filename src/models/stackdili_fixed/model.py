import io
import contextlib
import os
import shutil
import subprocess
import numpy as np
import pandas as pd

from typing import Optional
from sklearn.model_selection import StratifiedKFold
from models.stackdili_fixed.ga.base import BaseGA
from models.stackdili_fixed.stacking.base import BaseStacking


class Model:
    """데이터 정제 → (선택) GA → Stacking 파이프라인 조립기."""

    def __init__(
        self,
        stacking: BaseStacking,
        ga: Optional[BaseGA] = None,
        stacking_version: str = "unknown",
        ga_version: Optional[str] = None,
        env: Optional[str] = None,
    ):
        # src/models/stackdili_fixed/ 기준으로 프로젝트 루트 계산
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.ga               = ga
        self.stacking         = stacking
        self.stacking_version = stacking_version
        self.ga_version       = ga_version
        self.env              = env  # None 또는 "env1" → 외부검증, "env2" → 10-Fold CV

    def _restore_features(self, features_path: str) -> None:
        """Feature_raw*.csv → Feature.csv 복원."""
        features_dir = os.path.dirname(features_path)
        custom = getattr(self.ga, "feature_raw_csv", None)
        if custom:
            raw_path = os.path.join(features_dir, custom)
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"[오류] GA 전용 피처 파일이 없습니다: {raw_path}\n"
                    f"  먼저 './run.sh add-features' 를 실행하세요."
                )
            shutil.copy2(raw_path, features_path)
            print(f"[원본 복원] {custom} → Feature.csv")
        else:
            raw_path = os.path.join(features_dir, "Feature_raw.csv")
            if os.path.exists(raw_path):
                shutil.copy2(raw_path, features_path)
                print("[원본 복원] Feature_raw.csv → Feature.csv")
            elif os.path.exists(features_path):
                shutil.copy2(features_path, raw_path)
                print("[최초 백업] Feature.csv → Feature_raw.csv (이후 실행부터 자동 복원)")

    def _build_save_dir(self, clean: bool) -> str:
        dir_name = f"stacking_{self.stacking_version}"
        if self.ga_version:
            dir_name += f"_ga_{self.ga_version}"
        if self.env == "env2":
            dir_name += "_env2"
        if clean:
            dir_name += "_clean"
        return os.path.join(
            self.project_root, "src", "models", "stackdili_fixed", "Model", dir_name
        )

    def run(self, clean: bool = False):
        features_path = os.path.join(self.project_root, "src", "features", "Feature.csv")

        # 매 실행 시 원본 자동 복원
        self._restore_features(features_path)

        # 데이터 정제 (Train-Test 중복 제거, 선택 사항)
        if clean:
            print("[전처리] 데이터 정제")
            clean_script = os.path.join(self.project_root, "src", "preprocessing", "make_clean_data.py")
            subprocess.run(["python", clean_script], check=True, text=True)
            cleaned_csv = os.path.join(self.project_root, "src", "features", "Feature_cleaned.csv")
            shutil.copy2(cleaned_csv, features_path)

        if self.env == "env2":
            self._run_env2(features_path, clean)
        else:
            # env1 (기본값): 외부 검증 — NCTR/Greene/Xu/Liew 학습, DILIrank 테스트
            self._run_env1(features_path, clean)

    # ------------------------------------------------------------------
    # env1: 외부 검증
    # ------------------------------------------------------------------
    def _run_env1(self, features_path: str, clean: bool):
        print("[env1] 외부 검증 모드 — NCTR/Greene/Xu/Liew 학습, DILIrank 테스트")

        # GA 피처 선택 (학습 데이터에서만)
        if self.ga is not None:
            print("[GA] 피처 선택 중...")
            raw = pd.read_csv(features_path)
            train_raw   = raw[raw['ref'] != 'DILIrank']
            X_train_raw = train_raw.drop(['SMILES', 'Label', 'ref'], axis=1)
            y_train_raw = train_raw['Label']
            selected_cols = self.ga.select_features(X_train_raw, y_train_raw)
            if not all(c in raw.columns for c in selected_cols):
                raw = pd.read_csv(features_path)
            raw[['SMILES', 'Label', 'ref'] + selected_cols].to_csv(features_path, index=False)

        cleaned = pd.read_csv(features_path)
        train = cleaned[cleaned['ref'] != 'DILIrank']
        test  = cleaned[cleaned['ref'] == 'DILIrank']

        X_train = train.drop(['SMILES', 'Label', 'ref'], axis=1)
        y_train = train['Label'].values
        X_test  = test.drop(['SMILES', 'Label', 'ref'], axis=1)
        y_test  = test['Label'].values

        save_dir = self._build_save_dir(clean)
        os.makedirs(save_dir, exist_ok=True)
        self.stacking.fit(X_train, y_train, X_test, y_test, save_dir)
        self.stacking.evaluate(X_test, y_test, save_dir)

        ga_label = f"GA {self.ga_version}" if self.ga_version else "GA 없음"
        print(f"\nStacking {self.stacking_version}  |  {ga_label}  |  env=env1  |  clean={clean}")

    # ------------------------------------------------------------------
    # env2: 10-Fold CV (전체 데이터셋 합산)
    # ------------------------------------------------------------------
    def _run_env2(self, features_path: str, clean: bool):
        print("[env2] 10-Fold CV 모드 — 전체 데이터셋 합산")

        data = pd.read_csv(features_path)
        X    = data.drop(['SMILES', 'Label', 'ref'], axis=1)
        y    = data['Label'].values

        # GA 피처 선택 (전체 데이터 기준, CV 루프 밖에서 1회 수행)
        if self.ga is not None:
            print("[GA] 피처 선택 중 (전체 데이터 기준)...")
            selected_cols = self.ga.select_features(X, pd.Series(y))
            X = X[selected_cols]

        save_dir_base = self._build_save_dir(clean)
        skf     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        metrics_keys = ["auc", "acc", "mcc", "f1", "prec", "sens", "spec"]
        fold_results = []

        print(f"\n{'─'*90}")
        print(f"  {'Fold':>6}  {'AUC':>7}  {'ACC':>7}  {'MCC':>7}  {'F1':>7}  {'Prec':>7}  {'Sens':>7}  {'Spec':>7}")
        print(f"{'─'*90}")

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            X_tr = X.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_te = X.iloc[te_idx]
            y_te = y[te_idx]

            fold_dir = os.path.join(save_dir_base, f"fold_{fold_idx + 1:02d}")
            os.makedirs(fold_dir, exist_ok=True)

            with contextlib.redirect_stdout(io.StringIO()):
                self.stacking.fit(X_tr, y_tr, X_te, y_te, fold_dir)
                result = self.stacking.evaluate(X_te, y_te, fold_dir, verbose=False)

            fold_results.append(result)
            print(
                f"  {fold_idx + 1:>5}/10"
                f"  {result['auc']:>7.4f}"
                f"  {result['acc']:>7.4f}"
                f"  {result.get('mcc', float('nan')):>7.4f}"
                f"  {result['f1']:>7.4f}"
                f"  {result['prec']:>7.4f}"
                f"  {result['sens']:>7.4f}"
                f"  {result['spec']:>7.4f}"
            )

        print(f"{'─'*90}")
        means = {k: np.mean([r[k] for r in fold_results if k in r]) for k in metrics_keys}
        stds  = {k: np.std( [r[k] for r in fold_results if k in r]) for k in metrics_keys}
        print(
            f"  {'평균':>6}"
            f"  {means['auc']:>7.4f}"
            f"  {means['acc']:>7.4f}"
            f"  {means['mcc']:>7.4f}"
            f"  {means['f1']:>7.4f}"
            f"  {means['prec']:>7.4f}"
            f"  {means['sens']:>7.4f}"
            f"  {means['spec']:>7.4f}"
        )
        print(
            f"  {'표준편차':>6}"
            f"  {stds['auc']:>7.4f}"
            f"  {stds['acc']:>7.4f}"
            f"  {stds['mcc']:>7.4f}"
            f"  {stds['f1']:>7.4f}"
            f"  {stds['prec']:>7.4f}"
            f"  {stds['sens']:>7.4f}"
            f"  {stds['spec']:>7.4f}"
        )
        print(f"{'─'*90}")

        ga_label = f"GA {self.ga_version}" if self.ga_version else "GA 없음"
        print(f"\nStacking {self.stacking_version}  |  {ga_label}  |  env=env2  |  clean={clean}")

    def predict(self, _):
        return None
