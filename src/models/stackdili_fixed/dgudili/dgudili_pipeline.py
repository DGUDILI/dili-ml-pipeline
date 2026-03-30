"""
DGUDILI 전체 파이프라인

env1: NCTR+Greene+Xu+Liew 학습, DILIrank 테스트
env2: 10-Fold Stratified CV

학습 흐름:
  1. FP(218) → 4개 그룹으로 분리
  2. ChemBERTa CLS 임베딩 로드 (캐시)
  3. DGUDILIModel (Cross-Attention) PyTorch 학습
  4. 16-dim feature 추출
  5. sklearn LogisticRegression 피팅
  6. 지표 평가 및 저장
"""

import io
import os
import contextlib
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from models.stackdili_fixed.cross_attention.cross_attention import DGUDILIModel, get_fp_groups


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    mcc  = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(auc=auc, acc=acc, mcc=mcc, f1=f1, prec=prec, sens=sens, spec=spec)


def _print_metrics(label: str, m: dict):
    print(
        f"  {label:<24}"
        f"  AUC={m['auc']:.4f}  ACC={m['acc']:.4f}  MCC={m['mcc']:.4f}"
        f"  F1={m['f1']:.4f}  Prec={m['prec']:.4f}"
        f"  Sens={m['sens']:.4f}  Spec={m['spec']:.4f}"
    )


def _save_metrics(m: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    lines = [
        f"AUC:  {m['auc']:.4f}",
        f"MCC:  {m['mcc']:.4f}",
        f"ACC:  {m['acc']:.4f}",
        f"F1:   {m['f1']:.4f}",
        f"Prec: {m['prec']:.4f}",
        f"Sens: {m['sens']:.4f}",
        f"Spec: {m['spec']:.4f}",
    ]
    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        f.write("\n".join(lines))
    # result.txt 첫 줄에 AUC 단독 기재 (기존 포맷 호환)
    with open(os.path.join(save_dir, "result.txt"), "w") as f:
        f.write(str(m["auc"]))


# ─────────────────────────────────────────────────────────────────────────────
# FP / ChemBERTa 전처리
# ─────────────────────────────────────────────────────────────────────────────

def _split_fp_groups(X_df: pd.DataFrame, group_cols: dict) -> list:
    """DataFrame → 그룹별 numpy array 리스트."""
    return [X_df[cols].values.astype(np.float32) for cols in group_cols.values()]


def _load_cb_embeddings(smiles_series: pd.Series) -> np.ndarray:
    """SMILES Series → (n, 768) numpy array."""
    import sys
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    features_dir = os.path.join(src_dir, "features")
    if features_dir not in sys.path:
        sys.path.insert(0, features_dir)

    from chemberta_encoder import load_embeddings_for_df
    return load_embeddings_for_df(smiles_series).astype(np.float32)


def _to_tensors(fp_groups_np: list, cb_np: np.ndarray, y_np: np.ndarray, device):
    fp_tensors = [torch.tensor(g, dtype=torch.float32, device=device) for g in fp_groups_np]
    cb_tensor  = torch.tensor(cb_np,  dtype=torch.float32, device=device)
    y_tensor   = torch.tensor(y_np,   dtype=torch.float32, device=device)
    return fp_tensors, cb_tensor, y_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 학습 / 추론
# ─────────────────────────────────────────────────────────────────────────────

def _train_dgudili(
    fp_groups_train: list,
    cb_train: np.ndarray,
    y_train: np.ndarray,
    fp_group_dims: dict,
    device,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> DGUDILIModel:
    """Cross-Attention 모델 학습 후 반환."""
    _set_seed(seed)

    model = DGUDILIModel(fp_group_dims=fp_group_dims).to(device)

    # 클래스 불균형 가중치
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(y_train)
    indices = np.arange(n)

    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            fp_batch = [torch.tensor(g[idx], dtype=torch.float32, device=device)
                        for g in fp_groups_train]
            cb_batch = torch.tensor(cb_train[idx], dtype=torch.float32, device=device)
            y_batch  = torch.tensor(y_train[idx],  dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(fp_batch, cb_batch).squeeze(1)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        scheduler.step()
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:>3}/{epochs}  loss={epoch_loss/n:.4f}")

    model.eval()
    return model


def _extract_features(
    model: DGUDILIModel,
    fp_groups: list,
    cb: np.ndarray,
    device,
    batch_size: int = 128,
) -> np.ndarray:
    """학습된 모델로 16-dim feature 추출 (numpy)."""
    model.eval()
    n = cb.shape[0]
    features = []
    for start in range(0, n, batch_size):
        fp_batch = [torch.tensor(g[start:start+batch_size], dtype=torch.float32, device=device)
                    for g in fp_groups]
        cb_batch = torch.tensor(cb[start:start+batch_size], dtype=torch.float32, device=device)
        feat = model.extract_features(fp_batch, cb_batch)
        features.append(feat.cpu().numpy())
    return np.vstack(features)


def _fit_lr(X_feat: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
    """16-dim feature로 sklearn LR 피팅. (scaler, lr) 반환."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, class_weight="balanced")
    lr.fit(X_scaled, y)
    return scaler, lr


def _predict_lr(scaler, lr_model, X_feat: np.ndarray) -> tuple:
    X_scaled = scaler.transform(X_feat)
    y_prob = lr_model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인 클래스
# ─────────────────────────────────────────────────────────────────────────────

class DGUDILIPipeline:
    """
    DGUDILI 전체 파이프라인 오케스트레이터.

    사용법:
        pipeline = DGUDILIPipeline(mode='A', env='env1')
        pipeline.run(features_path, save_dir)
    """

    def __init__(
        self,
        env: str = "env1",        # 'env1' or 'env2'
        epochs: int = 80,
        batch_size: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.env        = env
        self.epochs     = epochs
        self.batch_size = batch_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.lr         = lr
        self.seed       = seed
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, features_path: str, save_dir: str):
        print(f"\n[DGUDILI] env={self.env}  device={self.device}")

        df      = pd.read_csv(features_path)
        X_all   = df.drop(["SMILES", "Label", "ref"], axis=1)
        smiles  = df["SMILES"]

        # FP 그룹 컬럼 확인
        group_cols = get_fp_groups(X_all)
        fp_group_dims = {g: len(cols) for g, cols in group_cols.items()}
        print(f"[DGUDILI] FP 그룹: { {g: len(c) for g, c in group_cols.items()} }")

        # ChemBERTa 임베딩 로드
        print("[DGUDILI] ChemBERTa 임베딩 로드 중...")
        cb_all = _load_cb_embeddings(smiles)

        if self.env == "env1":
            self._run_env1(df, X_all, cb_all, group_cols, fp_group_dims, save_dir)
        else:
            self._run_env2(df, X_all, cb_all, group_cols, fp_group_dims, save_dir)

    # ── env1 ──────────────────────────────────────────────────────────────────

    def _run_env1(self, df, X_all, cb_all, group_cols, fp_group_dims, save_dir):
        print("[env1] 외부 검증 - NCTR/Greene/Xu/Liew 학습, DILIrank 테스트")

        train_mask = df["ref"] != "DILIrank"
        test_mask  = df["ref"] == "DILIrank"

        X_tr   = X_all[train_mask]
        X_te   = X_all[test_mask]
        y_tr   = df["Label"].values[train_mask]
        y_te   = df["Label"].values[test_mask]
        cb_tr  = cb_all[train_mask.values]
        cb_te  = cb_all[test_mask.values]

        fp_tr = _split_fp_groups(X_tr, group_cols)
        fp_te = _split_fp_groups(X_te, group_cols)

        print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")

        # Cross-Attention 학습
        print("[DGUDILI] Cross-Attention 모델 학습 중...")
        ca_model = _train_dgudili(
            fp_tr, cb_tr, y_tr, fp_group_dims,
            device=self.device,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, seed=self.seed,
        )

        # 16-dim feature 추출
        feat_tr = _extract_features(ca_model, fp_tr, cb_tr, self.device)
        feat_te = _extract_features(ca_model, fp_te, cb_te, self.device)

        # LR 학습 및 평가
        scaler, lr_model = _fit_lr(feat_tr, y_tr, self.seed)
        y_pred, y_prob   = _predict_lr(scaler, lr_model, feat_te)
        metrics          = _compute_metrics(y_te, y_pred, y_prob)

        print("\n[최종 성능 - env1]")
        print("=" * 90)
        _print_metrics("DGUDILI", metrics)
        print("=" * 90)

        # 저장
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ca_model.state_dict(), os.path.join(save_dir, "dgudili_model.pt"))
        with open(os.path.join(save_dir, "dgudili_model_config.pkl"), "wb") as f:
            pickle.dump({
                "fp_group_dims": fp_group_dims,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
            }, f)
        with open(os.path.join(save_dir, "lr_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(save_dir, "lr_classifier.pkl"), "wb") as f:
            pickle.dump(lr_model, f)
        with open(os.path.join(save_dir, "result.txt"), "w") as f:
            f.write(str(metrics["auc"]))

        print(f"\n[DGUDILI] 저장 완료: {save_dir}")
        return metrics

    # ── env2 ──────────────────────────────────────────────────────────────────

    def _run_env2(self, df, X_all, cb_all, group_cols, fp_group_dims, save_dir):
        print("[env2] 10-Fold Stratified CV")

        y_all  = df["Label"].values
        skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        keys   = ["auc", "acc", "mcc", "f1", "prec", "sens", "spec"]
        fold_results = []

        print(f"\n{'─'*100}")
        print(f"  {'Fold':>5}  {'AUC':>7}  {'ACC':>7}  {'MCC':>7}  {'F1':>7}"
              f"  {'Prec':>7}  {'Sens':>7}  {'Spec':>7}")
        print(f"{'─'*100}")

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
            X_tr   = X_all.iloc[tr_idx]
            X_te   = X_all.iloc[te_idx]
            y_tr   = y_all[tr_idx]
            y_te   = y_all[te_idx]
            cb_tr  = cb_all[tr_idx]
            cb_te  = cb_all[te_idx]

            fp_tr = _split_fp_groups(X_tr, group_cols)
            fp_te = _split_fp_groups(X_te, group_cols)

            fold_dir = os.path.join(save_dir, f"fold_{fold_idx + 1:02d}")
            os.makedirs(fold_dir, exist_ok=True)

            with contextlib.redirect_stdout(io.StringIO()):
                ca_model = _train_dgudili(
                    fp_tr, cb_tr, y_tr, fp_group_dims,
                    device=self.device,
                    epochs=self.epochs, batch_size=self.batch_size,
                    lr=self.lr, seed=self.seed,
                )

            feat_tr = _extract_features(ca_model, fp_tr, cb_tr, self.device)
            feat_te = _extract_features(ca_model, fp_te, cb_te, self.device)

            scaler, lr_model = _fit_lr(feat_tr, y_tr, self.seed)
            y_pred, y_prob   = _predict_lr(scaler, lr_model, feat_te)
            m = _compute_metrics(y_te, y_pred, y_prob)
            fold_results.append(m)

            with open(os.path.join(fold_dir, "result.txt"), "w") as f:
                f.write(str(m["auc"]))

            print(
                f"  {fold_idx + 1:>4}/10"
                f"  {m['auc']:>7.4f}  {m['acc']:>7.4f}  {m['mcc']:>7.4f}"
                f"  {m['f1']:>7.4f}  {m['prec']:>7.4f}  {m['sens']:>7.4f}  {m['spec']:>7.4f}"
            )

        print(f"{'─'*100}")
        means = {k: np.mean([r[k] for r in fold_results]) for k in keys}
        stds  = {k: np.std( [r[k] for r in fold_results]) for k in keys}
        print(
            f"  {'평균':>5}"
            f"  {means['auc']:>7.4f}  {means['acc']:>7.4f}  {means['mcc']:>7.4f}"
            f"  {means['f1']:>7.4f}  {means['prec']:>7.4f}  {means['sens']:>7.4f}  {means['spec']:>7.4f}"
        )
        print(
            f"  {'표준편차':>5}"
            f"  {stds['auc']:>7.4f}  {stds['acc']:>7.4f}  {stds['mcc']:>7.4f}"
            f"  {stds['f1']:>7.4f}  {stds['prec']:>7.4f}  {stds['sens']:>7.4f}  {stds['spec']:>7.4f}"
        )
        print(f"{'─'*100}")

        return fold_results
