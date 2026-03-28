"""
RDKit 기반 Feature 추가 스크립트
Feature_raw.csv에 아래 두 종류의 feature를 추가하고 덮어씁니다.

  1. Morgan ECFP4 fingerprints  (radius=2, nBits=1024) → 컬럼명 Morgan_0 ~ Morgan_1023
  2. Physicochemical descriptors (9종)                → 컬럼명 RDKit_*

사용법 (Docker 내부에서 한 번만 실행):
  python src/features/add_rdkit_features.py
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(SCRIPT_DIR, "Feature_raw.csv")


# ──────────────────────────────────────────────
# 피처 계산 함수
# ──────────────────────────────────────────────

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def _morgan_fp(smiles: str) -> list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 1024
    return list(_morgan_gen.GetFingerprintAsNumPy(mol))


def _physchem(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "RDKit_MolLogP": 0.0,
            "RDKit_MolWt": 0.0,
            "RDKit_TPSA": 0.0,
            "RDKit_NumHDonors": 0,
            "RDKit_NumHAcceptors": 0,
            "RDKit_NumRotatableBonds": 0,
            "RDKit_NumAromaticRings": 0,
            "RDKit_RingCount": 0,
            "RDKit_FractionCSP3": 0.0,
        }
    return {
        "RDKit_MolLogP":           Descriptors.MolLogP(mol),
        "RDKit_MolWt":             Descriptors.MolWt(mol),
        "RDKit_TPSA":              rdMolDescriptors.CalcTPSA(mol),
        "RDKit_NumHDonors":        rdMolDescriptors.CalcNumHBD(mol),
        "RDKit_NumHAcceptors":     rdMolDescriptors.CalcNumHBA(mol),
        "RDKit_NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RDKit_NumAromaticRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
        "RDKit_RingCount":         rdMolDescriptors.CalcNumRings(mol),
        "RDKit_FractionCSP3":      rdMolDescriptors.CalcFractionCSP3(mol),
    }


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print(f"[RDKit Features] Feature_raw.csv 로딩 중...")
    df = pd.read_csv(RAW_PATH)
    n_orig_cols = len(df.columns)
    print(f"  샘플 수: {len(df)}, 기존 컬럼 수: {n_orig_cols}")

    # 이미 추가된 컬럼이 있으면 제거하고 재계산 (멱등성 보장)
    keep_cols = [c for c in df.columns if not (c.startswith("Morgan_") or c.startswith("RDKit_"))]
    df = df[keep_cols]

    # ── 1. Morgan ECFP4 fingerprints ──────────────
    print("[RDKit Features] Morgan ECFP4 fingerprints 계산 중... (radius=2, nBits=1024)")
    morgan_rows = df["SMILES"].apply(_morgan_fp)
    morgan_df   = pd.DataFrame(morgan_rows.tolist(),
                               columns=[f"Morgan_{i}" for i in range(1024)])

    # ── 2. Physicochemical descriptors ────────────
    print("[RDKit Features] Physicochemical descriptors 계산 중... (9종)")
    physchem_rows = df["SMILES"].apply(_physchem)
    physchem_df   = pd.DataFrame(physchem_rows.tolist())

    # ── 합치기 & 저장 ─────────────────────────────
    result_df = pd.concat(
        [df.reset_index(drop=True), morgan_df, physchem_df],
        axis=1
    )
    result_df.to_csv(RAW_PATH, index=False)

    added = len(result_df.columns) - len(keep_cols)
    print(f"[완료] Feature_raw.csv 업데이트 완료")
    print(f"  기존 컬럼: {len(keep_cols)}  →  추가: {added}  →  최종: {len(result_df.columns)}")
    print(f"  (Morgan ECFP4 x1024  +  RDKit physchem x9 = {added})")


if __name__ == "__main__":
    main()
