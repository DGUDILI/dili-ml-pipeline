# CLAUDE.md — DGUDILI Development Guide

## 프로젝트 목표

### 1차 목표: DGUDILI 모델 구현 및 검증

| 목표 | 세부 사항 |
|------|---------|
| **Feature 압축** | FP(425) + ChemBERTa(768) 이중 입력 → Cross-Attention → 16개 feature로 압축 |
| **성능** | StackDILI (ga_v0) 대비 AUC ≥ 0.930, MCC ≥ 0.480 (env1 기준) |
| **Interpretability** | 학습 후 SHAP로 원본 FP 425개 중 어떤 feature가 중요했는지 사후 해석 |

### 2차 목표: Cross-Attention 아키텍처 검증

| 관점 | 검증 항목 |
|------|---------|
| **CA 효과** | Cross-Attention 제거(단순 concatenate+MLP) 대비 ΔMCC ≥ 0.03 향상 필수 |
| **DGUDILI vs StackDILI** | DGUDILI(16-dim) vs StackDILI(ga_v0) 성능 비교 |

---

## 전체 아키텍처

```
SMILES (Dataset.csv)
    │
    ├──→ Feature.py (iFeatureOmegaCLI)
    │       Constitution + Pharmacophore + MACCS + E-state
    │       → dataset_features.csv  (총 425개 FP)
    │
    └──→ chemberta_encoder.py
            seyonec/ChemBERTa-zinc-base-v1
            → CLS embedding (768-dim, 캐싱)

FP(425) → 4개 그룹별 Linear Projection → FP_emb (batch, 4, d_model)
CB(768) → Linear Projection            → CB_emb  (batch, 1, d_model)

         ┌──────────────────────────────────────────────┐
         │       Cross-Attention (Mode B 고정)          │
         │  Q=CB_1token, K=V=FP_4tokens                │
         │  → ChemBERTa가 FP 4개 그룹 중 선택         │
         │  → output: (batch, 1, d_model)               │
         └──────────────────┬───────────────────────────┘
                            │
                   squeeze + nn.Linear → 16
                            │
                   Feature Space (k=16)
                            │
                  Logistic Regression
                            │
                  DILI Prediction (AUC, MCC)

[Post-hoc 해석]
학습 완료 후 XGBoost SHAP → 원본 FP 425개 feature 중요도 순위 시각화
```

---

## 핵심 설계 의사결정

### Q1. FP 차원 축소 방법: Selection vs Projection

**결론: Linear Projection (변환) — 사전 선택 없음**

FP를 4개 feature 그룹으로 분리해 시퀀스 토큰으로 처리:
```
const      (173)  → Linear(173, d_model) → token_0
pc         (  6)  → Linear(  6, d_model) → token_1
maccs      (167)  → Linear(167, d_model) → token_2
estate     ( 79)  → Linear( 79, d_model) → token_3
→ FP_emb: (batch, 4, d_model)
```
컬럼명 prefix 기준 자동 분류 (`get_fp_groups()`).

### Q2. Cross-Attention 모드 선택

**결론: Mode B 단일 고정**

| 모드 | 구조 | env1 결과 |
|------|------|---------|
| Mode A | Q=FP(4), K=V=CB(1) | AUC=0.8677, MCC=0.5565 |
| **Mode B (채택)** | **Q=CB(1), K=V=FP(4)** | **AUC=0.9176, MCC=0.7467** |

Mode B가 MCC 목표(0.480) 초과 달성. Mode A 폐기.

### Q3. SHAP의 역할

**결론: Feature Selection 도구 → Post-hoc 해석 도구**

`shap_interpreter.py`: 학습 완료 후 XGBoost SHAP로 원본 FP 425개의 기여도 시각화.

### Q4. Classifier

**결론: Logistic Regression 고정**

k=16 저차원에서 tree ensemble(RF/ET/HistGB/XGB)은 overfitting 위험.

---

## Cross-Attention 모듈 상세 설계 (Mode B)

```
d_model = 64 (기본값)

FP branch:
  const(173)  → Linear(173, 64)+LayerNorm+ReLU → (batch, 1, 64)
  pc(  6)     → Linear(  6, 64)+LayerNorm+ReLU → (batch, 1, 64)
  maccs(167)  → Linear(167, 64)+LayerNorm+ReLU → (batch, 1, 64)
  estate( 79) → Linear( 79, 64)+LayerNorm+ReLU → (batch, 1, 64)
  stack → FP_emb: (batch, 4, 64)

ChemBERTa branch:
  CB(768) → Linear(768, 64)+LayerNorm+ReLU → CB_emb: (batch, 1, 64)

Mode B — Q=CB, K=V=FP:
  Q: (batch, 1, 64), K: (batch, 4, 64), V: (batch, 4, 64)
  Attention: softmax(QKᵀ / √64) → (batch, 1, 4)
  Output: (batch, 1, 64) → Residual+LayerNorm → squeeze → Linear(64, 16) → (batch, 16)

최종 출력: k=16 feature ✓
```

### 학습 설정

- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss (pos_weight balanced)
- Epochs: 80
- Scheduler: CosineAnnealingLR
- Batch size: 32

---

## 파일 구조

```
src/
├── features/
│   ├── Feature.py                   # SMILES → dataset_features.csv (425 FP)
│   ├── dataset_features.csv         # 전체 FP (425개, GA 미수정)
│   └── chemberta_encoder.py         # SMILES → 768-dim CLS embedding + 캐싱
│
├── models/stackdili_fixed/
│   ├── ga/
│   │   ├── base.py
│   │   ├── ga_v0.py
│   │   └── shap_interpreter.py      # 학습 후 XGBoost SHAP 해석
│   │
│   ├── cross_attention/
│   │   ├── __init__.py
│   │   └── cross_attention.py       # DGUDILIModel (Mode B 고정)
│   │
│   └── dgudili/
│       ├── __init__.py
│       └── dgudili_pipeline.py      # 전체 파이프라인 오케스트레이터
│
├── train_dgudili.py                 # DGUDILI 진입점
└── train.py                         # StackDILI 진입점
```

**데이터 파일 분리 원칙**:
- `dataset_features.csv`: Feature.py 원본 출력 (425 FP). 절대 덮어쓰지 않음.
- StackDILI(model.py)는 `dataset_features.csv`를 읽어 GA 선택을 in-memory로만 처리.

---

## 평가 프로토콜

| 프로토콜 | 내용 | 성공 기준 |
|---------|------|---------|
| **env1** | External validation: NCTR+Greene+Xu+Liew → DILIrank test | AUC ≥ 0.930, MCC ≥ 0.480 |
| **env2** | 10-fold stratified CV | AUC_mean ≥ 0.925, MCC_mean ≥ 0.475, std < 0.03 |

### 현재 성능 (env1)

| 모델 | AUC | MCC | Sens | Spec |
|------|-----|-----|------|------|
| **DGUDILI** | 0.9176 | 0.7467 | 0.902 | 0.855 |
| StackDILI ga_v0 | ~0.890 | ~0.594 | 0.856 | 0.748 |

---

## 구현 체크리스트

### 개발 단계

- [x] **Step 1: 데이터 검증**
  - [x] FP 컬럼 그룹 확인 (const 173, pc 6, maccs 167, estate 79 = 425개)
  - [x] ChemBERTa 단일 SMILES 추론 테스트

- [x] **Step 2: ChemBERTa 인코더**
  - [x] `src/features/chemberta_encoder.py` 작성
  - [x] 전체 Dataset SMILES 인코딩 후 `chemberta_embeddings.npy` 저장

- [x] **Step 3: Cross-Attention 모듈**
  - [x] `src/models/stackdili_fixed/cross_attention/cross_attention.py` 작성 (Mode B 고정)
  - [x] Shape 검증: (batch, 1, 64) → squeeze → Linear(64,16) → (batch, 16)

- [x] **Step 4: DGUDILI 파이프라인**
  - [x] `src/models/stackdili_fixed/dgudili/dgudili_pipeline.py` 작성
  - [x] env1 실행 완료
  - [ ] env2 실행

- [ ] **Step 5: SHAP 해석기**
  - [ ] FP 그룹별 SHAP bar plot 출력

### 검증 단계

- [x] DGUDILI 실행 → AUC=0.9176, MCC=0.7467 (env1)
- [x] StackDILI (ga_v0) vs DGUDILI 성능 비교 완료
- [ ] Cross-Attention vs concatenate+MLP 비교

### 최종 산출물

- [x] `dgudili_model.pt` (Cross-Attention 모델)
- [x] `lr_classifier.pkl` (LR classifier)
- [x] `result.txt` (AUC)
- [ ] `shap_fp_importance.png` (원본 FP feature 중요도)

---

## 실행 명령

```bash
# env1 (외부 검증)
python src/train_dgudili.py --env env1

# env2 (10-Fold CV)
python src/train_dgudili.py --env env2

# SHAP 해석 포함
python src/train_dgudili.py --env env1 --shap

# StackDILI 비교
python src/train.py --stacking s0 --ga g0 --env env1
```

---

## 주의사항

- **FP 그룹 경계**: 컬럼명 prefix 기준 자동 분류 (`get_fp_groups()`). 하드코딩 금지.
- **ChemBERTa 캐싱**: 임베딩 추출은 GPU 없이 수 분 소요. 반드시 `.npy` 캐싱.
- **클래스 불균형**: Loss 및 LR 모두 `class_weight='balanced'`
- **재현성**: `torch.manual_seed(42)`, `np.random.seed(42)` 전역 설정
- **dataset_features.csv 보호**: StackDILI 실행 시에도 이 파일은 덮어쓰지 않음
