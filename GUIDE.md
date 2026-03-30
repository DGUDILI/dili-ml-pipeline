# 실행 가이드

---

## 목차

1. [환경 세팅 (Docker)](#환경-세팅)
2. [StackDILI 실행](#stackdili-실행)
3. [DGUDILI 실행](#dgudili-실행)
4. [현재 버전 목록](#현재-버전-목록)
5. [파이프라인 흐름](#파이프라인-흐름)
6. [새 버전 추가하기](#새-버전-추가하기)

---

## 환경 세팅

> **Windows / Mac 공통** — Docker만 설치되어 있으면 됩니다.

### 1. Docker Desktop 설치

- [Docker Desktop 다운로드](https://www.docker.com/products/docker-desktop/)

### 2. 이미지 빌드

```bash
./run.sh build
```

Windows PowerShell에서는:
```powershell
bash run.sh build
```

> 첫 빌드는 conda 환경 생성 때문에 5~10분 걸릴 수 있습니다.

### 3. 환경 확인

```bash
./run.sh env-test
```

---

## StackDILI 실행

```bash
./run.sh run [s버전] [g버전] [env버전] [clean]
```

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `s버전` | Stacking 버전. `s` 접두사 사용 (`s0`, `s0.5`, `s1`) | `s1` |
| `g버전` | GA 버전. `g` 접두사 사용 (`g0`). 생략하면 GA 없이 실행 | 생략 |
| `env버전` | 실험 환경. `env` 접두사 사용 (`env1`, `env2`). 생략 시 env1과 동일 | 생략 |
| `clean` | Train-Test 중복 제거 실행 | 생략 |

인자는 **순서 무관** — 접두사(`s`, `g`, `env`)로 구분합니다.

### 실험 환경 (env)

| 환경 | 설명 |
|------|------|
| `env1` | **외부 검증** — NCTR, Greene, Xu, Liew로 학습 → DILIrank로 테스트 |
| `env2` | **10-Fold CV** — 전체 5개 데이터셋 합산 후 10겹 교차 검증, 평균 AUC 출력 |

### 예시

```bash
./run.sh run                        # Stacking s1, GA 없음 (env1)
./run.sh run s1 env1                # Stacking s1, 외부 검증
./run.sh run s1 env2                # Stacking s1, 10-Fold CV
./run.sh run s1 g0 env1             # Stacking s1 + GA g0 + 외부 검증
./run.sh run s1 g0 env2             # Stacking s1 + GA g0 + 10-Fold CV
./run.sh run s1 g0 env2 clean       # + 데이터 정제
```

> - GA를 생략하면 `dataset_features.csv`의 전체 피처(425개)로 바로 Stacking을 실행합니다.
> - `env2`는 10개 fold를 순차 실행하므로 env1보다 시간이 약 10배 걸립니다.
> - 데이터 정제(`clean`)는 Train-Test 중복 제거가 필요한 경우에만 사용하세요.

### 결과 저장 위치

버전 조합별로 별도 디렉토리에 저장되므로 결과가 덮어써지지 않습니다.

```
src/models/stackdili_fixed/Model/
├── stacking_s1/                          # ./run.sh run s1
├── stacking_s1_ga_g0/                    # ./run.sh run s1 g0 env1
├── stacking_s1_env2/                     # ./run.sh run s1 env2
│   ├── fold_01/
│   ├── fold_02/
│   └── ...fold_10/
└── stacking_s1_ga_g0_env2_clean/         # ./run.sh run s1 g0 env2 clean
    ├── fold_01/
    └── ...
```

### 컨테이너 쉘 접속 (직접 실험할 때)

```bash
./run.sh shell

# 컨테이너 안에서
conda run -n dili_ml_pipeline_env python src/train.py --stacking s1 --ga g0 --env env2
```

---

## DGUDILI 실행

DGUDILI는 FP(425개) + ChemBERTa(768-dim) 이중 입력을 Cross-Attention으로 결합해
16-dim 피처로 압축한 뒤 Logistic Regression으로 분류하는 모델입니다.

```bash
./run.sh dgudili [옵션]
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--env` | 실험 환경 (`env1` 또는 `env2`) | `env1` |
| `--epochs` | 학습 에포크 수 | `80` |
| `--batch_size` | 배치 크기 | `32` |
| `--d_model` | Cross-Attention 임베딩 차원 | `64` |
| `--n_heads` | Multi-head Attention 헤드 수 | `4` |
| `--lr` | 학습률 | `1e-3` |
| `--seed` | 랜덤 시드 | `42` |
| `--clean` | Train-Test 중복 제거 후 실행 | 생략 |
| `--shap` | 학습 후 SHAP 피처 중요도 분석 (env1 전용) | 생략 |

### 예시

```bash
./run.sh dgudili                         # env1 외부 검증 (기본값)
./run.sh dgudili --env env1              # env1 외부 검증 (명시)
./run.sh dgudili --env env2              # env2 10-Fold CV
./run.sh dgudili --env env1 --shap       # env1 + SHAP 피처 중요도 분석
./run.sh dgudili --env env1 --clean      # 데이터 정제 후 env1 실행
./run.sh dgudili --env env1 --epochs 100 --d_model 128  # 하이퍼파라미터 변경
```

### 사전 준비

DGUDILI 첫 실행 전 ChemBERTa 임베딩을 미리 추출해 두어야 합니다.
(GPU 없이 수 분 소요 — 이후 `.npy` 파일로 캐싱되어 재실행 불필요)

```bash
./run.sh shell

# 컨테이너 안에서
conda run -n dili_ml_pipeline_env python src/features/chemberta_encoder.py
```

### 결과 저장 위치

```
src/models/stackdili_fixed/Model/
├── dgudili_modeB_env1/               # ./run.sh dgudili --env env1
│   ├── dgudili_model.pt              # Cross-Attention 모델 가중치
│   ├── lr_classifier.pkl             # Logistic Regression 분류기
│   ├── scaler.pkl                    # StandardScaler
│   ├── config.pkl                    # 모델 설정
│   ├── result.txt                    # AUC 결과
│   └── shap/                         # --shap 옵션 사용 시
│       ├── shap_bar.png
│       └── shap_importance.csv
└── dgudili_modeB_env2/               # ./run.sh dgudili --env env2
    ├── fold_01/
    ├── fold_02/
    └── ...fold_10/
```

---

## 현재 버전 목록

### GA

| 버전 | 파일 | 설명 |
|------|------|------|
| `g0` | `ga/ga_v0.py` | 원본 StackDILI GA — DEAP 기반, RF 5-fold CV 피트니스 |

### Stacking

| 버전 | 파일 | 설명 |
|------|------|------|
| `s0` | `stacking/stacking_v0.py` | 원본 StackDILI — 직접 예측 기반, ExtraTrees 메타 모델 |
| `s0.5` | `stacking/stacking_v0_5.py` | OOF 기반, LR/SVC(스케일) + ExtraTrees 메타 모델 |
| `s1` | `stacking/stacking_v1.py` | OOF 기반, LogisticRegression 메타 모델 + 피처 힌트 + MCC 임계값 최적화 |

**Stacking 버전 비교:**

| | `s0` | `s0.5` | `s1` |
|---|---|---|---|
| 예측 방식 | 직접 예측 (데이터 누수) | OOF 5-fold | OOF 5-fold |
| 베이스 모델 | RF, ET, HistGB, XGB | LR, SVC, RF, XGB, LGBM | RF, ET, HistGB, XGB |
| 메타 모델 | ExtraTrees | ExtraTrees | LogisticRegression |
| 피처 힌트 | 없음 | 없음 | TOP 5 피처 추가 |

---

## 파이프라인 흐름

### StackDILI

```
[전처리] Feature.py (iFeatureOmegaCLI)
     ↓ src/features/dataset_features.csv 생성 (425개 FP, 덮어쓰기 금지)
     ↓
[데이터 정제, 선택] make_clean_data.py  ← --clean 옵션 지정 시 실행
     ↓ Train-Test 중복 제거 → dataset_features_cleaned.csv
     ↓
[GA, 선택] ga_vN.py → select_features()  ← --ga 옵션 지정 시 실행
     ↓ in-memory 피처 선택 (원본 파일 변경 없음)
     ↓
[스태킹] stacking_vN.py → fit() → evaluate()
     ↓ src/models/stackdili_fixed/Model/stacking_{sv}_ga_{gv}/ 에 pkl 저장
     ↓
결과 출력 (OOF AUC / Eval AUC)
```

### DGUDILI

```
[전처리] Feature.py (iFeatureOmegaCLI)
     ↓ src/features/dataset_features.csv (425개 FP)
     ↓
[임베딩] chemberta_encoder.py
     ↓ src/features/chemberta_embeddings.npy (768-dim, 캐싱)
     ↓
[모델 학습] DGUDILIModel (Cross-Attention Mode B)
     ↓ FP 4그룹 Linear Projection + ChemBERTa Projection
     ↓ Q=ChemBERTa(1 token), K=V=FP(4 tokens) → 16-dim 피처
     ↓
[분류기] LogisticRegression (k=16 피처 입력)
     ↓ src/models/stackdili_fixed/Model/dgudili_modeB_{env}/ 에 저장
     ↓
결과 출력 (AUC / MCC / Sens / Spec)
```

### 데이터 파일 원칙

- `dataset_features.csv`: Feature.py 원본 출력 (425 FP). **절대 덮어쓰지 않음.**
- GA 피처 선택은 in-memory로만 처리 — 원본 파일 유지.
- `--clean` 사용 시에만 `dataset_features_cleaned.csv`가 별도 생성됨.

---

## 새 버전 추가하기

### GA 새 버전 추가 (예: ga_v2.py)

**1. 파일 생성** `src/models/stackdili_fixed/ga/ga_v2.py`

```python
from models.stackdili_fixed.ga.base import BaseGA
import pandas as pd

class GAv2(BaseGA):
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        # 새 피처 선택 로직 작성
        # 반드시 선택된 컬럼명 리스트를 반환해야 함
        selected_cols = [...]
        return selected_cols
```

**2. registry.py에 등록** `src/registry.py`

```python
GA_REGISTRY = {
    "g0": None,
    "g2": None,  # 추가
}

# _load_ga() 함수에도 분기 추가:
if version == "g2":
    from models.stackdili_fixed.ga.ga_v2 import GAv2
    return GAv2
```

**3. 실행**

```bash
./run.sh run s1 g2
```

---

### Stacking 새 버전 추가 (예: stacking_v2.py)

**1. 파일 생성** `src/models/stackdili_fixed/stacking/stacking_v2.py`

```python
from models.stackdili_fixed.stacking.base import BaseStacking
import pandas as pd
import numpy as np

class StackingV2(BaseStacking):
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_test: pd.DataFrame, y_test: np.ndarray,
            save_dir: str) -> None:
        # 학습 로직 작성
        # 모델은 save_dir에 pkl로 저장
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray,
                 save_dir: str) -> dict:
        # 평가 로직 작성
        # 반드시 {"auc": float, "threshold": float} 딕셔너리 반환
        return {"auc": ..., "threshold": ...}
```

**2. registry.py에 등록** `src/registry.py`

```python
STACKING_REGISTRY = {
    "s0":  None,
    "s0.5": None,
    "s1":  None,
    "s2":  None,  # 추가
}

# _load_stacking() 함수에도 분기 추가:
if version == "s2":
    from models.stackdili_fixed.stacking.stacking_v2 import StackingV2
    return StackingV2
```

**3. 실행**

```bash
./run.sh run s2
```

---

## 인터페이스 규칙

새 버전을 만들 때 반드시 지켜야 하는 규칙입니다.

### BaseGA (`src/models/stackdili_fixed/ga/base.py`)

```python
def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
    ...
    return selected_cols  # 반드시 컬럼명 문자열 리스트 반환
```

### BaseStacking (`src/models/stackdili_fixed/stacking/base.py`)

```python
def fit(self, X_train, y_train, X_test, y_test, save_dir) -> None:
    # save_dir에 pkl 저장

def evaluate(self, X_test, y_test, save_dir) -> dict:
    return {"auc": float, "threshold": float}  # 이 형식 유지
```
