# dili-ml-pipeline
### Senior-Project_3

Dongguk University Capston Design Class 2, Group 3 Repository

# DILI ML Pipeline

StackDILI 개선 모델 개발 및 실험 파이프라인

자세한 실행 방법 및 가이드는 [GUIDE.md](GUIDE.md)를 참고하세요.

---

## 팀 역할 분담

| 역할 | 담당 파일 |
|------|-----------|
| 데이터 전처리 | `src/features/`, `src/preprocessing/` |
| GA + 베이스 모델 | `src/models/stackdili_fixed/ga/`, `src/models/stackdili_fixed/base_models/` |
| 스태킹 + 결과 분석 | `src/models/stackdili_fixed/stacking/` |
| XAI | `src/xai/` |

---

## 프로젝트 구조

```
dili_ml_pipeline/
├── run.sh                          # 실행 진입점
├── Dockerfile
├── docker-compose.yml
├── docker/
│   └── environment.yml             # conda 환경 정의
│
├── src/
│   ├── train.py                    # StackDILI 학습 진입점
│   ├── train_dgudili.py            # DGUDILI 학습 진입점
│   ├── registry.py                 # GA / Stacking 버전 등록
│   ├── env_test.py                 # 환경 확인용
│   │
│   ├── features/
│   │   ├── Feature.py              # SMILES → dataset_features.csv (425 FP)
│   │   ├── dataset_features.csv    # 전체 FP 피처 (425개, 덮어쓰기 금지)
│   │   └── chemberta_encoder.py    # SMILES → 768-dim CLS 임베딩 (캐싱)
│   │
│   ├── preprocessing/
│   │   └── make_clean_data.py      # Train-Test 중복 제거
│   │
│   └── models/
│       └── stackdili_fixed/
│           ├── model.py            # 조립기: GA → Stacking 순서 관리
│           │
│           ├── ga/
│           │   ├── base.py         # GA 인터페이스 (BaseGA)
│           │   ├── ga_v0.py        # GA 구현 v0 (원본 StackDILI, DEAP 기반)
│           │   └── shap_interpreter.py  # 학습 후 XGBoost SHAP 해석
│           │
│           ├── stacking/
│           │   ├── base.py         # Stacking 인터페이스 (BaseStacking)
│           │   ├── stacking_v0.py  # 원본 StackDILI (직접 예측 + ExtraTrees 메타)
│           │   ├── stacking_v0_5.py
│           │   └── stacking_v1.py  # OOF + LR 메타 + MCC 임계값 최적화
│           │
│           ├── cross_attention/
│           │   └── cross_attention.py   # DGUDILIModel (Mode B Cross-Attention)
│           │
│           ├── dgudili/
│           │   └── dgudili_pipeline.py  # DGUDILI 파이프라인 오케스트레이터
│           │
│           └── Model/              # 학습된 모델 저장 (자동 생성)
```
