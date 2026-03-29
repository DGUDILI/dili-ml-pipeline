#!/usr/bin/env bash

CMD=${1}

case "$CMD" in
  build)
    docker compose build
    ;;
  shell)
    docker compose run --rm ml bash
    ;;
  run)
    STACKING="s1"
    GA=""
    ENV=""
    CLEAN=""

    # 접두사(s=stacking, g=ga, env=환경)로 인자 구분 — 순서 무관
    for arg in "${@:2}"; do
      case "$arg" in
        s*)    STACKING="$arg" ;;
        g*)    GA="$arg" ;;
        env*)  ENV="$arg" ;;
        clean) CLEAN="clean" ;;
      esac
    done

    GA_ARG=""
    ENV_ARG=""
    CLEAN_ARG=""
    if [ -n "$GA" ]; then
      GA_ARG="--ga=$GA"
    fi
    if [ -n "$ENV" ]; then
      ENV_ARG="--env=$ENV"
    fi
    if [ "$CLEAN" = "clean" ]; then
      CLEAN_ARG="--clean"
    fi
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/train.py --stacking="$STACKING" $GA_ARG $ENV_ARG $CLEAN_ARG
    ;;
  env-test)
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/env_test.py
    ;;
  add-features)
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/features/add_rdkit_features.py
    ;;
  *)
    echo "Usage: ./run.sh {build|shell|run [...options]|env-test|add-features}"
    echo "  stacking: s0 | s0.5 | s1          (기본값: s1, 접두사 s)"
    echo "  ga:       g0                       (생략 가능, 접두사 g)"
    echo "  env:      env1 | env2              (생략 시 env1과 동일, 접두사 env)"
    echo "              env1 = 외부 검증 (NCTR/Greene/Xu/Liew 학습, DILIrank 테스트)"
    echo "              env2 = 10-Fold CV (전체 데이터셋 합산)"
    echo "  clean:    'clean' 입력 시 Train-Test 중복 제거 실행"
    echo "  * 순서 무관"
    echo ""
    echo "  예시: ./run.sh run                       # Stacking s1, GA 없음 (env1)"
    echo "        ./run.sh run s1 env1               # Stacking s1, env1 외부 검증"
    echo "        ./run.sh run s1 env2               # Stacking s1, env2 10-Fold CV"
    echo "        ./run.sh run s1 g0 env1            # Stacking s1 + GA g0 + env1"
    echo "        ./run.sh run s1 g0 env2 clean      # Stacking s1 + GA g0 + env2 + 정제"
    echo "        ./run.sh add-features              # RDKit feature 추가 (최초 1회)"
    exit 1
    ;;
esac
