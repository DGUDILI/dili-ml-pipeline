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
    CLEAN=""

    # 접두사(s=stacking, g=ga)로 인자 구분 — 순서 무관
    for arg in "${@:2}"; do
      case "$arg" in
        s*)  STACKING="$arg" ;;
        g*)  GA="$arg" ;;
        clean) CLEAN="clean" ;;
      esac
    done

    GA_ARG=""
    CLEAN_ARG=""
    if [ -n "$GA" ]; then
      GA_ARG="--ga=$GA"
    fi
    if [ "$CLEAN" = "clean" ]; then
      CLEAN_ARG="--clean"
    fi
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/train.py --stacking="$STACKING" $GA_ARG $CLEAN_ARG
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
    echo "  stacking: s0 | s0.5 | s1 | s3   (기본값: s1, 접두사 s)"
    echo "  ga:       g0 | g1 | g4 | g5 (생략 가능, 접두사 g)"
    echo "  clean:    'clean' 입력 시 Train-Test 중복 제거 실행"
    echo "  * 순서 무관"
    echo ""
    echo "  예시: ./run.sh run                  # Stacking s1, GA 없음"
    echo "        ./run.sh run s1               # Stacking s1, GA 없음"
    echo "        ./run.sh run s1 g4            # Stacking s1 + GA g4"
    echo "        ./run.sh run g4 s1            # 순서 바꿔도 동일"
    echo "        ./run.sh run s1 g4 clean      # Stacking s1 + GA g4 + 정제"
    echo "        ./run.sh run clean s1         # Stacking s1 + 정제, GA 없음"
    echo "        ./run.sh add-features         # RDKit feature 추가 (최초 1회)"
    exit 1
    ;;
esac
