import argparse
from registry import GA_REGISTRY, STACKING_REGISTRY, ENV_REGISTRY, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacking", default="s1", choices=list(STACKING_REGISTRY),
                        help="Stacking 버전 (예: s0, s0.5, s1)")
    parser.add_argument("--ga", default=None, choices=list(GA_REGISTRY.keys()),
                        help="GA 버전 (예: g0). 생략하면 GA 없이 실행")
    parser.add_argument("--env", default=None, choices=list(ENV_REGISTRY.keys()),
                        help="실험 환경 (env1: 외부 검증, env2: 10-Fold CV). 생략 시 env1과 동일")
    parser.add_argument("--clean", action="store_true",
                        help="Train-Test 중복 제거 데이터 정제 실행")
    args = parser.parse_args()

    model = build_model(stacking_version=args.stacking, ga_version=args.ga, env=args.env)
    model.run(clean=args.clean)


if __name__ == "__main__":
    main()
