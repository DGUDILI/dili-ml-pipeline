from typing import Optional
from models.stackdili_fixed.model import Model


def _load_ga(version: str):
    """요청된 GA 버전만 import."""
    if version == "g0":
        from models.stackdili_fixed.ga.ga_v0 import GAv0
        return GAv0
    raise KeyError(f"GA 버전 '{version}'이 존재하지 않습니다. 가능한 버전: {list(GA_REGISTRY)}")


def _load_stacking(version: str):
    """요청된 Stacking 버전만 import."""
    if version == "s0":
        from models.stackdili_fixed.stacking.stacking_v0 import StackingV0
        return StackingV0
    if version == "s0.5":
        from models.stackdili_fixed.stacking.stacking_v0_5 import StackingV05
        return StackingV05
    if version == "s1":
        from models.stackdili_fixed.stacking.stacking_v1 import StackingV1
        return StackingV1
    raise KeyError(f"Stacking 버전 '{version}'이 존재하지 않습니다. 가능한 버전: {list(STACKING_REGISTRY)}")


# train.py의 choices= 에 사용하기 위한 키 목록 (import 없이 반환)
GA_REGISTRY = {
    "g0": None,
}

ENV_REGISTRY = {
    "env1": "외부 검증 - NCTR/Greene/Xu/Liew 학습, DILIrank 테스트",
    "env2": "10-Fold CV - 전체 데이터셋 합쳐서 10겹 교차 검증",
}

STACKING_REGISTRY = {
    "s0":   None,
    "s0.5": None,
    "s1":   None,
}


def build_model(
    stacking_version: str,
    ga_version: Optional[str] = None,
    env: Optional[str] = None,
) -> Model:
    ga       = _load_ga(ga_version)() if ga_version else None
    stacking = _load_stacking(stacking_version)()
    return Model(
        stacking=stacking, ga=ga,
        stacking_version=stacking_version, ga_version=ga_version,
        env=env,
    )
