from tico.quantization.recipes.stages.base import Stage
from tico.quantization.recipes.stages.ptq import PTQStage

_STAGE_REGISTRY = {
    "ptq": PTQStage(),
}


def get_stage(name: str) -> Stage:
    key = name.lower()
    if key not in _STAGE_REGISTRY:
        raise KeyError(
            f"Unknown quantization stage: {name}. available={sorted(_STAGE_REGISTRY)}"
        )
    return _STAGE_REGISTRY[key]
