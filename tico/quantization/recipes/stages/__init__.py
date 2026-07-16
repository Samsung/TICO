from tico.quantization.recipes.stages.base import Stage
from tico.quantization.recipes.stages.cle import CLEStage
from tico.quantization.recipes.stages.gptq import GPTQStage
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.recipes.stages.smoothquant import SmoothQuantStage
from tico.quantization.recipes.stages.spinquant import SpinQuantStage
from tico.quantization.recipes.stages.weight_sparsity import WeightSparsityStage

_STAGE_REGISTRY = {
    "gptq": GPTQStage(),
    "ptq": PTQStage(),
    "spinquant": SpinQuantStage(),
    "cle": CLEStage(),
    "smoothquant": SmoothQuantStage(),
    "weight_sparsity": WeightSparsityStage(),
}


def get_stage(name: str) -> Stage:
    key = name.lower()
    if key not in _STAGE_REGISTRY:
        raise KeyError(
            f"Unknown quantization stage: {name}. available={sorted(_STAGE_REGISTRY)}"
        )
    return _STAGE_REGISTRY[key]
