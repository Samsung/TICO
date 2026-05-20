from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter

_ADAPTERS = {
    "llama": LlamaAdapter(),
}


def get_adapter(family: str) -> ModelAdapter:
    key = family.lower()
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown model family: {family}. available={sorted(_ADAPTERS)}")
    return _ADAPTERS[key]
