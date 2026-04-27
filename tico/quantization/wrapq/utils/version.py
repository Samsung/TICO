from tico.utils.version import package_version_is_at_least

MIN_VERSIONS = {
    "llama": "4.36.0",  # transformers==4.31.0 supports llama but without layer_idx and position_embeddings feature
    "qwen3-vl": "4.57.0",
}


def has_transformers_for(model_type: str) -> bool:
    required_version = MIN_VERSIONS.get(model_type)
    if not required_version:
        raise ValueError(f"Invalid model_type {model_type}")

    return package_version_is_at_least(
        package_name="transformers", required_version=required_version
    )
