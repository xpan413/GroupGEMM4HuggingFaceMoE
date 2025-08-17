import logging

logger = logging.getLogger(__name__)


def apply_groupgemm_to_qwen3_moe() -> None:
    r"""Apply GroupGEMM to replace original implementation in HuggingFace Qwen3 models."""

    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from .qwen3_moe import GroupGEMMQwen3MoeSparseMoeBlock

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = GroupGEMMQwen3MoeSparseMoeBlock
    logger.info("Applied GroupGEMM to Qwen3 Moe Sparse Moe Block.")


MODEL_TYPE_TO_APPLY_GROUPGEMM = {
    "qwen3_moe": apply_groupgemm_to_qwen3_moe,
}


def _apply_groupgemm(model_type: str) -> None:
    """Apply groupgemm optimization for supported model types.

    Args:
        model_type (str): The type of the model.
    """
    print(model_type)
    if model_type not in MODEL_TYPE_TO_APPLY_GROUPGEMM.keys():
        logger.info(
            f"There are currently no groupgemm supported for model type: {model_type}."
        )
        return

    apply_fn = MODEL_TYPE_TO_APPLY_GROUPGEMM[model_type]
    apply_fn()
