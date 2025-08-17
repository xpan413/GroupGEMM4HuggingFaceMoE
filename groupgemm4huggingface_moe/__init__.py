import importlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groupgemm4huggingface_moe.hf.auto_model import (
        AutoGroupGEMMForCausalLM,
    )  # noqa: F401
    from groupgemm4huggingface_moe.hf.monkey_patch import (
        _apply_groupgemm,
    )  # noqa: F401


def __getattr__(name: str):
    """
    Handles lazy access to transformer-dependent attributes.
    If 'transformers' is not installed, raises a user-friendly ImportError.
    """

    if name == "AutoGroupGEMMForCausalLM":
        module = importlib.import_module("groupgemm4huggingface_moe.hf.auto_model")
        return getattr(module, name)

    monkey_patch_symbols = {
        "_apply_groupgemm",
    }

    if name in monkey_patch_symbols:
        module = importlib.import_module("groupgemm4huggingface_moe.hf.monkey_patch")
        return getattr(module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")
