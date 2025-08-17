import inspect

from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from .monkey_patch import _apply_groupgemm

__all__ = [
    "AutoGroupGEMMForCausalLM",
]


class AutoGroupGEMMForCausalLM(AutoModelForCausalLM):
    r"""
    This class serves as a drop-in replacement for AutoModelForCausalLM, which applies the groupgemm
    optimization to the model if it is applicable.
    """

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        r"""Load model configuration and apply groupgemm optimization.

        Args:
            config (PretrainedConfig): The model configuration.

        Returns:
            AutoGroupGEMMForCausalLM: The initialized model.
        """
        print(config.model_type)
        _apply_groupgemm(config.model_type)
        return super().from_config(config)
