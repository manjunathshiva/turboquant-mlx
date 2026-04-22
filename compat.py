"""Compatibility shims for upstream libraries.

Importing this module is a side-effect: it patches third-party classes in
place to work around gaps between bleeding-edge model releases and the
installed library versions. All patches are idempotent and self-disable
once upstream catches up.
"""


def _patch_nemotron_h_pattern():
    """Teach NemotronHConfig about MLP ("-") block types.

    Nemotron 3 models encode layer types as a string like
    "M-M-M-MM-M-M*-M-M*-..." where M=mamba, *=attention, -=MLP, E=MoE.
    transformers' NemotronHConfig hard-codes its pattern alphabet in two
    places — `_pattern_to_list` (decodes the string) and
    `validate_layers_block_type` (checks the resulting list). Both miss
    "mlp". mlx-lm already handles the "-" block type, so extending the
    config's alphabet is enough to unblock loading.
    """
    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (
            NemotronHConfig,
        )
    except ImportError:
        return

    try:
        NemotronHConfig._pattern_to_list("-")
    except KeyError:
        @staticmethod
        def _pattern_to_list(pattern: str) -> list:
            mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
            return [mapping[c] for c in pattern]
        NemotronHConfig._pattern_to_list = _pattern_to_list

    valid_types = {"mamba", "attention", "moe", "mlp"}

    @staticmethod
    def validate_layers_block_type(self):
        if not isinstance(self.layers_block_type, list):
            raise ValueError(
                f"`layers_block_type` must be a list of strings. Got type: "
                f"{type(self.layers_block_type)}"
            )
        invalid = set(self.layers_block_type) - valid_types
        if invalid:
            raise ValueError(
                f"`layers_block_type` contains invalid types: {invalid}. "
                f"Must be one of: {valid_types}"
            )
        if getattr(self, "num_nextn_predict_layers", 0) > 0:
            if self.mtp_layers_block_type is None:
                raise ValueError(
                    "mtp_layers_block_type is required when "
                    "num_nextn_predict_layers > 0."
                )
            if not isinstance(self.mtp_layers_block_type, list):
                raise ValueError(
                    f"`mtp_layers_block_type` must be a list of strings. "
                    f"Got type: {type(self.mtp_layers_block_type)}"
                )
            invalid = set(self.mtp_layers_block_type) - valid_types
            if invalid:
                raise ValueError(
                    f"`mtp_layers_block_type` contains invalid types: "
                    f"{invalid}. Must be one of: {valid_types}"
                )

    NemotronHConfig.validate_layers_block_type = validate_layers_block_type

    # huggingface_hub's dataclass machinery captures validator functions into
    # __class_validators__ at class-definition time, so swapping the attribute
    # above is not enough — we also have to replace the entry in that list.
    validators = getattr(NemotronHConfig, "__class_validators__", None)
    if validators is not None:
        for i, fn in enumerate(validators):
            if getattr(fn, "__name__", "") == "validate_layers_block_type":
                # validate_layers_block_type is defined as @staticmethod, so
                # __class_validators__ stores the underlying function; use the
                # same shape here.
                validators[i] = validate_layers_block_type.__func__
                break


_patch_nemotron_h_pattern()
