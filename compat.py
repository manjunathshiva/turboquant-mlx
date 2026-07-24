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


def _register_laguna_model():
    """Register the MLX Laguna model so mlx-lm's ``_get_classes`` can find it.

    mlx-lm dispatches a checkpoint's ``model_type`` via
    ``importlib.import_module(f"mlx_lm.models.{model_type}")``. Poolside's Laguna
    architecture has no upstream mlx-lm module yet, so we alias our
    implementation into ``sys.modules`` under that name — ``importlib`` consults
    ``sys.modules`` first, so this makes ``load_turboquant`` (and plain mlx-lm)
    resolve ``laguna`` to our ``Model`` / ``ModelArgs``. Idempotent, and
    self-disables the moment mlx-lm ships native Laguna support.
    """
    import sys

    if "mlx_lm.models.laguna" in sys.modules:
        return
    try:
        import mlx_lm.models.laguna  # noqa: F401 — native support exists; defer
        return
    except ImportError:
        pass

    from turboquant_mlx.models import laguna as _laguna

    sys.modules["mlx_lm.models.laguna"] = _laguna


def _patch_glm47_tool_name_strip():
    """Strip whitespace off GLM-style parsed tool-call function names.

    mlx-lm's glm47 parser (also selected for Laguna, whose template is
    ``laguna_glm_thinking_v5``) pulls the name with
    ``re.compile(r"^(.*?)<arg_key>", re.DOTALL)`` and does not strip it. The
    wire format puts the name on its own line::

        <tool_call>list_dir
        <arg_key>path</arg_key><arg_value>/tmp</arg_value>
        </tool_call>

    so the name comes back as ``"list_dir\\n"``. Two things break: agent
    harnesses fail to match the tool by name, and the untrimmed name misses in
    ``_get_string_arg_names``, so string arguments get run through
    ``_deserialize`` and a value like ``"2024"`` silently becomes an int.

    Self-disables once upstream strips the name itself.
    """
    try:
        from mlx_lm.tool_parsers import glm47
    except ImportError:
        return

    if getattr(glm47, "_tq_name_strip_patched", False):
        return

    orig = glm47.parse_tool_call

    def parse_tool_call(text, tools=None):
        result = orig(text, tools)
        name = result.get("name")
        if isinstance(name, str) and name != name.strip():
            stripped = name.strip()
            # re-resolve arguments with the correct name so string args that
            # were wrongly deserialized come back as strings
            result = orig(text.replace(name, stripped, 1), tools)
            result["name"] = stripped
        return result

    glm47.parse_tool_call = parse_tool_call
    glm47._tq_name_strip_patched = True


_patch_nemotron_h_pattern()
_register_laguna_model()
_patch_glm47_tool_name_strip()
