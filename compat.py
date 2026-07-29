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


def _register_kimi_k3_model():
    """Register the MLX Kimi K3 model so mlx-lm's ``_get_classes`` can find it.

    Same mechanism as :func:`_register_laguna_model`: K3 checkpoints carry
    ``model_type: "kimi_k3"`` at the top level (the text tower inside
    ``text_config`` is ``kimi_linear``, which mlx-lm knows, but the wrapper is
    what ``_get_classes`` dispatches on). Idempotent; self-disables the moment
    mlx-lm ships native support.
    """
    import sys

    if "mlx_lm.models.kimi_k3" in sys.modules:
        return
    try:
        import mlx_lm.models.kimi_k3  # noqa: F401 — native support exists; defer
        return
    except ImportError:
        pass

    from turboquant_mlx.models import kimi_k3 as _kimi_k3

    sys.modules["mlx_lm.models.kimi_k3"] = _kimi_k3


def _patch_compressed_tensors_mxfp4():
    """Fix mlx-lm's quantization mode for compressed-tensors mxfp4 checkpoints.

    ``mlx_lm.utils.load_model`` maps ``quant_method == "compressed-tensors"`` to
    ``{"group_size": 32, "bits": 4, "mode": "affine"}`` without consulting
    ``quantization_config["format"]``. For ``mxfp4-pack-quantized`` checkpoints
    (Kimi K3's routed experts) the packed nibbles are fp4-e2m1 codes with E8M0
    uint8 scales — decoding them as affine int4 produces garbage. MLX decodes
    them bit-exactly with ``mode="mxfp4"`` (verified against compressed-tensors
    0.17.1), so rewrite the mode before the quantize wrapper runs.

    Self-disables once upstream branches on the format itself.
    """
    import inspect

    import mlx_lm.utils as _utils

    if getattr(_utils, "_tq_mxfp4_patched", False):
        return

    orig = _utils.load_model
    sig = inspect.signature(orig)

    def load_model(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            config = _utils.load_config(bound.arguments["model_path"])
        except Exception:
            config = {}
        qc = config.get("quantization_config") or {}
        if not qc:
            qc = (config.get("text_config") or {}).get("quantization_config") or {}
        if (
            qc.get("quant_method") == "compressed-tensors"
            and qc.get("format") == "mxfp4-pack-quantized"
        ):
            # A "quantization" key in the merged config takes priority over the
            # legacy quantization_config branch, so injecting it via
            # model_config routes the load through mode="mxfp4".
            mc = dict(bound.arguments.get("model_config") or {})
            mc["quantization"] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            bound.arguments["model_config"] = mc
        return orig(*bound.args, **bound.kwargs)

    load_model.__signature__ = sig
    _utils.load_model = load_model
    _utils._tq_mxfp4_patched = True


def _patch_local_tokenizer_trust():
    """Opt local-directory models into their own tokenizer code.

    Kimi K3 (and other repos with a custom tokenizer class, e.g.
    ``tokenization_kimi.py``) requires ``trust_remote_code=True`` to load —
    without it transformers either raises or blocks on an interactive prompt,
    which kills ``turboquant-generate``/``turboquant-serve`` on any converted
    K3 model. For a *local directory* the "remote" code is already sitting on
    the user's disk next to the weights they chose to run, so defaulting the
    flag on is the same trust decision they already made. Hub repo ids are
    left untouched.
    """
    from pathlib import Path

    import mlx_lm.tokenizer_utils as _tok

    if getattr(_tok, "_tq_local_trust_patched", False):
        return

    orig = _tok.load

    def load(model_path, tokenizer_config_extra=None, eos_token_ids=None):
        extra = dict(tokenizer_config_extra or {})
        if "trust_remote_code" not in extra and Path(model_path).is_dir():
            extra["trust_remote_code"] = True
        return orig(model_path, extra, eos_token_ids=eos_token_ids)

    _tok.load = load
    _tok._tq_local_trust_patched = True
    # mlx_lm.utils imports it by value as _load_tokenizer; re-point that too.
    import mlx_lm.utils as _utils

    if getattr(_utils, "_load_tokenizer", None) is orig:
        _utils._load_tokenizer = load


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
_register_kimi_k3_model()
_patch_compressed_tensors_mxfp4()
_patch_local_tokenizer_trust()
_patch_glm47_tool_name_strip()
