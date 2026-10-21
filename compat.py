"""Compatibility shims for upstream libraries.

Importing this module is a side-effect: it patches third-party classes in
place to work around gaps between bleeding-edge model releases and the
installed library versions. All patches are idempotent and self-disable
once upstream catches up.
"""

import os


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

    # transformers >= 5.20 decodes "-" itself and has no such validator; there
    # is nothing to replace.
    validators = getattr(NemotronHConfig, "__class_validators__", None) or []
    if not hasattr(NemotronHConfig, "validate_layers_block_type") and not any(
        getattr(fn, "__name__", "") == "validate_layers_block_type"
        for fn in validators
    ):
        return

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


#: ``model_type`` -> our module under ``turboquant_mlx.models``. TurboQuant
#: checkpoints of these types (and the sources we convert them from) use OUR
#: module's parameter layout, so :func:`get_model_classes` always picks ours.
#: mlx-lm 0.32 ships its own ``laguna`` and ``kimi_k3`` with different names
#: (K3 fuses q/k/v into ``qkv_proj``; Laguna's experts and gate are keyed
#: differently), so deferring to them would mis-load every published build.
#: For K3 the key is the *wrapper* type its config declares at top level, not
#: the ``kimi_linear`` text tower inside ``text_config`` that mlx-lm also knows.
LOCAL_PORTS = {
    "laguna": "laguna",
    "kimi_k3": "kimi_k3",
    "sarvam_moe": "sarvam_moe",
    # Qwen3.8-Flash-Next. Vendored from mlx-lm PR #1788.
    "qwen4_exp": "qwen4_exp",
}


def get_model_classes(config: dict):
    """mlx-lm's ``_get_classes``, except our own ports win for their types.

    Pass this as ``get_model_classes`` to ``mlx_lm.utils.load_model``, or call it
    in place of ``_get_classes``. It changes nothing for plain mlx-lm users: the
    choice is made only where TurboQuant builds or converts a model.
    """
    import importlib

    module_name = LOCAL_PORTS.get(config.get("model_type"))
    if module_name is not None:
        arch = importlib.import_module(f"turboquant_mlx.models.{module_name}")
        return arch.Model, arch.ModelArgs
    from mlx_lm.utils import _get_classes

    return _get_classes(config)


def load_source(path_or_hf_repo, tokenizer_config=None, lazy=False,
                trust_remote_code=False):
    """``mlx_lm.utils.load`` for conversion sources, built via our ports.

    Same steps as upstream's ``load`` (download, ``load_model``,
    ``load_tokenizer``) with :func:`get_model_classes` as the class resolver.
    Always returns ``(model, tokenizer, config)``: every converter needs the
    config. ``trust_remote_code`` is mlx-lm 0.32's gate on a checkpoint's own
    model code (``model_file``); the tokenizer's trust travels in
    ``tokenizer_config``.
    """
    from mlx_lm.utils import _download, load_model, load_tokenizer

    model_path = _download(path_or_hf_repo)
    model, config = load_model(
        model_path, lazy, get_model_classes=get_model_classes,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = load_tokenizer(
        model_path, tokenizer_config,
        eos_token_ids=config.get("eos_token_id", None),
    )
    return model, tokenizer, config


def _register_local_model(model_type: str, module_name: str = None):
    """Let plain mlx-lm find one of our ports that upstream lacks.

    mlx-lm dispatches a checkpoint's ``model_type`` via
    ``importlib.import_module(f"mlx_lm.models.{model_type}")``, which consults
    ``sys.modules`` first, so aliasing our module there makes ``_get_classes``
    (and mlx-lm internals such as ``_tokenizer_defaults``) resolve it.
    Idempotent, and self-disables per architecture once mlx-lm ships its own
    module — never overriding upstream globally, so plain mlx-lm users keep
    upstream's. TurboQuant's own loaders don't depend on this alias: they go
    through :func:`get_model_classes`.
    """
    import importlib
    import sys

    target = f"mlx_lm.models.{model_type}"
    if target in sys.modules:
        return
    try:
        importlib.import_module(target)  # native support exists; defer to it
        return
    except ImportError:
        pass

    sys.modules[target] = importlib.import_module(
        f"turboquant_mlx.models.{module_name or model_type}"
    )


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
    # mlx-lm 0.32 maps the compressed-tensors format itself, to the same mode
    # this patch injects; leave load_model alone there.
    native = getattr(_utils, "_compressed_tensors_quantization", None)
    if native is not None:
        try:
            if native({"format": "mxfp4-pack-quantized"}).get("mode") == "mxfp4":
                return
        except Exception:  # an unexpected upstream shape: keep the patch
            pass

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


def is_local_kimi_k3(model_path) -> bool:
    """True iff ``model_path`` is a local directory whose ``config.json``
    declares ``model_type: "kimi_k3"``. Only meaningful on a path the user
    typed: a Hub repo id's cache snapshot is a local directory too."""
    import json
    from pathlib import Path

    p = Path(model_path)
    if not p.is_dir():
        return False
    try:
        with open(p / "config.json") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    return cfg.get("model_type") == "kimi_k3"


# realpaths of the local K3 directories the user named on a command line
_TRUSTED_K3_DIRS = set()


def trust_local_kimi_k3(path_arg) -> bool:
    """Opt the user's own model argument into K3 tokenizer trust.

    Call with the argument exactly as the user gave it (``--model``,
    ``--hf-path``), before anything resolves it. Only an existing local
    directory whose ``config.json`` says ``kimi_k3`` is registered, by
    realpath; returns whether it was. A Hub repo id is never registered, so
    the cache snapshot it later resolves to never auto-trusts its own code.
    """
    if not is_local_kimi_k3(path_arg):
        return False
    _TRUSTED_K3_DIRS.add(os.path.realpath(path_arg))
    return True


def is_trusted_kimi_k3(model_path) -> bool:
    """True iff ``model_path`` is a directory registered by
    :func:`trust_local_kimi_k3`."""
    return os.path.realpath(model_path) in _TRUSTED_K3_DIRS


def _patch_kimi_k3_tokenizer_trust():
    """Opt registered local Kimi K3 directories into their own tokenizer code.

    K3 repos ship a custom tokenizer class (``tokenization_kimi.py``) that
    requires ``trust_remote_code=True`` to load — without it transformers
    either raises or blocks on an interactive prompt, which kills
    ``turboquant-generate``/``turboquant-serve`` on any converted K3 model.

    The flag is injected only for a directory the user named on the command
    line (see :func:`trust_local_kimi_k3`). The tokenizer loader only ever
    sees a path, and a Hub repo id or a model named in a server request has
    already been resolved to a local snapshot by then, so the decision cannot
    be made here. An explicit ``trust_remote_code`` from the caller always
    wins; ``None`` (mlx-lm's server without ``--trust-remote-code``) counts
    as unset.
    """
    import mlx_lm.tokenizer_utils as _tok

    if getattr(_tok, "_tq_local_trust_patched", False):
        return

    orig = _tok.load

    # **kwargs: mlx-lm 0.32 added `defaults=` (the model module's
    # TOKENIZER_DEFAULTS: eos ids, chat-template and tool-parser types — never
    # trust_remote_code). Forward whatever else upstream passes, unchanged.
    def load(model_path, tokenizer_config_extra=None, eos_token_ids=None,
             **kwargs):
        extra = dict(tokenizer_config_extra or {})
        if (extra.get("trust_remote_code") is None
                and is_trusted_kimi_k3(model_path)):
            extra["trust_remote_code"] = True
        return orig(model_path, extra, eos_token_ids=eos_token_ids, **kwargs)

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
for _model_type, _module_name in LOCAL_PORTS.items():
    _register_local_model(_model_type, _module_name)
_patch_compressed_tensors_mxfp4()
_patch_kimi_k3_tokenizer_trust()
_patch_glm47_tool_name_strip()
