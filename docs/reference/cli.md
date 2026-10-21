# Command line

Every command's own `--help`, captured from `turboquant-mlx-full` 0.29.0. The guides explain the options in context.

## `turboquant-plan`

Will a model fit this Mac (or another one)? Reads only the safetensors headers. See [Get started](../getting-started.md#1-check-that-it-fits-before-downloading).

```text
usage: turboquant-plan [-h] --model MODEL [--context CONTEXT] [--kv-bits KV_BITS]
                       [--prefill-step-size PREFILL_STEP_SIZE] [--wired-gb WIRED_GB]
                       [--ram-gb RAM_GB] [--ngram-offload] [--json]

Will this model run on this Mac, and with what flags? Reads only safetensors headers — no tensors
are loaded.

options:
  -h, --help            show this help message and exit
  --model MODEL         local dir or HF repo id
  --context CONTEXT     context length to plan for (default 16384)
  --kv-bits KV_BITS     plan with TurboQuant KV quantization
  --prefill-step-size PREFILL_STEP_SIZE
                        force a chunk size instead of picking the largest safe one
  --wired-gb WIRED_GB   assume this Metal working set (plan for another Mac)
  --ram-gb RAM_GB       assume this much system RAM, as the machine is sold (--ram-gb 16 = a 16 GB
                        Mac = 17.18e9 bytes)
  --ngram-offload       project with the n-gram table served from disk (Qwen3.8-Flash-Next), as
                        the loaders' --ngram-offload does
  --json                machine-readable output
```

## `turboquant-doctor`

The same projection plus a read-only readiness check, with stable exit codes (0 ok, 1 won't fit or files missing, 2 bad usage).

```text
usage: turboquant-doctor [-h] --model MODEL [--context CONTEXT] [--kv-bits KV_BITS]
                         [--prefill-step-size PREFILL_STEP_SIZE] [--wired-gb WIRED_GB]
                         [--ram-gb RAM_GB] [--ngram-offload] [--json]

Read-only readiness check: model files, environment, and whether the projected placement is
runnable.

options:
  -h, --help            show this help message and exit
  --model MODEL         local dir or HF repo id
  --context CONTEXT     context length to plan for (default 16384)
  --kv-bits KV_BITS     plan with TurboQuant KV quantization
  --prefill-step-size PREFILL_STEP_SIZE
                        force a chunk size instead of picking the largest safe one
  --wired-gb WIRED_GB   assume this Metal working set (plan for another Mac)
  --ram-gb RAM_GB       assume this much system RAM, as the machine is sold (--ram-gb 16 = a 16 GB
                        Mac = 17.18e9 bytes)
  --ngram-offload       project with the n-gram table served from disk (Qwen3.8-Flash-Next), as
                        the loaders' --ngram-offload does
  --json                machine-readable output
```

## `turboquant-convert`

Convert a Hugging Face model (BF16/FP16) to TurboQuant. Also `python -m turboquant_mlx.convert`. See [Convert a model](../guides/convert.md).

```text
usage: turboquant-convert [-h] --hf-path HF_PATH [--trust-remote-code] [--mlx-path MLX_PATH]
                          [--bits {2,3,4}] [--group-size {32,64,128}]
                          [--rotation {hadamard,blockwise_hadamard,none}]
                          [--rotation-seed ROTATION_SEED] [--use-qjl]
                          [--dtype {float16,bfloat16,float32}] [--attn-bits {2,3,4}]
                          [--mlp-bits {2,3,4}] [--mlp-group-size {16,32,64,128}]
                          [--ternary-experts] [--expert-down-bits {2,3,4}] [--streaming]
                          [--protect-expert-layers PROTECT_EXPERT_LAYERS]
                          [--expert-layer-tiers JSON] [--protect-bits {2,3,4}] [--quantize-extras]
                          [--extras-bits {2,3,4,8}] [--extras-group-size {32,64,128}] [--keep-mtp]

Convert HuggingFace model to TurboQuant-compressed MLX format

options:
  -h, --help            show this help message and exit
  --hf-path HF_PATH, --model HF_PATH
                        HuggingFace model path or local path
  --trust-remote-code   Allow the source's own code to run: its tokenizer code (tokenization_*.py)
                        and any model code its config names (model_file). Needed for Kimi K3 by
                        Hub repo id; a local K3 directory's tokenizer is trusted without it.
  --mlx-path MLX_PATH   Output directory for MLX model (default: mlx_model)
  --bits {2,3,4}, -b {2,3,4}
                        Quantization bit-width (default: 3)
  --group-size {32,64,128}, -g {32,64,128}
                        Quantization group size (default: 64)
  --rotation {hadamard,blockwise_hadamard,none}
                        Rotation method (default: hadamard)
  --rotation-seed ROTATION_SEED
                        Random seed for rotation signs (default: 42)
  --use-qjl             Enable QJL 1-bit residual correction (adds ~1 bit overhead)
  --dtype {float16,bfloat16,float32}
                        Model dtype before quantization
  --attn-bits {2,3,4}   Override bits for attention-block linears (q/k/v/o_proj). Defaults to
                        --bits when omitted.
  --mlp-bits {2,3,4}    Override bits for MLP and MoE expert linears. Defaults to --bits when
                        omitted.
  --mlp-group-size {16,32,64,128}
                        Override group size (block-scale granularity) for MLP/MoE expert linears,
                        so a sub-2-bit expert tier can carry a finer scale than attention.
                        Defaults to --group-size when omitted.
  --ternary-experts     Quantize MoE expert weights to the ternary {-c,0,+c} codebook (1.58-bit),
                        stored in the 2-bit slot. The zero level clears the 1-bit cardinality
                        wall; data-free sub-2-bit expert tier (tq2a-tqTe). Attention stays at
                        --attn-bits/--bits.
  --expert-down-bits {2,3,4}
                        Asymmetric expert precision: quantize MoE expert down projections at this
                        Gaussian-codebook width while up/gate take the --mlp-bits / --ternary-
                        experts tier. The down projection is the SwiGLU summation bottleneck;
                        llama.cpp-family 2-bit mixes keep it above up/gate for exactly this
                        reason.
  --streaming           Memory-bounded conversion: write each quantized layer to a shard and free
                        it, so the full quantized model never resides in RAM. Use for models too
                        big to convert in memory (e.g. 200B+ on 64 GB). (--dtype override is not
                        supported in this mode.)
  --protect-expert-layers PROTECT_EXPERT_LAYERS
                        Layer indices whose routed experts use a --protect-bits Gaussian codebook
                        instead of the expert tier (ternary or --mlp-bits). Comma list with
                        inclusive ranges, e.g. '0-5,42-47'. Changes bit width only, never the
                        expert group size, so the loader needs no matching rule. Warns if a listed
                        layer matches nothing.
  --expert-layer-tiers JSON
                        Per-layer expert tiers: a JSON file mapping MoE layer index to 'ternary',
                        '2', '3' or '4' (or {"tiers": {...}}). Layers not listed keep the expert
                        tier. Exclusive with --protect-expert-layers and --expert-down-bits.
  --protect-bits {2,3,4}
                        Codebook width for --protect-expert-layers (default 3).
  --quantize-extras     Also quantize the bf16 remainder the polar path never touches --
                        nn.Embedding above all -- to MLX affine. Off by default because on a dense
                        model that remainder is just a token embedding and an lm_head. REQUIRED on
                        models with a large lookup table: Qwen3.8-Flash-Next's sharded n-gram/PLE
                        table is 51.2B params (95.4 GiB bf16, 28% of the model), so without this a
                        ternary-expert build is ~124 GiB instead of ~54 GiB. Routers and the QSA
                        block indexer stay full precision either way.
  --extras-bits {2,3,4,8}
                        Bit-width for --quantize-extras (default 4).
  --extras-group-size {32,64,128}
                        Group size for --quantize-extras (default 64).
  --keep-mtp            Copy the source model's multi-token-prediction head into the output, for
                        self-speculative decoding. mlx-lm discards these tensors in sanitize(), so
                        without this flag the head is lost. Opt-in because it adds the head at its
                        source dtype (810 MiB on Qwen3.8-27B) and nothing consumes it yet. No-op
                        if the source has no MTP head.
```

## `turboquant-generate`

Generate text with a TurboQuant model. See [Generate text](../guides/generate.md).

```text
usage: turboquant-generate [-h] --model MODEL [--prompt PROMPT] [--max-tokens MAX_TOKENS]
                           [--temp TEMP] [--top-p TOP_P] [--top-k TOP_K]
                           [--rep-penalty REP_PENALTY] [--no-think] [--multi-think] [--stop TOKEN]
                           [--rep-ctx REP_CTX] [--fast] [--ngram-offload] [--trust-remote-code]
                           [--min-tokens MIN_TOKENS] [--kv-bits KV_BITS] [--kv-k-bits KV_K_BITS]
                           [--kv-v-bits KV_V_BITS] [--kv-min-tokens KV_MIN_TOKENS]
                           [--kv-group-size KV_GROUP_SIZE]

Generate text using a TurboQuant-compressed model

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to TurboQuant model directory
  --prompt PROMPT       Input prompt
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 200)
  --temp TEMP           Sampling temperature (default: 0.7)
  --top-p TOP_P         Nucleus sampling threshold. Defaults to the model's generation_config.json
                        value when present, else disabled. Pass 0 to force-disable.
  --top-k TOP_K         Top-k truncation. Defaults to the model's generation_config.json value
                        when present, else disabled. Pass 0 to force-disable. Without truncation,
                        rare tokens like a stray '</think>' can be sampled in place of EOS.
  --rep-penalty REP_PENALTY
                        Repetition penalty (e.g. 1.05). Defaults to the model's
                        generation_config.json value when present, else disabled. Pass 0 or 1 to
                        force-disable. Recommended for thinking-mode models on low-bit builds
                        (breaks think-block loops); omit for numeric/math prompts.
  --no-think            Disable thinking mode via the chat template (enable_thinking=False). Much
                        faster and immune to think-block loops on thinking-capable models.
  --multi-think         Allow more than one </think> token per generation. By default a second
                        </think> is masked once one has been emitted, which prevents low-bit
                        thinking models from re-emitting their answer.
  --stop TOKEN          Extra stop token to terminate generation on, as a token string
                        ('</assistant>') or a token id ('24'). Repeatable. The model's
                        generation_config.json terminators are applied automatically; use this
                        only for one-off overrides. Must be a single token.
  --rep-ctx REP_CTX     Repetition penalty context window in tokens (default: 256). Only used when
                        --rep-penalty is set.
  --fast                Fast mode: skip QJL correction for ~25% faster decode (slightly lower
                        quality)
  --ngram-offload       Serve an n-gram embedding table (Qwen3.8-Flash-Next) from memory-mapped
                        files on the CPU instead of GPU memory: ~17.9 GiB less resident on the
                        2-bit build, bit-identical output. Keep the model directory in place while
                        the model is loaded. No effect on models without such a table.
  --trust-remote-code   Allow the model's own tokenizer code (tokenization_*.py) to run. Needed
                        for Kimi K3 loaded by Hub repo id; a local K3 directory is trusted without
                        it.
  --min-tokens MIN_TOKENS
                        Mask EOS until at least this many tokens are generated. Useful for
                        thinking-mode models (Nemotron 3, etc.) whose chat template primes EOS as
                        the top-1 logit (default: 0)
  --kv-bits KV_BITS     Symmetric KV-cache bits (K=V). Use --kv-k-bits / --kv-v-bits for mixed
                        precision instead. Omit to keep fp16 cache.
  --kv-k-bits KV_K_BITS
                        Bits for K cache (recommended: 8). Pair with --kv-v-bits.
  --kv-v-bits KV_V_BITS
                        Bits for V cache (recommended: 3). Pair with --kv-k-bits. Mixed K8/V3 is
                        the v0.2 recommended default.
  --kv-min-tokens KV_MIN_TOKENS
                        Keep the first N cached tokens in fp16 (attention-sink protection). 128 is
                        a reasonable default for long-context generation. Default: 0 (quantize
                        from token 0).
  --kv-group-size KV_GROUP_SIZE
                        Hadamard rotation group size for KV quantization (default: 64).
```

## `turboquant-serve`

OpenAI-compatible server (mlx_lm.server with a TurboQuant loader). **Most of TurboQuant's own options are not shown by `--help`** (`--disk-cache`, `--cache-budget-gb`, `--kv-k-bits`/`--kv-v-bits`/`--kv-min-tokens`, `--tool-syntax-greedy`, …). `--kv-bits` and `--kv-group-size` do appear, with mlx-lm's descriptions, but TurboQuant handles them. All are listed in [Serve an OpenAI-compatible API](../guides/serve.md#options).

```text
TurboQuant-MLX serve  ·  OpenAI-compatible HTTP server (backend: mlx_lm.server, TurboQuant-aware loader)
usage: turboquant-serve [-h] [--model MODEL] [--adapter-path ADAPTER_PATH] [--host HOST]
                        [--port PORT] [--allowed-origins ALLOWED_ORIGINS]
                        [--draft-model DRAFT_MODEL] [--num-draft-tokens NUM_DRAFT_TOKENS]
                        [--trust-remote-code] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        [--chat-template CHAT_TEMPLATE] [--use-default-chat-template]
                        [--temp TEMP] [--top-p TOP_P] [--top-k TOP_K] [--min-p MIN_P]
                        [--max-tokens MAX_TOKENS] [--chat-template-args CHAT_TEMPLATE_ARGS]
                        [--decode-concurrency DECODE_CONCURRENCY]
                        [--prompt-concurrency PROMPT_CONCURRENCY]
                        [--prefill-step-size PREFILL_STEP_SIZE]
                        [--prompt-cache-size PROMPT_CACHE_SIZE]
                        [--prompt-cache-bytes PROMPT_CACHE_BYTES] [--kv-bits KV_BITS]
                        [--kv-group-size KV_GROUP_SIZE] [--quantized-kv-start QUANTIZED_KV_START]
                        [--pipeline]

MLX Http Server.

options:
  -h, --help            show this help message and exit
  --model MODEL         The path to the MLX model weights, tokenizer, and config
  --adapter-path ADAPTER_PATH
                        Optional path for the trained adapter weights and config.
  --host HOST           Host for the HTTP server (default: 127.0.0.1)
  --port PORT           Port for the HTTP server (default: 8080)
  --allowed-origins ALLOWED_ORIGINS
                        Allowed origins (default: *)
  --draft-model DRAFT_MODEL
                        A model to be used for speculative decoding.
  --num-draft-tokens NUM_DRAFT_TOKENS
                        Number of tokens to draft when using speculative decoding.
  --trust-remote-code   Enable trusting remote code for tokenizer
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
  --chat-template CHAT_TEMPLATE
                        Specify a chat template for the tokenizer
  --use-default-chat-template
                        Use the default chat template
  --temp TEMP           Default sampling temperature (default: 0.0)
  --top-p TOP_P         Default nucleus sampling top-p (default: 1.0)
  --top-k TOP_K         Default top-k sampling (default: 0, disables top-k)
  --min-p MIN_P         Default min-p sampling (default: 0.0, disables min-p)
  --max-tokens MAX_TOKENS
                        Default maximum number of tokens to generate (default: 512)
  --chat-template-args CHAT_TEMPLATE_ARGS
                        A JSON formatted string of arguments for the tokenizer's
                        apply_chat_template, e.g. '{"enable_thinking":false}'
  --decode-concurrency DECODE_CONCURRENCY
                        When a request is batchable then decode that many requests in parallel
  --prompt-concurrency PROMPT_CONCURRENCY
                        When a request is batchable then process that many prompts in parallel
  --prefill-step-size PREFILL_STEP_SIZE
                        Step size for prefill processing (default: 2048)
  --prompt-cache-size PROMPT_CACHE_SIZE
                        Maximum number of distinct KV caches to hold in the prompt cache
  --prompt-cache-bytes PROMPT_CACHE_BYTES
                        Maximum size in bytes of the KV caches
  --kv-bits KV_BITS     Number of bits for KV cache quantization (e.g., 4 or 8). Reduces memory
                        usage for long contexts. Disables batching, so requests are served one at
                        a time. Default: None (full precision)
  --kv-group-size KV_GROUP_SIZE
                        Group size for KV cache quantization (default: 64)
  --quantized-kv-start QUANTIZED_KV_START
                        Token position to start KV cache quantization (default: 5000)
  --pipeline            Use pipelining instead of tensor parallelism
```

## `turboquant-serve-vlm`

OpenAI- and Anthropic-compatible server for vision models (mlx-vlm's server with a TurboQuant loader). See [Vision models](../guides/vision.md#serve-it).

```text
turboquant-serve-vlm: mlx-vlm's server, taught to load TurboQuant checkpoints.

Added by turboquant-mlx:
  --reasoning-strength LEVEL
                        Default reasoning effort for models whose chat template
                        takes one (Muse Glimmer: low/medium/high/xhigh, default
                        high). Used when a request does not ask for one.
  --host HOST           Defaults to 127.0.0.1 here, not mlx-vlm's 0.0.0.0.
                        Pass --host 0.0.0.0 to accept connections from other
                        machines.

Muse Glimmer also gets its reasoning channel split out of `content`
automatically; pass --thinking-start-token / --thinking-end-token to override.

Everything below is mlx-vlm's own:

usage: turboquant-serve-vlm [-h] [--host HOST] [--port PORT] [--trust-remote-code] [--model MODEL]
                            [--image-model IMAGE_MODEL] [--tts-model TTS_MODEL]
                            [--stt-model STT_MODEL] [--embedding-model EMBEDDING_MODEL]
                            [--decision-model DECISION_MODEL] [--reranker-model RERANKER_MODEL]
                            [--model-dir MODEL_DIR] [--adapter-path ADAPTER_PATH]
                            [--vision-cache-size VISION_CACHE_SIZE] [--min-pixels MIN_PIXELS]
                            [--max-pixels MAX_PIXELS] [--prefill-step-size PREFILL_STEP_SIZE]
                            [--log-progress-interval LOG_PROGRESS_INTERVAL]
                            [--max-tokens MAX_TOKENS] [--enable-thinking]
                            [--thinking-budget THINKING_BUDGET]
                            [--thinking-start-token THINKING_START_TOKEN]
                            [--thinking-end-token THINKING_END_TOKEN] [--kv-bits KV_BITS]
                            [--kv-key-bits KV_KEY_BITS] [--kv-value-bits KV_VALUE_BITS]
                            [--kv-key-scheme {uniform,turboquant}]
                            [--kv-value-scheme {uniform,turboquant}]
                            [--kv-quant-scheme {uniform,turboquant}]
                            [--kv-group-size KV_GROUP_SIZE] [--max-kv-size MAX_KV_SIZE]
                            [--quantized-kv-start QUANTIZED_KV_START]
                            [--expert-cache-gb EXPERT_CACHE_GB] [--draft-model DRAFT_MODEL]
                            [--draft-kind {dflash,eagle3,mtp}]
                            [--draft-block-size DRAFT_BLOCK_SIZE] [--max-num-seqs MAX_NUM_SEQS]
                            [--top-logprobs-k TOP_LOGPROBS_K] [--api-key API_KEY] [--reload]
                            [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

MLX VLM Http Server.

options:
  -h, --help            show this help message and exit
  --host HOST           Host for the HTTP server (default:0.0.0.0)
  --port PORT           Port for the HTTP server (default: 8080)
  --trust-remote-code   Trust remote code when loading models from Hugging Face Hub.
  --model MODEL         Pre-load a language model at startup (e.g. mlx-
                        community/Qwen2.5-VL-3B-Instruct-4bit).
  --image-model IMAGE_MODEL
                        Pre-load an image generation model at startup.
  --tts-model TTS_MODEL
                        Pre-load a text-to-speech model at startup.
  --stt-model STT_MODEL
                        Pre-load a speech-to-text model at startup.
  --embedding-model EMBEDDING_MODEL
                        Pre-load an embedding model at startup.
  --decision-model DECISION_MODEL
                        Pre-load a decision model at startup.
  --reranker-model RERANKER_MODEL
                        Pre-load a supported reranker model at startup.
  --model-dir MODEL_DIR
                        Additional model directory, or parent directory containing model folders.
                        Repeat for multiple paths. Overrides MLX_VLM_MODEL_PATHS.
  --adapter-path ADAPTER_PATH
                        Adapter weights to load with the model.
  --vision-cache-size VISION_CACHE_SIZE
                        Max number of cached vision features (default: 20).
  --min-pixels MIN_PIXELS
                        Default minimum image pixel area for supported processors; requests may
                        override it.
  --max-pixels MAX_PIXELS
                        Default maximum image pixel area for supported processors; requests may
                        override it.
  --prefill-step-size PREFILL_STEP_SIZE
                        Tokens per prefill step (default: 2048).
  --log-progress-interval LOG_PROGRESS_INTERVAL
                        Decoded tokens between progress log messages; 0 disables periodic decode
                        progress (default: 10).
  --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate.
  --enable-thinking     Enable thinking mode by default for requests that do not set
                        enable_thinking explicitly.
  --thinking-budget THINKING_BUDGET
                        Default maximum number of tokens allowed inside a thinking block. Requests
                        can override this with thinking_budget.
  --thinking-start-token THINKING_START_TOKEN
                        Default token that opens a thinking block. Requests can override this with
                        thinking_start_token.
  --thinking-end-token THINKING_END_TOKEN, --thinking-eos-token THINKING_END_TOKEN
                        Default token that closes a thinking block. Requests can override this
                        with thinking_end_token.
  --kv-bits KV_BITS     Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).
  --kv-key-bits KV_KEY_BITS
                        Override the TurboQuant key bit-width (defaults to floor(--kv-bits)).
  --kv-value-bits KV_VALUE_BITS
                        Override the TurboQuant value bit-width (defaults to ceil(--kv-bits)).
  --kv-key-scheme {uniform,turboquant}
                        Override the KV quantization backend for keys only.
  --kv-value-scheme {uniform,turboquant}
                        Override the KV quantization backend for values only.
  --kv-quant-scheme {uniform,turboquant}
                        KV cache quantization backend.
  --kv-group-size KV_GROUP_SIZE
                        Group size for uniform KV cache quantization.
  --max-kv-size MAX_KV_SIZE
                        Maximum KV cache size in tokens.
  --quantized-kv-start QUANTIZED_KV_START
                        Start index for quantized KV cache.
  --expert-cache-gb EXPERT_CACHE_GB
                        For an mlx_vlm.moe_offload checkpoint, bound the resident routed-expert
                        set to this many GB (default: 70% of the GPU's recommended working set).
                        Ignored for a normal, non-offloaded checkpoint.
  --draft-model DRAFT_MODEL
                        Speculative drafter path or HF id. Qwen3.5/3.6/3.8 checkpoints with
                        bundled MTP weights can use the same path as --model.
  --draft-kind {dflash,eagle3,mtp}
                        Drafter family -- 'dflash', 'eagle3', or 'mtp' (native/assistant MTP).
                        Default: auto-detected from the drafter's HF model_type.
  --draft-block-size DRAFT_BLOCK_SIZE
                        Override the drafter's configured block size.
  --max-num-seqs MAX_NUM_SEQS
                        Maximum number of sequences decoded concurrently in the continuous batch.
                        Requests beyond this wait in the queue (backpressure), bounding peak
                        memory. Default: unbounded. Maps to MLX_VLM_MAX_NUM_SEQS.
  --top-logprobs-k TOP_LOGPROBS_K
                        Server-side cap for per-token top_logprobs (0-20, default 0 = disabled).
                        Maps to the TOP_LOGPROBS_K env var.
  --api-key API_KEY     Optional bearer token required for inference, model discovery, and
                        management endpoints. Maps to the MLX_VLM_SERVER_API_KEY env var.
  --reload              Enable auto-reload for development.
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO).
```

## `python -m turboquant_mlx.convert_vlm`

Convert a vision or diffusion model. See [Vision models](../guides/vision.md#convert-one).

```text
usage: convert_vlm.py [-h] --hf-path HF_PATH --mlx-path MLX_PATH [--bits {2,3,4}]
                      [--group-size {32,64,128}] [--rotation {hadamard,blockwise_hadamard,none}]
                      [--rotation-seed ROTATION_SEED] [--attn-bits {2,3,4}] [--mlp-bits {2,3,4}]
                      [--quantize-extras] [--extras-bits {4,8}] [--extras-group-size {32,64,128}]
                      [--protect-expert-layers PROTECT_EXPERT_LAYERS] [--protect-bits {3,4}]
                      [--ternary-experts] [--expert-down-bits {2,3,4}]

Convert an mlx-vlm model to TurboQuant-compressed MLX format

options:
  -h, --help            show this help message and exit
  --hf-path HF_PATH, --model HF_PATH
                        HuggingFace model path or local path
  --mlx-path MLX_PATH   Output directory for the MLX model
  --bits {2,3,4}, -b {2,3,4}
                        Quantization bit-width (default: 3)
  --group-size {32,64,128}, -g {32,64,128}
                        Group size (default: 32)
  --rotation {hadamard,blockwise_hadamard,none}
  --rotation-seed ROTATION_SEED
  --attn-bits {2,3,4}   Override bits for attention-block linears
  --mlp-bits {2,3,4}    Override bits for MLP / MoE expert linears
  --quantize-extras     Also quantize embeddings, dense MLP and vision tower to affine (for
                        memory-constrained machines, e.g. 16 GB Macs)
  --extras-bits {4,8}   Bit width for --quantize-extras (default 8). Use 4 when the extras are a
                        big share of the model — Muse Glimmer's embedding + vision tower are 3.3B
                        params, ~3.9 GB at 8-bit vs ~1.8 GB at 4-bit.
  --extras-group-size {32,64,128}
                        Group size for --quantize-extras (default 64)
  --protect-expert-layers PROTECT_EXPERT_LAYERS
                        Comma-separated layer indices whose EXPERTS keep --protect-bits instead of
                        --bits (quality lift for 2-bit experts, e.g. '0,1,2,27,28,29')
  --protect-bits {3,4}  Bit width for protected expert layers (default 3)
  --ternary-experts     Quantize MoE expert weights to the ternary {-c,0,+c} codebook (1.58-bit),
                        stored in the 2-bit slot. Attention and everything else stay at --attn-
                        bits/--bits.
  --expert-down-bits {2,3,4}
                        Quantize MoE expert down projections (down_proj, or a SwitchMLP's fc2) at
                        this Gaussian-codebook width while the rest of each expert takes the
                        --mlp-bits / --ternary-experts tier.
```

## `python -m turboquant_mlx.generate_vlm`

Generate with a vision or diffusion model, optionally with an image.

```text
usage: generate_vlm.py [-h] --model MODEL [--prompt PROMPT] [--max-tokens MAX_TOKENS]
                       [--temp TEMP] [--image IMAGE] [--prefill-step-size PREFILL_STEP_SIZE]
                       [--max-denoising-steps MAX_DENOISING_STEPS]
                       [--max-canvas-length MAX_CANVAS_LENGTH] [--reasoning LEVEL] [--no-think]

Generate text with a TurboQuant-compressed mlx-vlm model

options:
  -h, --help            show this help message and exit
  --model MODEL         TurboQuant model directory or HF repo ID
  --prompt PROMPT
  --max-tokens MAX_TOKENS
  --temp TEMP, --temperature TEMP
                        Sampling temperature (default: 0.0)
  --image IMAGE         Optional image path/URL for multimodal prompts
  --prefill-step-size PREFILL_STEP_SIZE
                        Tokens per prefill chunk (mlx-vlm default 2048). This is the memory knob
                        for long prompts: the transient prefill workspace scales with the chunk,
                        and a chunk of 256 or less also keeps every TurboQuant layer on the fused
                        Metal kernel, which materializes nothing. `turboquant-plan` recommends a
                        value when the default would not fit.
  --max-denoising-steps MAX_DENOISING_STEPS
                        Cap diffusion denoising steps (model default 48). Lower = faster, mild
                        quality cost (try 24)
  --max-canvas-length MAX_CANVAS_LENGTH
                        Cap the diffusion canvas length (model default 256). Lower = smaller
                        activation memory, useful on 16 GB machines (try 128)
  --reasoning LEVEL     Reasoning effort for models whose chat template takes one (Muse Glimmer:
                        low/medium/high/xhigh, default high). Lower spends far fewer tokens
                        thinking before answering — worth a lot on slow machines. Ignored by
                        templates without it.
  --no-think            Minimise reasoning: shorthand for the lowest effort the template supports,
                        plus `enable_thinking=False` for Qwen-style templates. NOTE this does not
                        guarantee zero reasoning — Muse Glimmer has no 'off' level and may still
                        emit a short thinking channel.
```

## `python -m turboquant_mlx.stream.stream_generate`

Generate with a Mixture-of-Experts model bigger than memory, streaming experts from disk. See [Models bigger than RAM](../guides/streaming.md).

```text
usage: stream_generate.py [-h] --model MODEL [--prompt PROMPT] [--max-tokens MAX_TOKENS]
                          [--temp TEMP] [--cache-budget-gb CACHE_BUDGET_GB] [--wire-memory]
                          [--prefetch-workers PREFETCH_WORKERS] [--prefetch-ahead PREFETCH_AHEAD]
                          [--fanout] [--pin-file PIN_FILE] [--no-learn-experts]
                          [--usage-file USAGE_FILE] [--no-hotlist]
                          [--max-active-experts MAX_ACTIVE_EXPERTS] [--use-page-cache]
                          [--no-page-cache] [--fast] [--ngram-offload] [--no-chat-template]
                          [--trust-remote-code] [--top-p TOP_P] [--top-k TOP_K]
                          [--rep-penalty REP_PENALTY] [--rep-ctx REP_CTX] [--no-think]
                          [--multi-think]

Stream-generate from a TurboQuant MoE model (experts paged from disk).

options:
  -h, --help            show this help message and exit
  --model MODEL         Local path or HF repo id of a TurboQuant model.
  --prompt PROMPT
  --max-tokens MAX_TOKENS
  --temp TEMP
  --cache-budget-gb CACHE_BUDGET_GB
                        Max resident expert memory (LRU-evicted). Lower = less RAM, more disk
                        reads. Pass 'auto' to size it from the machine (Metal's working set minus
                        resident weights and a runtime + KV reserve).
  --wire-memory         Raise MLX's wired-memory limit so weights + expert cache stay resident
                        under memory pressure (for constrained machines; roomy ones don't need
                        it).
  --prefetch-workers PREFETCH_WORKERS
                        Threads for parallel per-layer expert reads. 1 = serial baseline.
  --prefetch-ahead PREFETCH_AHEAD
                        Speculatively prefetch this many upcoming layers' experts (predicted from
                        the previous token's routing). 0 = off (default; helps only on fast NVMe
                        with spare bandwidth, ~neutral on a saturated USB bus). Set 1 to enable;
                        it self-disables if the storage proves bandwidth-bound.
  --fanout              Enable the same-layer read fan-out: submit a layer's known expert misses
                        to the read pool at layer start so the other projections' reads overlap
                        compute. Off by default; it trades away the coalesced serial read path,
                        which wins on a fast internal SSD with spare bandwidth and loses on
                        bandwidth-bound external storage. Unlike --prefetch-ahead it does not
                        self-disable, so measure it on your storage.
  --pin-file PIN_FILE   JSON {'pin': [[layer, expert], ...]} of hot experts to keep permanently
                        resident (from calibrate_experts.py). Without it, a hot_experts.json
                        shipped in the model directory is used automatically. Pins are preloaded
                        at startup.
  --no-learn-experts    don't record/reuse a per-user hot-expert profile (learning is on by
                        default; it only pins once it has seen enough of your own traffic)
  --usage-file USAGE_FILE
                        where to keep the learned expert profile (default: ~/.cache/turboquant-
                        mlx/usage/)
  --no-hotlist          Ignore a hot_experts.json shipped in the model directory.
  --max-active-experts MAX_ACTIVE_EXPERTS
                        Cap router top_k to min(native, this) on every MoE block (Flash-MoE
                        K-reduction: ~2x less streamed disk I/O at no quality cost up to K=4 on
                        validated models). Default 4; 0 = native routing.
  --use-page-cache      Force the OS page cache ON for expert reads ('trust the OS'; ~2.4x faster
                        decode when the model fits in RAM). Default: auto by model-size-vs-RAM.
  --no-page-cache       Force F_NOCACHE (page cache off). Default: auto by model-size-vs-RAM.
  --fast                Disable QJL correction for faster decode.
  --ngram-offload       Serve an n-gram embedding table (Qwen3.8-Flash-Next) from memory-mapped
                        files on the CPU instead of GPU memory: ~17.9 GiB less resident on the
                        2-bit build, bit-identical output. Keep the model directory in place while
                        the model is loaded. No effect on models without such a table.
  --no-chat-template
  --trust-remote-code   Allow the model's own tokenizer code (tokenization_*.py) to run. Needed
                        for Kimi K3 loaded by Hub repo id; a local K3 directory is trusted without
                        it.
  --top-p TOP_P         Nucleus sampling threshold. Defaults to the model's generation_config.json
                        value when present, else disabled. Pass 0 to force-disable.
  --top-k TOP_K         Top-k truncation. Defaults to the model's generation_config.json value
                        when present, else disabled. Pass 0 to force-disable.
  --rep-penalty REP_PENALTY
                        Repetition penalty (e.g. 1.05). Defaults to the model's
                        generation_config.json value when present, else disabled. Pass 0 or 1 to
                        force-disable.
  --rep-ctx REP_CTX     Repetition penalty context window in tokens (default: 256).
  --no-think            Disable thinking mode via the chat template (enable_thinking=False). Much
                        faster and immune to think-block loops on thinking-capable models.
  --multi-think         Allow more than one </think> token per generation. By default a second
                        </think> is masked once one has been emitted.
```

## `python -m turboquant_mlx.evaluate`

Perplexity evaluation (needs the `[eval]` extra).

```text
usage: evaluate.py [-h] --hf-path HF_PATH [--bits BITS [BITS ...]] [--group-size GROUP_SIZE]
                   [--include-affine] [--no-affine] [--include-qjl] [--no-qjl]
                   [--num-samples NUM_SAMPLES] [--seq-len SEQ_LEN] [--output OUTPUT]

Evaluate TurboQuant vs baselines on WikiText-2 perplexity

options:
  -h, --help            show this help message and exit
  --hf-path HF_PATH, --model HF_PATH
                        HuggingFace model path (e.g. meta-llama/Llama-3.2-1B)
  --bits BITS [BITS ...], -b BITS [BITS ...]
                        Bit-widths to evaluate (default: 2 3 4)
  --group-size GROUP_SIZE, -g GROUP_SIZE
                        Quantization group size (default: 64)
  --include-affine      Include MLX affine quantization baselines (default: True)
  --no-affine           Skip affine quantization baselines
  --include-qjl         Include TurboQuant+QJL configs (default: True)
  --no-qjl              Skip QJL configs
  --num-samples NUM_SAMPLES, -n NUM_SAMPLES
                        Number of evaluation chunks (default: 512)
  --seq-len SEQ_LEN     Sequence length per chunk (default: 512)
  --output OUTPUT, -o OUTPUT
                        Output JSON path (default: auto-generated)
```
