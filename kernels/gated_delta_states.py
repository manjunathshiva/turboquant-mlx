# Copyright (c) 2025 Prince Canuma — mlx-vlm, MIT License (see NOTICE).
# Adapted for TurboQuant-MLX — Copyright 2026 Manjunath Janardhan.
"""Gated-DeltaNet update that also returns the recurrent state after each step.

Vendored from mlx-vlm 0.7.0, ``mlx_vlm/models/qwen3_5/gated_delta.py``
(``gated_delta_update_with_states`` and its Metal kernel), unchanged apart from
imports and one fix: the ``states`` output strides batch rows by ``StateT``, its
own step count, where upstream used ``T``. With ``B > 1`` and ``state_steps < T``
the upstream offset writes past the buffer (pinned by
``test_state_capture_kernel_is_right_for_batches_and_partial_steps``). mlx-vlm is
not a dependency of the text path, so the code is copied rather than imported.

The stock kernel (``mlx_lm.models.gated_delta.gated_delta_update``) already
computes the state after every position and keeps only the last. This variant
also writes the first ``state_steps`` of them to a third output, which is what
lets speculative decoding rewind a hybrid cache without a snapshot or a replay
(see ``turboquant_mlx.mtp``).

Same arguments and the same ``(y, final_state)`` as ``gated_delta_update``, plus
``states`` of shape ``(B, state_steps, Hv, Dv, Dk)``.
"""

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def _compute_g_beta(A_log, a, b, dt_bias):
    return compute_g(A_log, a, dt_bias), mx.sigmoid(b)


def _make_gated_delta_with_states_kernel(has_mask: bool = False):
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        states += ((b_idx * StateT * Hv + hv_idx) * Dv) * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto states_ = states + dv_idx * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
            if ({mask_source}) {{
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                }}
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                }}
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {{
                    y[dv_idx] = static_cast<InT>(out);
                }}
            }} else {{
                y[dv_idx] = static_cast<InT>(0);
            }}

            if (t < StateT) {{
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    states_[s_idx] = static_cast<StT>(state[i]);
                }}
            }}

            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            if (t < StateT) {{
                states_ += Hv * Dv * Dk;
            }}
            g_ += Hv;
            beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"turboquant_gated_delta_with_states{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "states"],
        source=source,
    )


_gated_delta_with_states_kernel = _make_gated_delta_with_states_kernel(False)
_gated_delta_with_states_kernel_masked = _make_gated_delta_with_states_kernel(True)


def _gated_delta_with_states_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    state_steps: Optional[int] = None,
):
    B, T, Hk, _Dk = q.shape
    Hv = v.shape[-2]
    state_steps = T if state_steps is None else int(state_steps)
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    states = []
    for t in range(T):
        old_state = state
        decay = g[:, t, :, None, None]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        y = (state * q[:, t, :, None, :]).sum(axis=-1)

        if mask is not None:
            valid = mask[:, t]
            state = mx.where(valid[:, None, None, None], state, old_state)
            y = mx.where(valid[:, None, None], y, 0)

        ys.append(y.astype(q.dtype))
        if t < state_steps:
            states.append(state)
    if states:
        stacked_states = mx.stack(states, axis=1)
    else:
        stacked_states = mx.zeros((B, 0, *state.shape[1:]), dtype=state.dtype)
    return mx.stack(ys, axis=1), state, stacked_states


def gated_delta_update_with_states(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
    state_steps: Optional[int] = None,
):
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    B, T, Hk, Dk = k.shape
    state_steps = T if state_steps is None else int(state_steps)
    if not 0 <= state_steps <= T:
        raise ValueError("state_steps must be between zero and the sequence length.")
    if state is None:
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (
        g.ndim != 3
        or not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return _gated_delta_with_states_ops(q, k, v, g, beta, state, mask, state_steps)

    Hv, Dv = v.shape[2:]

    input_type = q.dtype
    state_type = state.dtype
    kernel = _gated_delta_with_states_kernel
    inputs = [q, k, v, g, beta, state, T]
    if mask is not None:
        kernel = _gated_delta_with_states_kernel_masked
        inputs.append(mask)
    if kernel is None:
        return _gated_delta_with_states_ops(q, k, v, g, beta, state, mask, state_steps)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("StateT", state_steps),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, state_steps, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type, state_type],
    )
