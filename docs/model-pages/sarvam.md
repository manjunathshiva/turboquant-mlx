# Sarvam MoE

[sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) and later models of that
type convert and run through the `sarvam_moe` port shipped here (mlx-lm has no
native module). The port is verified against the fp32 reference three ways: full
key/shape coverage of all 7122 source tensors, layer-1 parity at 5.4e-7, and
whole-model logit agreement.

> **No quantized sarvam build is published, deliberately.** At 4-bit this model
> degenerates on long generations — measured as a rate over repeated trials at
> temp 0.7 / top_p 0.9, **TurboQuant 4-bit degenerated 8/15 (53%) and mlx-lm
> affine 4-bit 15/15 (100%)**. TurboQuant is clearly the better of the two and
> neither is good enough to ship. Use a higher bit width and validate long-form
> output on your own prompts before relying on it.
