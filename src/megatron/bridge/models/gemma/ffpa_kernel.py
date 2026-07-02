# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FFPA (Faster Flash Prefill Attention) CuTeDSL *dense* kernel wrappers.

Vendored from NVIDIA-NeMo/Automodel
(``nemo_automodel/components/attention/ffpa_attention.py``) — only the dense
forward/backward and the low-level readiness probe, which is all the Gemma-4
dense **all-gather-KV** context-parallel path needs. The automodel varlen /
ring / HF-attention-interface machinery is intentionally *not* vendored: under
all-gather-KV each rank already holds full K/V, so the head_dim-512 global
layers are served by dense FFPA calls over per-(zigzag-)chunk key ranges
(past = non-causal, diagonal = causal) merged with online softmax — see
``gemma4_cp_attention.py``.

Requires ``ffpa-attn`` (>=0.2.1) + ``nvidia-cutlass-dsl`` (>=4.5.1) in the venv;
:func:`ffpa_dense_available` reports whether the CuTeDSL ops loaded. FFPA is the
head_dim-512 kernel (``_FFPA_HEAD_DIM``); it has no arbitrary-mask input (only a
``causal`` bool), which is why the caller splits the key range instead.
"""

from __future__ import annotations

import torch

_FFPA_HEAD_DIM = 512
_READY: bool | None = None


def ffpa_dense_available() -> bool:
    """Whether the FFPA CuTeDSL dense ops are importable and registered.

    Cached after the first probe (matches automodel ``_ffpa_low_level_ready``).
    """
    global _READY
    if _READY is None:
        try:
            import ffpa_attn.cute  # noqa: F401

            _ = torch.ops.ffpa_attn._fwd_cute
            _ = torch.ops.ffpa_attn._bwd_cute
            _READY = True
        except Exception:
            _READY = False
    return _READY


def ffpa_dense_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FFPA CuTeDSL *dense* forward on ``[B, H, N, D]`` SDPA-layout tensors.

    Returns ``(out[B, Hq, Nq, D], lse[B, Hq, Nq] fp32)``; handles GQA and
    causal/non-causal internally. Exposed (not via the package autograd) so the
    caller gets the per-chunk ``lse`` for its online-softmax merge.
    """
    from ffpa_attn.cute import _ffpa_attn_forward_cute

    return _ffpa_attn_forward_cute(q, k, v, float(scale), bool(causal), return_lse=True)


def ffpa_dense_bwd(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FFPA CuTeDSL *dense* backward using the *caller-supplied* out/lse.

    Returns ``(dq, dk, dv)`` all in ``[B, H, N, D]`` SDPA layout (dK/dV reduced
    to ``Hkv`` for GQA). The caller feeds the globally merged out/lse (not a
    chunk-local one) so each chunk's gradient sees the true full-row softmax.
    """
    from ffpa_attn.cute import _ffpa_attn_backward_cute

    return _ffpa_attn_backward_cute(grad_out, q, k, v, out, lse, float(scale), bool(causal))
