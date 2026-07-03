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

"""Context-parallel core attention for Gemma-4 Dense (opt-in via ``hybrid_cp_attention``).

The dense model interleaves sliding (head_dim 256) and global (head_dim 512)
attention layers. head_dim 512 has no fused flash/cuDNN kernel, so mcore/TE ring
CP cannot serve the global layers. This module provides a CP-capable core
attention that works for both head dims without materializing an ``[S,S]`` score
matrix, using an **all-gather-KV** scheme:

    - queries stay sequence-sharded (each CP rank owns ``S/cp`` rows),
    - K/V are all-gathered to full length (differentiable: reduce-scatter bwd),
    - each rank attends its local queries against the full K/V and returns its
      own ``S/cp`` output rows,
    - correctness comes from a causal (+ sliding-window) mask over *global* token
      positions, derived from mcore's load-balanced CP layout.

All-gather (rather than a ring) is chosen because the global layers have only a
couple of KV heads, so full-length K/V is small; and it needs no hand-written
ring backward. The kernel is PyTorch SDPA (memory-efficient backend handles
head_dim 512 with no S²); a FlexAttention ``mask_mod`` path (avoiding even the
``[S/cp, S]`` mask tensor) is a future optimization.

The standalone algorithm + parity vs no-CP is validated in
``rovo-train-bridge/cp/test_global_cp_parity.py``.
"""

import copy
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.profiler import record_function

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.gemma.ffpa_kernel import ffpa_dense_available, ffpa_dense_bwd, ffpa_dense_fwd
from megatron.bridge.models.gemma.modeling_gemma4 import _is_gemma4_sliding_layer


try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention as _TEDotProductAttention
except Exception:  # pragma: no cover - TE always present in our container, but keep import optional
    _TEDotProductAttention = None


_FLEX_ATTENTION = None
_FLEX_CREATE_BLOCK_MASK = None


def _flex_compile_enabled() -> bool:
    """Whether to ``torch.compile`` FlexAttention (default: yes).

    Eager FlexAttention is a *debug fallback*: its backward decomposition
    (``torch/_higher_order_ops/flex_attention.py::sdpa_dense_backward``)
    materializes per-head ``[S_q, S_kv]`` score gradients — a huge transient
    (~16 GiB at 32k for the all-gathered global layers) that OOMs and is the
    reason 32k didn't fit. ``torch.compile`` lowers FlexAttention to the
    flash-style Triton kernel whose forward *and* backward are block-wise (no S²
    materialization) and faster. Disable with ``GEMMA4_FLEX_COMPILE=0`` to fall
    back to eager (e.g. for debugging mask_mod with print/breakpoints).
    """
    return os.environ.get("GEMMA4_FLEX_COMPILE", "1") != "0"


def _load_flex():
    """Lazily import FlexAttention (torch >= 2.5), compiled by default. Cached.

    Returns ``(flex_attention, create_block_mask)``. ``flex_attention`` is
    ``torch.compile``d unless ``GEMMA4_FLEX_COMPILE=0`` (see
    :func:`_flex_compile_enabled`). FlexAttention is the only head-dim-512-capable
    kernel that avoids both the ``[b,h,S,S]`` scores and the explicit
    ``[S/cp, S]`` mask tensor: the causal / sliding-window / pack-id / padding
    predicate is a ``mask_mod`` evaluated block-sparsely inside the kernel.
    ``create_block_mask`` stays eager — it evaluates the predicate at 128×128
    block granularity (O((S/128)²), cheap), not per element.
    """
    global _FLEX_ATTENTION, _FLEX_CREATE_BLOCK_MASK
    if _FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        if _flex_compile_enabled():
            _FLEX_ATTENTION = torch.compile(flex_attention)
            # Eager create_block_mask densely materializes the [Q_LEN, KV_LEN] grid
            # to derive block sparsity (~21 GB GPU at the 16384×131072 global-layer
            # shape @128k); compiled it is block-sparse (~0.07 GB). Big long-context
            # win — build it compiled too.
            _FLEX_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
        else:
            _FLEX_ATTENTION = flex_attention
            _FLEX_CREATE_BLOCK_MASK = create_block_mask
    return _FLEX_ATTENTION, _FLEX_CREATE_BLOCK_MASK


class _AllGatherSeq(torch.autograd.Function):
    """All-gather along the sequence dim (dim 0); reduce-scatter (sum) backward.

    Assumes equal shards across the CP group (mcore load balancing gives every
    rank ``S/cp`` tokens).
    """

    @staticmethod
    def forward(ctx, x_local: Tensor, group) -> Tensor:
        ctx.group = group
        ctx.world = dist.get_world_size(group)
        gathered = [torch.empty_like(x_local) for _ in range(ctx.world)]
        dist.all_gather(gathered, x_local.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        shards = [g.contiguous() for g in grad_out.chunk(ctx.world, dim=0)]
        out = torch.empty_like(shards[0])
        dist.reduce_scatter(out, shards, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def _cp_global_positions(cp_rank: int, cp_size: int, s_local: int, device) -> Tensor:
    """Global token indices owned by this CP rank under mcore's load-balanced layout.

    mcore splits the sequence into ``2*cp`` chunks and gives rank ``r`` chunk
    ``r`` and chunk ``2*cp-1-r`` (concatenated). For ``cp==1`` this is just
    ``arange(s_local)``.
    """
    if cp_size == 1:
        return torch.arange(s_local, device=device)
    chunk = s_local // 2  # each rank holds two load-balanced chunks
    a = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    b_start = (2 * cp_size - 1 - cp_rank) * chunk
    b = torch.arange(b_start, b_start + chunk, device=device)
    return torch.cat([a, b], dim=0)


def _merge_online(outs: list[Tensor], lses: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Flash-style online-softmax combine of per-key-range attention chunks.

    ``outs[c]`` is ``[b, np, L, hn]`` and ``lses[c]`` is ``[b, np, L]`` (fp32 log-
    sum-exp of chunk ``c``'s key range). Returns the merged ``(out, lse)`` with
    each chunk re-weighted by its share of the full-row softmax mass::

        lse = logaddexp_c(lse_c);  out = sum_c out_c * exp(lse_c - lse)

    Accumulated in fp32; no ``[L, S]`` score matrix is formed.
    """
    lse = lses[0]
    for extra in lses[1:]:
        lse = torch.logaddexp(lse, extra)
    out = torch.zeros_like(outs[0], dtype=torch.float32)
    for o, l in zip(outs, lses):
        out = out + o.float() * torch.exp(l - lse).unsqueeze(-1)
    return out.to(outs[0].dtype), lse


def _contiguous_runs(q_pos: Tensor) -> list[tuple[int, int, int]]:
    """Split ``q_pos`` into maximal runs of consecutive global positions.

    Returns ``[(local_start, length, global_start), ...]``. Under mcore's
    load-balanced (zigzag) layout each rank owns two ascending runs (its low +
    mirrored-high chunk), so this yields two entries; contiguous / ``cp==1``
    layouts yield one. Each run is contiguous in *both* the local query tensor
    and global coordinates, so its diagonal key block (same global range) is
    aligned from index 0 — exactly what FFPA's ``causal=True`` expects.
    """
    p = q_pos.tolist()
    runs: list[tuple[int, int, int]] = []
    i, n = 0, len(p)
    while i < n:
        j = i
        while j + 1 < n and p[j + 1] == p[j] + 1:
            j += 1
        runs.append((i, j - i + 1, p[i]))
        i = j + 1
    return runs


def _contiguous_runs_packed(
    q_pos: Tensor, doc_q: Tensor, real_q: Tensor, pad_start: Tensor
) -> list[tuple[int, int, int, int]]:
    """Split packed queries into runs of consecutive, same-document, real tokens.

    Returns ``[(local_start, length, global_start, doc_start), ...]``. A run is a
    maximal block of local query indices that are (a) real tokens (not padding),
    (b) consecutive in global position, and (c) in the same document. ``doc_start``
    is that document's padded start offset (``cu_seqlens_q_padded[doc]``), used as
    the ``past_start`` for :class:`_AllGatherFFPA` so ``past`` keys stay inside the
    document (block-diagonal). Padding queries start no run (their output rows are
    left zero). Under the THD zigzag partition each document contributes up to two
    such runs per rank (its low + mirrored-high chunk), further truncated where a
    chunk runs past the document's real length into padding.
    """
    p = q_pos.tolist()
    dq = doc_q.tolist()
    rq = real_q.tolist()
    ps = pad_start.tolist()
    runs: list[tuple[int, int, int, int]] = []
    i, n = 0, len(p)
    while i < n:
        if not rq[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and rq[j + 1] and dq[j + 1] == dq[i] and p[j + 1] == p[j] + 1:
            j += 1
        runs.append((i, j - i + 1, p[i], ps[dq[i]]))
        i = j + 1
    return runs


class _AllGatherFFPA(torch.autograd.Function):
    """FFPA-dense attention for all-gather-KV CP: per query run, past + diagonal.

    Inputs are this rank's queries ``q`` ``[b, np, sq, hn]`` and the full,
    **globally sorted** K/V ``[b, ng, S, hn]`` (so ``k_sorted[..., g, :]`` is the
    token at global position ``g``). ``runs`` is a list of
    ``(local_start, length, global_start, past_start)``: for each contiguous query
    run at global ``[g0, g0+L)`` the causal rectangle over key range
    ``[past_start, g0+L)`` factors into two shapes FFPA can express with only its
    ``causal`` bool:

    * **past** keys ``[past_start, g0)`` — every key precedes every query in the
      run → full attention (``causal=False``);
    * **diagonal** keys ``[g0, g0+L)`` — same global range as the queries, aligned
      from 0 → ordinary lower-triangular (``causal=True``).

    ``past_start`` is ``0`` for the (non-packed) whole-sequence causal case and the
    query's **document start** ``d0`` for packed sequences (so ``past`` covers only
    the same document's earlier tokens — block-diagonal). Padding query rows are
    never in a run, so ``out`` is zero-initialized and they stay zero (their loss
    grad is zero, so this is exact for training).

    The two chunk outputs are merged by online softmax (:func:`_merge_online`).
    Backward recomputes each chunk's ``dq/dk/dv`` from the *globally merged*
    out/lse (so every chunk sees the true full-row softmax) — the automodel ring
    backward contract, here without p2p since K/V are already gathered.
    """

    @staticmethod
    def forward(ctx, q: Tensor, k_sorted: Tensor, v_sorted: Tensor, runs, scale: float) -> Tensor:
        b, np_, sq, hn = q.shape
        out = q.new_zeros(b, np_, sq, hn)
        merged = []  # (i0, L, g0, ps, out_run, lse_run) for backward
        for i0, L, g0, ps in runs:
            q_run = q[:, :, i0 : i0 + L].contiguous()
            outs, lses = [], []
            if g0 > ps:
                oP, lP = ffpa_dense_fwd(
                    q_run, k_sorted[:, :, ps:g0].contiguous(), v_sorted[:, :, ps:g0].contiguous(),
                    scale=scale, causal=False,
                )
                outs.append(oP)
                lses.append(lP)
            oD, lD = ffpa_dense_fwd(
                q_run, k_sorted[:, :, g0 : g0 + L].contiguous(), v_sorted[:, :, g0 : g0 + L].contiguous(),
                scale=scale, causal=True,
            )
            outs.append(oD)
            lses.append(lD)
            out_run, lse_run = _merge_online(outs, lses)
            out[:, :, i0 : i0 + L] = out_run
            merged.append((i0, L, g0, ps, out_run, lse_run))
        ctx.save_for_backward(q, k_sorted, v_sorted)
        ctx.merged = merged
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        q, k_sorted, v_sorted = ctx.saved_tensors
        scale = ctx.scale
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k_sorted)
        dv = torch.zeros_like(v_sorted)
        for i0, L, g0, ps, out_run, lse_run in ctx.merged:
            go = grad_out[:, :, i0 : i0 + L].contiguous()
            q_run = q[:, :, i0 : i0 + L].contiguous()
            dqD, dkD, dvD = ffpa_dense_bwd(
                go, q_run, k_sorted[:, :, g0 : g0 + L].contiguous(), v_sorted[:, :, g0 : g0 + L].contiguous(),
                out_run, lse_run, scale=scale, causal=True,
            )
            dq[:, :, i0 : i0 + L] += dqD
            dk[:, :, g0 : g0 + L] += dkD
            dv[:, :, g0 : g0 + L] += dvD
            if g0 > ps:
                dqP, dkP, dvP = ffpa_dense_bwd(
                    go, q_run, k_sorted[:, :, ps:g0].contiguous(), v_sorted[:, :, ps:g0].contiguous(),
                    out_run, lse_run, scale=scale, causal=False,
                )
                dq[:, :, i0 : i0 + L] += dqP
                dk[:, :, ps:g0] += dkP
                dv[:, :, ps:g0] += dvP
        return dq, dk, dv, None, None


class Gemma4DenseCPAttention(torch.nn.Module):
    """CP-capable core attention for Gemma-4 Dense (both sliding and global layers).

    Implements the mcore ``core_attention`` interface. Works for ``cp_size >= 1``
    (a no-op gather at ``cp_size == 1``, so it doubles as a non-CP kernel when the
    hybrid path is enabled).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.is_sliding = _is_gemma4_sliding_layer(config, layer_number)
        # Gemma-4 dense uses softmax_scale = 1.0 (set on the attention config).
        self.softmax_scale = softmax_scale if softmax_scale is not None else getattr(config, "softmax_scale", None)
        if self.softmax_scale is None:
            self.softmax_scale = 1.0
        # Sliding window (left) in tokens; None for global (full causal) layers.
        self.window_left: Optional[int] = None
        if self.is_sliding:
            ws = getattr(config, "window_size", None)
            if ws is not None:
                self.window_left = ws[0] if isinstance(ws, (tuple, list)) else int(ws)

        # Sliding layers: delegate to mcore TEDotProductAttention, which does CP itself. TE
        # supports sliding-window CP only via cp_comm_type in {"a2a","all_gather"} (NOT the
        # "p2p" ring — asserted in TE). We default to "a2a" (Ulysses): it shards KV by *heads*
        # so each rank's KV working set is full_KV/cp — the memory win (equivalent to a ring's
        # S/cp, partitioned by head) — while supporting the window (see _build_te_sliding).
        # This is the dominant memory lever (PLAN §32.0/§33): head_dim 256 + window are TE-
        # supported (global hd512 is not → it keeps the FFPA all-gather path). TE gets the
        # same config.window_size (left,0) and config.softmax_scale the flex/ffpa path used →
        # identical mask + scale (parity by construction, validated). Rollback / parity
        # baseline: GEMMA4_SLIDING_CP=allgather restores the flex all-gather sliding path.
        self._te_sliding = None
        if (
            self.is_sliding
            and self.window_left is not None
            and os.environ.get("GEMMA4_SLIDING_CP", "te") != "allgather"
        ):
            if _TEDotProductAttention is None:
                raise RuntimeError(
                    "GEMMA4_SLIDING_CP requests the TE sliding path but TEDotProductAttention "
                    "is unavailable; set GEMMA4_SLIDING_CP=allgather to use the flex all-gather path."
                )
            self._te_sliding = self._build_te_sliding(
                config, layer_number, attn_mask_type, attention_type, attention_dropout,
                softmax_scale, cp_comm_type, pg_collection,
            )
        # Cache of FlexAttention BlockMasks keyed by (sq, skv) for the *non-packed*
        # path: the causal(+window) predicate over this rank's static global
        # positions is identical every step and across every layer of this type,
        # so build the block mask once and reuse it (avoids re-running
        # create_block_mask each forward). Packed masks depend on the per-batch
        # cu_seqlens, so they are not cached here.
        self._block_mask_cache: dict = {}
        # Cache of (runs, sort_order) for the FFPA path, keyed by (sq, skv). The
        # per-rank query runs and the KV global-sort permutation are static per
        # sequence length (non-packed), identical every step and layer-of-type.
        self._ffpa_cache: dict = {}

    def _build_te_sliding(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float],
        softmax_scale: Optional[float],
        cp_comm_type: Optional[str],
        pg_collection,
    ):
        """Construct a windowed :class:`TEDotProductAttention` for this sliding layer.

        The window is *forced* on: mcore's ``is_layer_window_attention`` misreads Gemma's
        string-list ``window_attn_skip_freq`` (any non-empty string ⇒ True), so we set it to
        ``None`` on a deep copy (⇒ True whenever ``window_size`` is set) and pin
        ``window_size=(window_left, 0)`` — the exact left window the flex/ffpa path used, so
        the mask is identical. ``softmax_scale`` is pinned to the module's resolved scale
        (=config's, 1.0 for Gemma-4) so the score scaling matches too. TE handles CP itself
        (p2p ring by default → S/cp working set) and packed THD via ``packed_seq_params``.
        """
        te_config = copy.deepcopy(config)
        te_config.window_attn_skip_freq = None
        te_config.window_size = (int(self.window_left), 0)
        # TE CP supports sliding window ONLY with cp_comm_type in {"a2a","all_gather"} (NOT
        # "p2p" ring — asserted in TE context_parallel.py). Default to "a2a" (Ulysses):
        # it shards KV by *heads* so each rank's KV working set is full_KV/cp — the memory
        # win (equivalent to the ring's S/cp, partitioned by head instead of sequence) —
        # and it supports the window. "all_gather" is the fallback (full-S KV per rank =
        # no memory win, just a faster fused kernel). Requires heads % cp == 0 for a2a
        # (Gemma-4: 32 Q / 16 KV heads → ok through cp=16). Env-overridable.
        sliding_cp_comm = os.environ.get("GEMMA4_SLIDING_CP_COMM", "a2a")
        kwargs = dict(
            config=te_config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=self.softmax_scale,
            cp_comm_type=sliding_cp_comm,
        )
        if pg_collection is not None:
            kwargs["pg_collection"] = pg_collection
        return _TEDotProductAttention(**kwargs)

    def _build_mask(self, q_pos: Tensor, kv_pos: Tensor) -> Tensor:
        """[sq, sk] bool mask (True = attend): causal + optional sliding window.

        Only used by the legacy ``sdpa``/``naive`` fallback kernels — the default
        ``flex`` path expresses this predicate as a block-sparse ``mask_mod``
        without materializing the ``[sq, sk]`` tensor.
        """
        delta = q_pos[:, None] - kv_pos[None, :]  # >= 0 means kv is at/behind q
        allow = delta >= 0
        if self.window_left is not None:
            allow = allow & (delta <= self.window_left)
        return allow

    def _make_mask_mod(
        self,
        q_pos: Tensor,
        kv_pos: Tensor,
        pack_q: Optional[Tensor],
        pack_kv: Optional[Tensor],
        valid_kv: Optional[Tensor],
    ):
        """Build a FlexAttention ``mask_mod`` over *global* token positions.

        Maps each local query/key index to its global position (``q_pos``/
        ``kv_pos``) and encodes causal + optional sliding-window + optional
        pack-id (document) + optional key-validity (padding) — the same predicate
        the automodel reference uses (PLAN §9), evaluated block-sparsely so no
        ``[S/cp, S]`` tensor is formed. The ``is not None`` branches are resolved
        at trace time (captured Python values), not per element.
        """
        window_left = self.window_left

        def mask_mod(b_idx, h_idx, q_idx, kv_idx):  # noqa: ANN001 - flex trace signature
            qg = q_pos[q_idx]
            kg = kv_pos[kv_idx]
            allow = kg <= qg  # causal on global positions
            if window_left is not None:
                allow = allow & (qg - kg <= window_left)
            if pack_q is not None:
                allow = allow & (pack_q[q_idx] == pack_kv[kv_idx])  # same document
            if valid_kv is not None:
                allow = allow & valid_kv[kv_idx]  # drop padding keys
            return allow

        return mask_mod

    @staticmethod
    def _flex_kernel_options(head_dim: int) -> Optional[dict]:
        """Triton block/stage options for the compiled flex kernel, by head_dim.

        The default flex kernel tiles Q/K/V in 128-wide blocks; at **head_dim 512**
        (Gemma-4 global layers) that needs ~257 KB of shared memory per program —
        over the H200's 227 KB limit — so compilation fails with "No valid triton
        configs. out of resource". Forcing small **32×32** blocks (for the forward
        *and* both backward kernels: BLOCK_M/N, BLOCK_M1/N1, BLOCK_M2/N2) with
        ``num_stages=2`` brings it within budget (validated: fwd+bwd, ~0.2 GB, no
        S²). head_dim ≤ 256 (sliding layers) keeps the autotuner default (larger
        blocks, faster). Only meaningful for the compiled kernel. Env-overridable
        (``GEMMA4_FLEX_BLOCK`` / ``GEMMA4_FLEX_STAGES``) for tuning.
        """
        if head_dim <= 256:
            return None
        bm = int(os.environ.get("GEMMA4_FLEX_BLOCK", "32"))
        ns = int(os.environ.get("GEMMA4_FLEX_STAGES", "2"))
        return {
            "BLOCK_M": bm, "BLOCK_N": bm,
            "BLOCK_M1": bm, "BLOCK_N1": bm,
            "BLOCK_M2": bm, "BLOCK_N2": bm,
            "num_stages": ns,
        }

    def _flex_forward(
        self,
        q: Tensor,  # [b, np, sq, hn]
        k: Tensor,  # [b, ng, S, hn]  (GQA groups kept unexpanded)
        v: Tensor,
        q_pos: Tensor,
        kv_pos: Tensor,
        scale: float,
        pack_q: Optional[Tensor],
        pack_kv: Optional[Tensor],
        valid_kv: Optional[Tensor],
    ) -> Tensor:
        flex_attention, create_block_mask = _load_flex()
        sq = q.shape[2]
        skv = k.shape[2]
        # Reuse a cached block mask on the non-packed path (static positions);
        # rebuild every call when packing (mask depends on the batch's cu_seqlens).
        cache_key = None if pack_q is not None else (sq, skv)
        if cache_key is not None and cache_key in self._block_mask_cache:
            block_mask = self._block_mask_cache[cache_key]
        else:
            mask_mod = self._make_mask_mod(q_pos, kv_pos, pack_q, pack_kv, valid_kv)
            block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=sq, KV_LEN=skv, device=q.device)
            if cache_key is not None:
                self._block_mask_cache[cache_key] = block_mask
        # head_dim-512 needs small Triton blocks to fit shared memory (see above);
        # only applied to the compiled kernel (eager ignores/ can't use them).
        kernel_options = self._flex_kernel_options(q.shape[-1]) if _flex_compile_enabled() else None
        # enable_gqa lets K/V keep `ng` heads (no repeat_interleave → no O(S·np·hn) expansion).
        return flex_attention(
            q, k, v, block_mask=block_mask, scale=scale,
            enable_gqa=(q.shape[1] != k.shape[1]), kernel_options=kernel_options,
        )

    def _ffpa_forward(
        self,
        q: Tensor,  # [b, np, sq, hn]
        k: Tensor,  # [b, ng, S, hn]  (gather order, GQA groups unexpanded)
        v: Tensor,
        q_pos: Tensor,  # [sq]  global positions of this rank's queries
        kv_pos: Tensor,  # [S]   global positions of the gathered K/V (gather order)
        scale: float,
    ) -> Tensor:
        """Global (head_dim-512) layer via FFPA-dense over all-gathered K/V.

        The gathered K/V arrive in CP gather order, not global order, so we first
        permute them so ``k_sorted[..., g, :]`` is the token at global position
        ``g`` (dense coverage of ``0..S-1`` — the non-packed CP case). Then each
        query run runs the past/diagonal FFPA split (:class:`_AllGatherFFPA`).
        Same causal-over-global result as the flex path, faster head_dim-512
        kernel. Falls back to flex for the sliding, packed, or FFPA-unavailable
        cases (handled by the caller).
        """
        sq = q.shape[2]
        skv = k.shape[2]
        cached = self._ffpa_cache.get((sq, skv))
        if cached is None:
            order = torch.argsort(kv_pos)  # gather-order index that sorts positions ascending
            # Dense coverage check: sorted positions must be exactly 0..S-1 so that
            # "past = [:g0]" / "diagonal = [g0:g0+L]" slice by count is correct.
            sorted_pos = kv_pos[order]
            if not torch.equal(sorted_pos, torch.arange(skv, device=kv_pos.device)):
                raise RuntimeError("FFPA CP path requires dense global positions (non-packed).")
            # Non-packed: one causal sequence, so past starts at global 0 for every run.
            runs = [(i0, L, g0, 0) for (i0, L, g0) in _contiguous_runs(q_pos)]
            self._ffpa_cache[(sq, skv)] = (runs, order)
        else:
            runs, order = cached
        k_sorted = k.index_select(2, order)
        v_sorted = v.index_select(2, order)
        return _AllGatherFFPA.apply(q, k_sorted, v_sorted, runs, scale)

    def _ffpa_forward_packed(
        self,
        q: Tensor,  # [b, np, sq, hn]
        k: Tensor,  # [b, ng, S, hn]  (gather order, GQA groups unexpanded)
        v: Tensor,
        q_pos: Tensor,  # [sq]  global positions of this rank's queries
        kv_pos: Tensor,  # [S]   global positions of the gathered K/V (gather order)
        scale: float,
        doc_q: Tensor,  # [sq]  document id of each local query
        real_q: Tensor,  # [sq]  bool: query is a real (non-padding) token
        pad_start: Tensor,  # [n_doc]  padded start offset of each document
    ) -> Tensor:
        """Global (head_dim-512) layer via FFPA-dense for a PACKED sequence under CP.

        Same all-gather + past/diagonal FFPA split as :meth:`_ffpa_forward`, but the
        query runs are per-document (built by :func:`_contiguous_runs_packed`) and
        each run's ``past`` starts at its document's padded start ``d0`` instead of
        global 0 — so a query only attends inside its own document (block-diagonal),
        matching the flex ``mask_mod`` (causal + pack-id + padding). Padding keys sit
        past every document's real range, so the ``[d0, g0)`` / ``[g0, g0+L)`` slices
        never touch them; padding query rows stay zero (loss-masked → zero grad). Not
        cached: the run layout depends on the batch's ``cu_seqlens``.
        """
        skv = k.shape[2]
        order = torch.argsort(kv_pos)
        sorted_pos = kv_pos[order]
        if not torch.equal(sorted_pos, torch.arange(skv, device=kv_pos.device)):
            raise RuntimeError("FFPA CP packed path requires dense global positions (all-gathered THD).")
        runs = _contiguous_runs_packed(q_pos, doc_q, real_q, pad_start)
        k_sorted = k.index_select(2, order)
        v_sorted = v.index_select(2, order)
        return _AllGatherFFPA.apply(q, k_sorted, v_sorted, runs, scale)

    def _thd_cp_positions(self, packed_seq_params: PackedSeqParams, sq: int, cp_size: int, cp_rank: int, device):
        """Global token indices this CP rank owns under mcore's *packed* (THD) layout.

        When packing is on, ``gpt_step`` distributes tokens with TE's
        ``thd_get_partitioned_indices`` (a per-document load-balanced split), NOT
        the whole-sequence zigzag of :func:`_cp_global_positions`. We reproduce
        the *identical* partition here from the full (padded) ``cu_seqlens`` so the
        module's global positions agree with how the batch was actually sharded —
        the essential consistency for the mask_mod. Requires the padded
        ``cu_seqlens`` (documents padded to a multiple of ``2*cp``) so every rank
        gets an equal ``sq``-length shard (the all-gather assumes equal shards).
        """
        import transformer_engine.pytorch  # noqa: F401  (loads the tex C++ extension)
        import transformer_engine_torch as tex

        cu_pad = packed_seq_params.cu_seqlens_q_padded
        if cu_pad is None:
            cu_pad = packed_seq_params.cu_seqlens_q
        cu_pad = cu_pad.to(device=device, dtype=torch.int32).flatten()
        total = sq * cp_size  # full padded sequence length
        idx = tex.thd_get_partitioned_indices(cu_pad, total, cp_size, cp_rank)
        idx = idx.to(device=device, dtype=torch.long)
        assert idx.numel() == sq, (
            f"THD-CP partition gave {idx.numel()} indices for a shard of {sq}; documents must be "
            "padded to a multiple of 2*cp (cu_seqlens_q_padded) for equal CP shards."
        )
        return idx

    def _packed_ids(self, packed_seq_params: PackedSeqParams, q_pos: Tensor, kv_pos: Tensor, cp_size: int):
        """Per-token document id + key-validity for packed (THD) sequences.

        In *global* token coordinates, derives each query/key's document id (from
        the padded slot boundaries ``cu_seqlens_q_padded``) and whether each key
        is a real (non-padding) token (from the unpadded lengths
        ``cu_seqlens_q``). The mask_mod then forbids cross-document and
        padding-key attention. This is layout-general: it works off whatever
        global positions ``q_pos``/``kv_pos`` carry, so it serves both CP=1
        (``q_pos = arange``) and CP>1 (``q_pos`` = this rank's THD-partitioned
        global indices, gathered into ``kv_pos``) — see ``_thd_cp_positions``.

        Assumes a single packed row (``b == 1``, the THD convention); each query
        row shares one pack layout. ``cp_size`` is accepted for signature
        symmetry but not needed (everything is already global).
        """
        cu = packed_seq_params.cu_seqlens_q
        cu_pad = packed_seq_params.cu_seqlens_q_padded
        if cu_pad is None:
            cu_pad = cu
        device = q_pos.device
        cu = cu.to(device=device, dtype=torch.long).flatten()
        cu_pad = cu_pad.to(device=device, dtype=torch.long).flatten()
        n_doc = cu_pad.numel() - 1
        real_len = cu[1:] - cu[:-1]  # [n_doc] real tokens per document
        pad_start = cu_pad[:-1]  # [n_doc] padded start offset of each document

        def doc_of(pos: Tensor) -> Tensor:
            d = torch.searchsorted(cu_pad, pos, right=True) - 1
            return d.clamp_(min=0, max=n_doc - 1)

        doc_q = doc_of(q_pos)
        doc_kv = doc_of(kv_pos)
        valid_kv = (kv_pos - pad_start[doc_kv]) < real_len[doc_kv]
        # real_q / pad_start feed the FFPA packed path (per-document run building);
        # the flex/naive paths only use doc_q, doc_kv, valid_kv.
        real_q = (q_pos - pad_start[doc_q]) < real_len[doc_q]
        return doc_q, doc_kv, valid_kv, real_q, pad_start

    def forward(
        self,
        query: Tensor,  # [sq, b, np, hn]  CP-local query shard
        key: Tensor,  # [sk, b, ng, hn]  CP-local key shard
        value: Tensor,  # [sk, b, ng, hn]  CP-local value shard
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        assert attention_bias is None, "attention_bias not supported."

        # Sliding layers delegate to TE (its own p2p-ring CP + fused windowed kernel). TE
        # consumes the raw sbhd/thd inputs and does the CP KV exchange internally, so we
        # short-circuit before the all-gather/permute path below. Global (hd512) layers and
        # the GEMMA4_SLIDING_CP=allgather rollback fall through to the flex/ffpa all-gather.
        if self._te_sliding is not None:
            return self._te_sliding(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type if attn_mask_type is not None else self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        kernel = os.environ.get("GEMMA4_CP_KERNEL", "flex")

        cp_group = parallel_state.get_context_parallel_group()
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        # Under packing (qkv_format='thd') mcore squeezes the batch dim and passes
        # 3D [t, h, d]; add it back so the sbhd path below is uniform, then restore.
        thd = query.dim() == 3
        if thd:
            query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)

        sq, b, np_, hn = query.shape

        # Gather full-length K/V (differentiable) and their global positions.
        # (record_function scopes below let the PyTorch profiler attribute GPU time
        # to CP comm vs global/sliding attention — see PLAN §27.)
        if cp_size > 1:
            with record_function("cp_allgather_kv"):
                key_full = _AllGatherSeq.apply(key, cp_group)  # [S, b, ng, hn]
                value_full = _AllGatherSeq.apply(value, cp_group)
        else:
            key_full, value_full = key, value

        # Global position of each local query token. Packing uses the THD
        # partition (matching gpt_step); otherwise the whole-sequence zigzag.
        if packed_seq_params is not None and cp_size > 1:
            q_pos = self._thd_cp_positions(packed_seq_params, sq, cp_size, cp_rank, query.device)
        else:
            q_pos = _cp_global_positions(cp_rank, cp_size, sq, query.device)
        if cp_size > 1:
            pos_shards = [torch.empty_like(q_pos) for _ in range(cp_size)]
            dist.all_gather(pos_shards, q_pos.contiguous(), group=cp_group)
            kv_pos = torch.cat(pos_shards, dim=0)
        else:
            kv_pos = q_pos

        ng = key_full.shape[2]

        # Packing (document masking + padding): derive per-token doc ids + key
        # validity from packed_seq_params. Serves CP=1 and CP>1 (positions are
        # already global; see _packed_ids / _thd_cp_positions).
        pack_q = pack_kv = valid_kv = real_q = pad_start = None
        if packed_seq_params is not None:
            pack_q, pack_kv, valid_kv, real_q, pad_start = self._packed_ids(
                packed_seq_params, q_pos, kv_pos, cp_size
            )

        # -> [b, heads, s, hn].
        q = query.permute(1, 2, 0, 3)  # [b, np, sq, hn]
        k = key_full.permute(1, 2, 0, 3)  # [b, ng, S, hn]
        v = value_full.permute(1, 2, 0, 3)
        scale = self.softmax_scale

        # FFPA (head_dim-512 CuTeDSL) for the global layers: ~2.5-4x the flex
        # kernel (PLAN §24-§26). Global layers (full causal, incl. packed via the
        # per-document past/diagonal split) go to FFPA; sliding (windowed) layers
        # stay on flex — FFPA's kernel has no window input — as does any layer when
        # FFPA is unavailable in the venv.
        use_ffpa = (
            kernel == "ffpa"
            and self.window_left is None
            and hn == 512
            and ffpa_dense_available()
        )
        attn_scope = "attn_global" if self.window_left is None else "attn_sliding"
        with record_function(attn_scope):
            if use_ffpa and pack_q is not None:
                context = self._ffpa_forward_packed(
                    q, k, v, q_pos, kv_pos, scale, pack_q, real_q, pad_start
                )
            elif use_ffpa:
                context = self._ffpa_forward(q, k, v, q_pos, kv_pos, scale)
            elif kernel in ("flex", "ffpa"):
                context = self._flex_forward(q, k, v, q_pos, kv_pos, scale, pack_q, pack_kv, valid_kv)
            else:
                # Legacy fallbacks (parity/debug): GQA-expand + explicit [sq, S] mask.
                if np_ // ng > 1:
                    k = k.repeat_interleave(np_ // ng, dim=1)
                    v = v.repeat_interleave(np_ // ng, dim=1)
                allow = self._build_mask(q_pos, kv_pos)  # [sq, S]
                if pack_q is not None:
                    allow = allow & (pack_q[:, None] == pack_kv[None, :])
                if valid_kv is not None:
                    allow = allow & valid_kv[None, :]
                attn_mask = allow[None, None]  # broadcast over [b, heads]
                if kernel == "naive":
                    # Explicit path (fp64 debugging): materializes [sq, S] scores.
                    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                    context = torch.matmul(torch.softmax(scores, dim=-1), v)  # [b, np, sq, hn]
                else:
                    context = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, scale=scale
                    )  # [b, np, sq, hn]

        # -> [sq, b, np, hn] -> [sq, b, hp]
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(sq, b, np_ * hn)
        if thd:
            context = context.squeeze(1)  # thd expects [t, hp]
        return context
