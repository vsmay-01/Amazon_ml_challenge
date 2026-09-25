"""
graph_reasoning.py -- Innovation 3: Graph Attention as an EXPERIMENTAL module.

Per Section 8: this branch is optional and must survive the mandatory GAT
ablation (Section 8.1) before it can be part of the final pipeline. It is
implemented as an isolated, opt-in component so the rest of the pipeline
(blocking, features, gradient-boosted model, calibration, decision policy)
runs completely independently of it.

If PyTorch is not installed, `GAT_AVAILABLE` is False and
`GraphAttentionScorer` raises a clear ImportError only when actually
instantiated -- importing this module never fails and never silently
degrades the rest of the pipeline.

Initial hyperparameters (candidates, not requirements, per Section 8):
  layers=2, heads<=8, hidden_dim=128, out_dim=64, LeakyReLU, dropout~0.3
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    GAT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    GAT_AVAILABLE = False


@dataclass
class GraphConfig:
    layers: int = 2
    heads: int = 8
    hidden_dim: int = 128
    out_dim: int = 64
    dropout: float = 0.3


def build_local_graph(
    source1_id: str,
    candidate_rows: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """
    Build a local graph G_q = (V_q, E_q) for one Source 1 query, per
    Section 7.1: V_q = {q} + C2(q) + C3(q).

    Returns (node_ids, edge_feature_matrix) where node_ids[0] is the query
    node and edge_feature_matrix rows correspond one-to-one with
    candidate_rows (edges from the query node to each candidate).
    Same-side (S2-S2 or S3-S3) edges are intentionally omitted: only
    query<->candidate and cross-source (S2<->S3, via `cross_support*`
    features already computed upstream) evidence is used.
    """
    node_ids = [source1_id] + candidate_rows["candidate_id"].tolist()
    edge_cols = [c for c in candidate_rows.columns if c not in (
        "source1_id", "candidate_source", "candidate_id", "label"
    )]
    edge_features = candidate_rows[edge_cols].fillna(0.0).to_numpy(dtype=float)
    return node_ids, edge_features


if GAT_AVAILABLE:

    class _GATLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
            super().__init__()
            self.heads = heads
            self.out_dim = out_dim
            self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
            self.a = nn.Parameter(torch.empty(heads, 2 * out_dim))
            nn.init.xavier_uniform_(self.a)
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(dropout)

        def forward(self, h: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            # h: [N, in_dim]; edge_index: [2, E] (src, dst)
            N = h.size(0)
            Wh = self.W(h).view(N, self.heads, self.out_dim)  # [N, heads, out_dim]
            src, dst = edge_index
            wh_src, wh_dst = Wh[src], Wh[dst]  # [E, heads, out_dim]
            cat = torch.cat([wh_src, wh_dst], dim=-1)  # [E, heads, 2*out_dim]
            e = self.leaky_relu((cat * self.a).sum(dim=-1))  # [E, heads]

            attn = torch.zeros_like(e)
            for head in range(self.heads):
                # segment softmax over destination nodes
                e_head = e[:, head]
                exp = torch.exp(e_head - e_head.max())
                denom = torch.zeros(N, device=h.device).index_add_(0, dst, exp) + 1e-9
                attn[:, head] = exp / denom[dst]
            attn = self.dropout(attn)

            out = torch.zeros(N, self.heads, self.out_dim, device=h.device)
            for head in range(self.heads):
                weighted = wh_src[:, head, :] * attn[:, head].unsqueeze(-1)
                out[:, head, :] = out[:, head, :].index_add(0, dst, weighted)
            return out.reshape(N, self.heads * self.out_dim)

    class GraphAttentionScorer(nn.Module):
        """
        Minimal multi-head GAT following Eq. 9-10 of the design doc, used
        purely as an experimental branch for the mandatory ablation
        (Section 8.1). Not wired into the default pipeline.
        """

        def __init__(self, in_dim: int, config: GraphConfig | None = None):
            super().__init__()
            cfg = config or GraphConfig()
            self.layer1 = _GATLayer(in_dim, cfg.hidden_dim, cfg.heads, cfg.dropout)
            self.layer2 = _GATLayer(cfg.hidden_dim * cfg.heads, cfg.out_dim, 1, cfg.dropout)
            self.scorer = nn.Linear(cfg.out_dim * 2, 1)

        def forward(self, h: "torch.Tensor", edge_index: "torch.Tensor", pair_index: "torch.Tensor") -> "torch.Tensor":
            h1 = F.elu(self.layer1(h, edge_index))
            h2 = self.layer2(h1, edge_index)
            src, dst = pair_index
            pair_repr = torch.cat([h2[src], h2[dst]], dim=-1)
            return torch.sigmoid(self.scorer(pair_repr)).squeeze(-1)

else:

    class GraphAttentionScorer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is not installed. GraphAttentionScorer is an optional, "
                "experimental branch (Innovation 3) -- install torch to run "
                "the GAT ablation, or skip it: the rest of the pipeline does "
                "not depend on this module."
            )
