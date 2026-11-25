"""Cold-start-safe PyTorch Two-Tower retrieval model with recency decay, collision masking, and logQ correction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    """Encodes item text + categorical side features into an L2-normalized 128-d vector."""

    def __init__(
        self,
        text_dim: int = 384,
        num_groups: int = 32,
        num_depts: int = 512,
        num_colours: int = 64,
        num_price_buckets: int = 12,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.group_emb = nn.Embedding(num_groups, 16)
        self.dept_emb = nn.Embedding(num_depts, 32)
        self.colour_emb = nn.Embedding(num_colours, 16)
        self.price_emb = nn.Embedding(num_price_buckets, 16)

        in_dim = text_dim + 16 + 32 + 16 + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    def forward(
        self,
        text_vec: torch.Tensor,
        group_id: torch.Tensor,
        dept_id: torch.Tensor,
        colour_id: torch.Tensor,
        price_bucket: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                text_vec,
                self.group_emb(group_id),
                self.dept_emb(dept_id),
                self.colour_emb(colour_id),
                self.price_emb(price_bucket),
            ],
            dim=-1,
        )
        return F.normalize(self.mlp(x), p=2, dim=-1)


class UserTower(nn.Module):
    """Encodes a user's last-K item vectors via exponential recency-decay pooling."""

    def __init__(
        self,
        item_dim: int = 128,
        num_channels: int = 4,
        embed_dim: int = 128,
        decay_lambda: float = 0.15,
        max_seq_len: int = 20,
    ) -> None:
        super().__init__()
        self.channel_emb = nn.Embedding(num_channels, 8)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer("decay_weights", torch.exp(-decay_lambda * positions))

        self.mlp = nn.Sequential(
            nn.Linear(item_dim + 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(
        self,
        hist_item_vecs: torch.Tensor,
        hist_mask: torch.Tensor,
        channel_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hist_item_vecs: (B, L, item_dim) ordered most-recent first (index 0 = newest).
            hist_mask: (B, L) float/bool mask where 1.0 = valid item, 0.0 = padding.
            channel_id: (B,) integer sales channel ID.
        """
        seq_len = hist_item_vecs.size(1)
        weights = self.decay_weights[:seq_len].unsqueeze(0) * hist_mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (hist_item_vecs * weights.unsqueeze(-1)).sum(dim=1) / denom

        x = torch.cat([pooled, self.channel_emb(channel_id)], dim=-1)
        return F.normalize(self.mlp(x), p=2, dim=-1)


def in_batch_logq_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    item_ids: torch.Tensor,
    log_q: torch.Tensor,
    tau: float = 0.07,
    apply_logq: bool = True,
) -> torch.Tensor:
    """Compute temperature-scaled in-batch softmax loss with collision masking and optional logQ correction."""
    batch_size = user_emb.size(0)
    logits = (user_emb @ item_emb.T) / tau
    off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)

    if apply_logq:
        logits = logits - off_diag.float() * log_q.unsqueeze(0)

    same_item = (item_ids.unsqueeze(1) == item_ids.unsqueeze(0)) & off_diag
    logits = logits.masked_fill(same_item, -1e9)

    labels = torch.arange(batch_size, device=logits.device)
    return F.cross_entropy(logits, labels)


def export_user_tower_onnx(
    user_tower: UserTower,
    out_path: Path,
    item_dim: int = 128,
    max_seq_len: int = 20,
) -> None:
    """Export UserTower to ONNX (opset 17) with dynamic batch size."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_tower_cpu = user_tower.eval().cpu()
    dummy_vecs = torch.randn(1, max_seq_len, item_dim, dtype=torch.float32)
    dummy_mask = torch.ones(1, max_seq_len, dtype=torch.float32)
    dummy_chan = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        user_tower_cpu,
        (dummy_vecs, dummy_mask, dummy_chan),
        str(out_path),
        input_names=["hist_item_vecs", "hist_mask", "channel_id"],
        output_names=["user_emb"],
        dynamic_axes={
            "hist_item_vecs": {0: "batch"},
            "hist_mask": {0: "batch"},
            "channel_id": {0: "batch"},
            "user_emb": {0: "batch"},
        },
        opset_version=17,
    )
