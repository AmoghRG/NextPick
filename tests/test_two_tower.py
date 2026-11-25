"""Unit and ONNX parity tests for Stage-1 Two-Tower retrieval model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.models.two_tower import (
    ItemTower,
    UserTower,
    export_user_tower_onnx,
    in_batch_logq_loss,
)


def test_towers_output_l2_normalized() -> None:
    item_tower = ItemTower()
    user_tower = UserTower()

    item_emb = item_tower(
        text_vec=torch.randn(4, 384),
        group_id=torch.randint(0, 10, (4,)),
        dept_id=torch.randint(0, 20, (4,)),
        colour_id=torch.randint(0, 10, (4,)),
        price_bucket=torch.randint(0, 10, (4,)),
    )
    assert item_emb.shape == (4, 128)
    norms = torch.linalg.norm(item_emb, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    user_emb = user_tower(
        hist_item_vecs=torch.randn(4, 20, 128),
        hist_mask=torch.ones(4, 20),
        channel_id=torch.randint(0, 3, (4,)),
    )
    assert user_emb.shape == (4, 128)
    u_norms = torch.linalg.norm(user_emb, dim=-1)
    assert torch.allclose(u_norms, torch.ones(4), atol=1e-5)


def test_in_batch_logq_loss_handles_collisions() -> None:
    torch.manual_seed(42)
    u = torch.nn.functional.normalize(torch.randn(6, 128), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(6, 128), dim=-1)
    # Force duplicate item_id at indices 0 and 3
    item_ids = torch.tensor([10, 20, 30, 10, 50, 60])
    log_q = torch.zeros(6)

    loss = in_batch_logq_loss(u, v, item_ids, log_q, tau=0.07, apply_logq=True)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_user_tower_onnx_export_parity(tmp_path: Path) -> None:
    torch.manual_seed(0)
    user_tower = UserTower().eval()
    onnx_path = tmp_path / "user_tower.onnx"
    export_user_tower_onnx(user_tower, onnx_path)
    assert onnx_path.exists()

    hist_vecs = torch.randn(3, 20, 128, dtype=torch.float32)
    hist_mask = torch.ones(3, 20, dtype=torch.float32)
    hist_mask[1, 5:] = 0.0
    chan = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        pt_out = user_tower(hist_vecs, hist_mask, chan).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["user_emb"],
        {
            "hist_item_vecs": hist_vecs.numpy(),
            "hist_mask": hist_mask.numpy(),
            "channel_id": chan.numpy(),
        },
    )[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    assert max_diff == pytest.approx(0.0, abs=1e-5)
