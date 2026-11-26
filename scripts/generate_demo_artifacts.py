"""Generate lightweight demo artifacts in ~2 seconds so anyone can boot NextPick without Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.two_tower import UserTower, export_user_tower_onnx


def generate_demo_artifacts(out_dir: Path, num_items: int = 500, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    groups = ["Garment Upper body", "Garment Lower body", "Socks & Tights"]
    depts = ["Jersey Basic", "Trousers", "Socks", "Knitwear", "Denim"]
    colours = ["Black", "White", "Blue", "Beige", "Grey", "Green"]
    types = ["Hoodie", "Trousers", "T-shirt", "Jeans", "Socks"]

    rows = []
    for i in range(num_items):
        rows.append(
            {
                "article_id": f"0{700000000 + i}",
                "prod_name": f"NextPick {types[i % len(types)]} #{i + 1}",
                "product_type_name": types[i % len(types)],
                "product_group_name": groups[i % len(groups)],
                "colour_group_name": colours[i % len(colours)],
                "department_name": depts[i % len(depts)],
                "detail_desc": f"Demo fashion SKU {i + 1}",
                "avg_price": round(float(0.015 + 0.005 * (i % 10)), 4),
                "train_tx_count": int(rng.integers(10, 500)),
            }
        )

    cat_df = pd.DataFrame(rows)
    cat_df.to_parquet(out_dir / "catalog.parquet", index=False)

    vecs = rng.normal(size=(num_items, 128)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    emb_df = pd.DataFrame({"article_id": cat_df["article_id"], "embedding": vecs.tolist()})
    emb_df.to_parquet(out_dir / "item_embeddings.parquet", index=False)

    export_user_tower_onnx(UserTower().eval(), out_dir / "user_tower.onnx")
    (out_dir / "demo_manifest.json").write_text(
        json.dumps(
            {
                "is_synthetic_demo": True,
                "model_origin": "synthetic_random_untrained",
                "num_items": num_items,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo artifacts for local 60-second boot.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num-items", type=int, default=500)
    args = parser.parse_args()
    generate_demo_artifacts(args.out_dir, num_items=args.num_items)
    print(f"Generated {args.num_items} demo SKUs + user_tower.onnx in {args.out_dir}")


if __name__ == "__main__":
    main()
