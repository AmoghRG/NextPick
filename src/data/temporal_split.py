"""DuckDB-powered 8-week temporal split and point-in-time rolling feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for the 8-week H&M temporal split."""

    total_weeks: int = 8
    train_weeks: int = 6
    val_weeks: int = 1
    history_window: int = 20


def build_temporal_split(
    transactions_path: Path,
    articles_path: Path,
    out_dir: Path,
    cfg: SplitConfig = SplitConfig(),
) -> dict[str, int]:
    """Slice transactions to the last 8 weeks, split temporally, and compute point-in-time histories.

    Args:
        transactions_path: Path to H&M transactions CSV or Parquet.
        articles_path: Path to H&M articles CSV or Parquet.
        out_dir: Output directory for train/val/test/catalog Parquet files.
        cfg: Temporal split and rolling window configuration.

    Returns:
        Row counts for train, val, test, and catalog tables.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")

    tx_scan = _scan_expr(transactions_path)
    art_scan = _scan_expr(articles_path)

    con.execute(f"CREATE TABLE raw_tx AS SELECT * FROM {tx_scan}")
    con.execute(f"CREATE TABLE raw_art AS SELECT * FROM {art_scan}")

    max_date = con.execute("SELECT MAX(CAST(t_dat AS DATE)) FROM raw_tx").fetchone()[0]
    if max_date is None:
        raise ValueError("Transactions dataset is empty.")

    total_days = cfg.total_weeks * 7
    train_days = cfg.train_weeks * 7
    val_days = cfg.val_weeks * 7

    # Anchor boundaries cleanly on dates:
    # [max_date - 55d, max_date] = 56 days (8 weeks)
    con.execute(
        f"""
        CREATE TABLE windowed_tx AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY CAST(t.t_dat AS DATE), t.customer_id, t.article_id) AS row_id,
            CAST(t.t_dat AS DATE) AS t_dat,
            CAST(t.customer_id AS VARCHAR) AS customer_id,
            CAST(t.article_id AS VARCHAR) AS article_id,
            CAST(t.price AS DOUBLE) AS price,
            COALESCE(CAST(t.sales_channel_id AS INTEGER), 1) AS sales_channel_id,
            COALESCE(CAST(a.product_type_name AS VARCHAR), 'unknown') AS product_type_name,
            COALESCE(CAST(a.product_group_name AS VARCHAR), 'unknown') AS product_group_name,
            COALESCE(CAST(a.department_name AS VARCHAR), 'unknown') AS department_name
        FROM raw_tx t
        LEFT JOIN raw_art a ON CAST(t.article_id AS VARCHAR) = CAST(a.article_id AS VARCHAR)
        WHERE CAST(t.t_dat AS DATE) >= DATE '{max_date}' - INTERVAL {total_days - 1} DAY
        """
    )

    # Point-in-time rolling window strictly prior to current event (1 PRECEDING)
    # Note: ordering by (t_dat, row_id) guarantees deterministic intra-day causal order.
    w = cfg.history_window
    con.execute(
        f"""
        CREATE TABLE enriched_tx AS
        SELECT
            *,
            COALESCE(LIST(article_id) OVER w, []) AS hist_article_ids,
            COALESCE(LIST(price) OVER w, []) AS hist_prices,
            COALESCE(LIST(product_group_name) OVER w, []) AS hist_groups,
            COALESCE(LIST(department_name) OVER w, []) AS hist_departments
        FROM windowed_tx
        WINDOW w AS (
            PARTITION BY customer_id
            ORDER BY t_dat, row_id
            ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING
        )
        """
    )

    train_end = f"DATE '{max_date}' - INTERVAL {total_days - train_days - 1} DAY"
    val_end = f"DATE '{max_date}' - INTERVAL {total_days - train_days - val_days - 1} DAY"

    counts: dict[str, int] = {}
    splits = {
        "train": f"t_dat < {train_end}",
        "val": f"t_dat >= {train_end} AND t_dat < {val_end}",
        "test": f"t_dat >= {val_end}",
    }
    for name, predicate in splits.items():
        target = out_dir / f"{name}.parquet"
        con.execute(f"COPY (SELECT * FROM enriched_tx WHERE {predicate}) TO '{target}' (FORMAT PARQUET)")
        counts[name] = con.execute(f"SELECT COUNT(*) FROM enriched_tx WHERE {predicate}").fetchone()[0]

    catalog_path = out_dir / "catalog.parquet"
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(a.article_id AS VARCHAR) AS article_id,
                COALESCE(CAST(a.prod_name AS VARCHAR), '') AS prod_name,
                COALESCE(CAST(a.product_type_name AS VARCHAR), 'unknown') AS product_type_name,
                COALESCE(CAST(a.product_group_name AS VARCHAR), 'unknown') AS product_group_name,
                COALESCE(CAST(a.colour_group_name AS VARCHAR), 'unknown') AS colour_group_name,
                COALESCE(CAST(a.department_name AS VARCHAR), 'unknown') AS department_name,
                COALESCE(CAST(a.detail_desc AS VARCHAR), '') AS detail_desc,
                COALESCE(AVG(w.price), 0.0) AS avg_price,
                COUNT(w.article_id) AS train_tx_count
            FROM raw_art a
            LEFT JOIN enriched_tx w
                ON CAST(a.article_id AS VARCHAR) = w.article_id
                AND w.t_dat < {train_end}
            GROUP BY 1, 2, 3, 4, 5, 6, 7
        ) TO '{catalog_path}' (FORMAT PARQUET)
        """
    )
    counts["catalog"] = con.execute(f"SELECT COUNT(*) FROM '{catalog_path}'").fetchone()[0]
    con.close()
    return counts


def _scan_expr(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return f"read_parquet('{path}')"
    if suffix == ".csv":
        return f"read_csv_auto('{path}', header=True)"
    raise ValueError(f"Unsupported file format: {path}")
