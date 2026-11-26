# Dataset, Preprocessing & Leakage Controls

NextPick trains and evaluates on an **8-week chronological slice** of the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset (`transactions_train.csv` and `articles.csv`).

---

## 1. Split Scale & Window Boundaries

Anchored on `max_date = MAX(t_dat)` in `transactions_train.csv` (`56` days total):

| Split | Date Predicate | Transactions | Role |
| :--- | :--- | :---: | :--- |
| **Train (Weeks 1–6)** | `t_dat >= max_date - 55d AND t_dat < max_date - 13d` | `1,678,323` | Two-Tower, Popularity, ItemKNN co-visitation |
| **Validation (Week 7)** | `t_dat >= max_date - 13d AND t_dat < max_date - 6d` | `255,241` | LightGBM LambdaRank training (`<= 30,000` queries) |
| **Test (Week 8)** | `t_dat >= max_date - 6d` | `240,311` | Unmodified future-horizon evaluation (`68,984` users) |
| **Active Catalog** | Distinct `article_id` with `t_dat < max_date - 6d` | `35,576` | Pre-Week-8 active SKU index |

---

## 2. Data Dictionary (Actual Pipeline Fields)

| Field | Type | Description | Used By |
| :--- | :--- | :--- | :--- |
| `customer_id` | `VARCHAR` | Anonymized customer identifier | DuckDB `PARTITION BY`, eval grouping |
| `article_id` | `VARCHAR` | Product SKU identifier | Item Tower, Qdrant point ID (UUIDv5), Stage-2 |
| `t_dat` | `DATE` | Calendar date of transaction | Temporal split & `ORDER BY t_dat, row_id` |
| `row_id` | `BIGINT` | Deterministic `ROW_NUMBER()` tie-breaker | Intra-day ordering in DuckDB |
| `price` | `DOUBLE` | Normalized transaction price | `user_avg_price`, `price_abs_diff`, `price_zscore` |
| `sales_channel_id` | `INTEGER` | Channel (`1` = store, `2` = online) | User Tower `channel_emb`, Stage-2 feature |
| `prod_name` | `VARCHAR` | Product display title | MiniLM text input, UI display |
| `product_type_name` | `VARCHAR` | Fine product type (e.g., `Trousers`) | MiniLM text input |
| `product_group_name` | `VARCHAR` | Coarse group (e.g., `Garment Lower body`) | Item Tower `group_emb`, Qdrant filter, Stage-2 |
| `department_name` | `VARCHAR` | Department name | Item Tower `dept_emb`, MiniLM text, Stage-2 |
| `colour_group_name` | `VARCHAR` | Colour group | Item Tower `colour_emb` |
| `detail_desc` | `VARCHAR` | Product description text | MiniLM text input |
| `avg_price` | `DOUBLE` | Catalog price aggregate | Stage-2 `item_price`, Qdrant `price` payload |
| `train_tx_count` | `BIGINT` | Weeks 1–6 training interaction count | `logQ` prior, `item_log_popularity`, cold-start |

---

## 3. Leakage Prevention & Known Pipeline Differences

1. **Strict Causal Rolling Windows:**
   - Both [`src/data/temporal_split.py`](../src/data/temporal_split.py) and [`notebooks/kaggle_train_pipeline.py`](../notebooks/kaggle_train_pipeline.py) build `hist_article_ids`, `hist_prices`, `hist_groups`, and `hist_departments` using:
     ```sql
     WINDOW w AS (
         PARTITION BY customer_id
         ORDER BY t_dat, row_id
         ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
     )
     ```
   - `1 PRECEDING` guarantees the target row is never included in its own history.
2. **Pre-Week-8 Catalog Restriction:**
   - `kaggle_train_pipeline.py` builds `catalog.parquet` exclusively from SKUs observed before Week 8 (`t_dat < val_end`), preventing future Week-8 catalog leakage.
   - `avg_price` and `train_tx_count` are both restricted to `t_dat < train_end` in the current `kaggle_train_pipeline.py` (note: the exported v6 Kaggle artifacts used `t_dat < val_end` for `avg_price`).
3. **Reusable Splitter vs. Authoritative Kaggle Pipeline:**
   - [`src/data/temporal_split.py`](../src/data/temporal_split.py) keeps all `raw_art` rows in `catalog.parquet` and is used for fast unit/integration tests, whereas [`notebooks/kaggle_train_pipeline.py`](../notebooks/kaggle_train_pipeline.py) filters to active pre-Week-8 SKUs (`35,576`) and is the authoritative training pipeline.
