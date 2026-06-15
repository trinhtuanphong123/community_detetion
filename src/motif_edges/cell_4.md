


# ============================================================
# Cell 4: Standardize Schema and Execute Quality Checks
# ============================================================

def standardize_transaction_schema(
    df_raw: pl.DataFrame,
    raw_to_internal_cols: Dict[str, str],
    matcher_columns: List[str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Standardize raw transaction dataframe into internal edge schema.

    Input raw schema expected:
        edge_id, step, type, amount, nameOrig, nameDest, isSAR, ...

    Internal schema:
        edge_id, src, dst, step, amount, is_sar

    Returns:
        df_full: full dataframe with renamed columns and metadata preserved.
        df_edges: minimal edge dataframe for motif matching.
    """
    required_raw_cols = [
        "edge_id",
        "step",
        "amount",
        "nameOrig",
        "nameDest",
        "isSAR",
    ]

    missing_cols = [c for c in required_raw_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required raw columns: {missing_cols}. "
            f"Available columns: {df_raw.columns}"
        )

    df_full = (
        df_raw
        .rename(raw_to_internal_cols)
        .with_columns([
            pl.col("edge_id").cast(pl.UInt32),
            pl.col("src").cast(pl.Int64),
            pl.col("dst").cast(pl.Int64),
            pl.col("step").cast(pl.Int64),
            pl.col("amount").cast(pl.Float64),
            pl.col("is_sar").cast(pl.Int8),
        ])
        .sort(["step", "edge_id"])
    )

    missing_internal_cols = [c for c in matcher_columns if c not in df_full.columns]
    if missing_internal_cols:
        raise ValueError(
            f"Missing internal matcher columns after standardization: {missing_internal_cols}"
        )

    df_edges = df_full.select(matcher_columns)

    return df_full, df_edges


# Standardize raw transaction dataframe loaded in Cell 3
df_full, df_edges = standardize_transaction_schema(
    df_raw=df_raw,
    raw_to_internal_cols=RAW_TO_INTERNAL_COLS,
    matcher_columns=MATCHER_COLUMNS,
)

print("Schema standardization completed.")
print("df_full shape:", df_full.shape)
print("df_edges shape:", df_edges.shape)

print("\nInternal matcher schema:")
print(df_edges.schema)

print("\nHead of df_edges:")
display(df_edges.head(10))


# ------------------------------------------------------------
# Core sanity checks
# ------------------------------------------------------------

schema_summary = df_edges.select([
    pl.len().alias("num_edges"),
    pl.col("edge_id").n_unique().alias("n_unique_edges"),
    pl.col("src").n_unique().alias("n_unique_src_nodes"),
    pl.col("dst").n_unique().alias("n_unique_dst_nodes"),
    pl.concat([pl.col("src"), pl.col("dst")]).n_unique().alias("n_unique_all_nodes"),
    pl.col("step").min().alias("min_step"),
    pl.col("step").max().alias("max_step"),
    pl.col("amount").min().alias("min_amount"),
    pl.col("amount").max().alias("max_amount"),
    pl.col("amount").mean().alias("mean_amount"),
    pl.col("is_sar").sum().alias("num_sar_edges"),
    pl.col("is_sar").mean().alias("sar_rate"),
])

print("\nSchema summary:")
display(schema_summary)


# ------------------------------------------------------------
# Data quality checks
# ------------------------------------------------------------

duplicate_edge_ids = (
    df_edges
    .group_by("edge_id")
    .len()
    .filter(pl.col("len") > 1)
)

self_loops = df_edges.filter(pl.col("src") == pl.col("dst"))

non_positive_amount = df_edges.filter(pl.col("amount") <= 0)

null_check = df_edges.select([
    pl.col(c).null_count().alias(f"{c}_nulls")
    for c in df_edges.columns
])

print("\nData quality checks:")
print("Duplicate edge_id rows:", duplicate_edge_ids.height)
print("Self-loop rows:", self_loops.height)
print("Non-positive amount rows:", non_positive_amount.height)

print("\nNull check:")
display(null_check)

if duplicate_edge_ids.height > 0:
    print("\nWarning: duplicate edge_id detected. First 10 duplicated IDs:")
    display(duplicate_edge_ids.head(10))

if self_loops.height > 0:
    print("\nWarning: self-loops detected. First 10 rows:")
    display(self_loops.head(10))

if non_positive_amount.height > 0:
    print("\nWarning: non-positive amount detected. First 10 rows:")
    display(non_positive_amount.head(10))


# ------------------------------------------------------------
# Optional metadata dataframe for later joins
# ------------------------------------------------------------

EDGE_METADATA_COLUMNS = [
    c for c in [
        "edge_id",
        "type",
        "bankOrig",
        "bankDest",
        "daysInBankOrig",
        "daysInBankDest",
        "phoneChangesOrig",
        "phoneChangesDest",
        "oldbalanceOrig",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "alertID",
        "modelType",
    ]
    if c in df_full.columns
]

df_edge_metadata = df_full.select(EDGE_METADATA_COLUMNS)

print("\ndf_edge_metadata shape:", df_edge_metadata.shape)
print("Metadata columns:", EDGE_METADATA_COLUMNS)

print("\nCell 4 completed.")

# Cell 4 — at the very end, after all dataframes are derived
del df_raw
del df_full
gc.collect()
print("df_raw and df_full released.")
