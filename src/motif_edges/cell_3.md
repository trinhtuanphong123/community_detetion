

# ============================================================
# Cell 3: Load Raw Parquet with Polars
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Input parquet file not found: {DATA_PATH}")

load_start = time.time()
df_raw = pl.read_parquet(DATA_PATH)
load_end = time.time()

print("Loaded raw transaction dataframe.")
print("Shape:", df_raw.shape)
print("Schema:")
print(df_raw.schema)
print("Load time:", round(load_end - load_start, 3), "seconds")

# Validate that required columns are in df_raw
missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df_raw.columns]
if missing_cols:
    raise ValueError(
        f"Missing required raw columns: {missing_cols}. "
        f"Available columns: {df_raw.columns}"
    )
print("All required raw columns are present.")
print("\nCell 3 completed.")

