# ============================================================================
# 1. DATA LOADING + PREPROCESSING (từ Cell 2)
# ============================================================================

def load_and_preprocess(cfg: Config) -> tuple:
    """
    Tải dữ liệu vào GPU VRAM, chuẩn hóa cột, mã hóa node thành int32.

    Returns:
        df (cudf.DataFrame): Dữ liệu đã xử lý [step, src, dst, amount, is_laundering]
        NODE_MAP (cudf.DataFrame): Bảng ánh xạ [node_name, node_id]
        n_alert (int), n_rows (int)
    """
    print("=" * 60)
    print("  BƯỚC 0: Tải dữ liệu vào GPU VRAM")
    print("=" * 60)

    if os.path.isdir(cfg.DATA_PATH):
        df = cudf.read_parquet(cfg.DATA_PATH + "/*.parquet")
        print(f"  Đã tải thư mục parquet.")
    else:
        df = cudf.read_csv(cfg.DATA_PATH)
        print(f"  Đã tải file CSV.")

    # ── Chuẩn hóa tên cột ──
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_").replace(".", "_")
        if   "time"    in cl or "step" in cl:   col_map[col] = "step"
        elif cl in ("from", "account", "sender", "from_id", "nameorig"):
                                                col_map[col] = "src"
        elif cl in ("to", "account_1", "receiver", "to_id", "namedest"):
                                                col_map[col] = "dst"
        elif "amount"  in cl:                   col_map[col] = "amount"
        elif "laundering" in cl or "label" in cl or "fraud" in cl or "issar" in cl:
                                                col_map[col] = "is_laundering"

    df = df.rename(columns=col_map)

    required = {"step", "src", "dst", "amount", "is_laundering"}
    missing = required - set(df.columns)
    assert not missing, f"Could not map columns: {missing}\n  Available: {list(df.columns)}"

    # ── Ép kiểu ──
    print("  Ép kiểu dữ liệu để tối ưu VRAM...")
    df["step"]          = df["step"].astype('int32')
    df["amount"]        = df["amount"].astype('float32')
    df["is_laundering"] = df["is_laundering"].astype('int8')

    # ── Global Node Encoding ──
    print("  Xây dựng Global Node Encoding...")

    all_node_names = cudf.concat([df["src"], df["dst"]]).unique()
    n_unique = len(all_node_names)

    NODE_MAP = cudf.DataFrame({
        "node_name": all_node_names,
        "node_id":   cudf.Series(range(n_unique), dtype="int32"),
    })

    df = df.merge(NODE_MAP.rename(columns={"node_name": "src", "node_id": "src_id"}),
                  on="src", how="left")
    df = df.merge(NODE_MAP.rename(columns={"node_name": "dst", "node_id": "dst_id"}),
                  on="dst", how="left")

    df = df.drop(columns=["src", "dst"]).rename(columns={"src_id": "src", "dst_id": "dst"})

    print(f"  Đã mã hóa {n_unique:,} node duy nhất thành int32.")

    # ── Summary ──
    n_steps = df["step"].nunique()
    n_alert = df["is_laundering"].sum()
    n_rows  = len(df)

    print(f"\n  Dataset summary (On GPU)")
    print(f"    Rows        : {n_rows:>12,}")
    print(f"    Unique steps: {n_steps:>11,}")
    print(f"    Unique nodes: {n_unique:>11,}")
    print(f"    Alert edges : {n_alert:>11,}  ({(n_alert/n_rows)*100:.2f}%)")
    print(f"    Step range  : {df['step'].min()} – {df['step'].max()}")

    gc.collect()
    print("  ✅ Dữ liệu đã sẵn sàng trên VRAM.\n")

    return df, NODE_MAP, int(n_alert), int(n_rows)