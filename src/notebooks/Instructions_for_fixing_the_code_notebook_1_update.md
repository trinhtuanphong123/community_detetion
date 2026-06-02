Your fix plan is mostly correct and technically coherent. It addresses the real Notebook 1 risks: target leakage through graph shards, overlapping temporal windows, missing `event_id`, duplicated `build_second_order_edges`, and schema mismatch between Notebook 1 and Notebook 2. The plan is also appropriately “minimal-impact” because it keeps the existing core artifact names and only changes the unsafe schemas. 

The main thing I would change is not the direction, but a few details to prevent silent breakage.

## Verdict

I would approve the plan with revisions.

The strongest parts are:

You correctly identify that `temporal_edges` and `second_order_edges` currently leak labels through `alert_1`, `alert_2`, and `n_alert`. You also correctly preserve `is_sar` in `transactions.parquet` because Notebook 2 still needs a label source. That separation is the right design.

You correctly keep `TEMPORAL_DIR = OUTPUT_DIR / "temporal_edges"` and `SECOND_ORDER_DIR = OUTPUT_DIR / "second_order_edges"` for feature-safe shards. That reduces downstream breakage.

You correctly add debug-labeled directories instead of deleting label diagnostics entirely. That is useful for analysis, but keeps the modeling contract clean.

You correctly keep global `NodeEncoder` unchanged. Global node encoding is acceptable if `node_id` is treated as a join key only, not as a feature.

## Problems or refinements needed

### 1. Do not add `event_id` to `LoaderConfig.dtypes` blindly

Your plan says to add `event_id` to `LoaderConfig.keep_cols` and `LoaderConfig.dtypes`. Adding to `keep_cols` is correct. Adding to `dtypes` is only correct if `event_id` exists before `_cast_dtypes()` runs.

In the current flow, `_cast_dtypes()` happens before the final sorting and before you would add `event_id`. If you add `"event_id": "int64"` to `dtypes`, it will simply be skipped because the column does not exist yet. That is not harmful, but it is misleading.

Better:

```python
# Step 9 — deterministic sort
df = df.sort_values(["step", "src_node", "dst_node", "amount"]).reset_index(drop=True)

# Step 10 — stable event id
df["event_id"] = np.arange(len(df), dtype=np.int64)

# Step 11 — final column order
return df[["event_id", "src_node", "dst_node", "amount", "step", "is_sar"]]
```

Then add `event_id` to `keep_cols`, but `dtypes` is optional. If you add it to `dtypes`, add a comment that it is assigned after casting.

### 2. Deterministic sorting by amount may still not uniquely order duplicate transactions

Sorting by:

```python
["step", "src_node", "dst_node", "amount"]
```

is deterministic only if the source file read order is stable, but identical duplicate rows will still be arbitrarily ordered relative to each other. That usually does not matter if `event_id` only needs to be unique and stable within one run. But if you want reproducible `event_id` across repeated raw file loads, you should preserve original row position first:

```python
df["_raw_row_id"] = np.arange(len(df), dtype=np.int64)
df = df.sort_values(
    ["step", "src_node", "dst_node", "amount", "_raw_row_id"]
).reset_index(drop=True)
df["event_id"] = np.arange(len(df), dtype=np.int64)
df = df.drop(columns=["_raw_row_id"])
```

This avoids unstable ordering among exact duplicates.

### 3. Decide clearly whether `_gap` stays `_gap` or becomes `gap`

Your plan keeps `_gap` in feature temporal edges. That is good for minimal compatibility because existing second-order code already uses `_gap`.

However, if you introduce debug builders and Notebook 2 updates, using `gap` without underscore is cleaner. The risk is schema mismatch.

For minimal-impact, I recommend keeping `_gap` for now:

```text
event_id_1, src_1, dst_1, step_1, amount_1,
event_id_2, src_2, dst_2, step_2, amount_2,
_gap
```

Then make every new function use `_gap`. Do not mix `_gap` and `gap` across cells.

### 4. Keep function names stable, but use explicit debug function names

Your plan says “second function or labeled flag.” I recommend a second function, not a `labeled: bool=False` flag.

Reason: a flag makes it easier to accidentally save labeled outputs into feature folders. Separate functions make the leakage boundary obvious.

Use:

```python
build_temporal_edges(...)
build_temporal_edges_debug(...)

build_second_order_edges(...)
build_second_order_edges_debug(...)
```

Where the normal names produce feature-safe artifacts. This keeps Notebook 2 compatible because it can still call/read the normal directories.

### 5. If you keep `DELTA_W = 5` with `WINDOW_SIZE = 7`, document the choice

Your plan says keep `DELTA_W` as-is unless aligning to 2. This is acceptable, but there is a modeling implication.

With `WINDOW_SIZE = 7` and `DELTA_W = 5`, you allow relays spanning most of the window. That is not wrong. It is simply less strict. If the AML motif logic is supposed to capture short-hop behavior, `DELTA_W = 2` is cleaner. If you want minimal behavioral change, keep `5`.

I would choose:

```python
WINDOW_SIZE = 7
WINDOW_STRIDE = 7
DELTA_W = 5
```

for the first code fix, because it changes the least. After the pipeline runs, test `DELTA_W = 2` as an experiment.

### 6. `windows_meta.split` is useful, but Notebook 2 must still avoid leakage in feature aggregation

Adding `split` to `windows_meta` is correct. But this alone does not solve leakage if Notebook 2 later aggregates node features across all windows before splitting.

So add a strong note in the manifest or Notebook 2 plan:

```text
Split must be applied before any cross-window aggregation.
```

In other words, Notebook 2 should not do:

```python
all_node_window_features.groupby("node").agg(...)
```

before split. It should either train on node-window rows or aggregate separately inside train, val, and test.

### 7. Be careful with `adj_sparsity_window`

Your plan says replace global `adj_sparsity` with `adj_sparsity_window`. That is good, but use `n_nodes_window`, not `encoder.n_nodes`.

Formula:

```python
n_nodes_window = int(pd.concat([
    snapshot_edges["src_node"],
    snapshot_edges["dst_node"]
]).nunique())

adj_sparsity_window = (
    len(snapshot_edges) / (n_nodes_window ** 2)
    if n_nodes_window > 0
    else 0.0
)
```

You do not need to build the CSR matrix just to compute `adj_nnz`; `adj_nnz` equals `len(snapshot_edges)` if each `(src_node, dst_node)` pair is already aggregated once.

So this part can be simplified and made safer.

### 8. Update `windows_meta` without removing old columns

Your plan says keep existing columns. Good. I would keep:

```text
window
start
end
n_tx
n_temporal
n_second
n_snapshot
adj_nnz
adj_sparsity
```

but add:

```text
window_key
split
n_nodes_window
adj_sparsity_window
```

Then later you can deprecate `adj_sparsity`. For minimal impact, do not remove it immediately unless Notebook 2 definitely does not use it.

If you keep `adj_sparsity`, make clear it is diagnostic only.

### 9. Debug artifacts should be opt-in

Your plan says optional debug-labeled shards are requested. I would implement:

```python
WRITE_DEBUG_SHARDS = True
```

or:

```python
EXPORT_DEBUG_SHARDS = True
```

in the execution config. This makes it explicit and avoids always writing larger label-containing files.

### 10. Add validation that feature shards do not contain forbidden columns

Your checklist says confirm it manually. Better to enforce it during execution.

Add:

```python
FORBIDDEN_FEATURE_LABEL_COLS = {"is_sar", "alert_1", "alert_2", "n_alert", "_n_alert"}

def assert_no_label_columns(df, artifact_name):
    leaked = FORBIDDEN_FEATURE_LABEL_COLS & set(df.columns)
    if leaked:
        raise ValueError(f"{artifact_name} contains label columns: {sorted(leaked)}")
```

Call this before saving temporal and second-order feature shards.

## Revised version of your plan

I would modify your plan as follows:

1. Add `event_id` after deterministic sorting in `load_transactions`. Preserve original raw row order with `_raw_row_id` before sorting. Add `event_id` to returned columns and execution validation.

2. Keep `NodeEncoder` unchanged. Add assertions after encoding: `event_id` exists, `event_id` is unique, and `node_id` is documented as identifier-only.

3. Modify canonical `build_temporal_edges` to be feature-safe. It should use `event_id`, not `is_sar`. It should output `event_id_1`, `event_id_2`, structural fields, amount fields, and `_gap`. Sort before `groupby().head(max_fan)`.

4. Add `build_temporal_edges_debug` as a separate function that attaches `alert_1` and `alert_2` from `transactions.parquet`.

5. Keep exactly one canonical `build_second_order_edges`, feature-safe, with no `n_alert`. Add `build_second_order_edges_debug` separately.

6. Use `WINDOW_SIZE = 7`, `WINDOW_STRIDE = 7`. Keep `DELTA_W = 5` for first minimal-impact run, then experiment with `2`.

7. Add `window_key` and chronological `split` to `windows_meta`. Preserve old metadata columns for compatibility.

8. Replace global `adj_sparsity` with `adj_sparsity_window`, or keep both temporarily but mark global `adj_sparsity` diagnostic-only.

9. Update schema checks to feature-safe schemas. Add separate debug schema checks only if debug writing is enabled.

10. Update `graph_manifest.json` with artifact schema version, feature-shard label policy, debug directory paths, window policy, and split policy.

11. Add automatic no-label validation for feature shards.

## One important downstream warning

Your plan correctly says: “If Notebook 2 currently expects `alert_*` or `n_alert`, it must be updated.” This is unavoidable. Removing label columns from feature shards is the correct fix, but Notebook 2 will need a schema update if it currently reconstructs labels from temporal edges.

The new rule should be:

```text
Notebook 2 reads labels from transactions.parquet only.
Notebook 2 reads graph structure from temporal_edges/ and second_order_edges/.
Notebook 2 never uses temporal_edges_debug/ or second_order_edges_debug/ for model features.
```

## Final assessment

Your plan is solid. The only real problems are implementation details: do not rely on `event_id` in `dtypes` before it exists, keep `_gap` naming consistent, use separate debug functions rather than a flag inside the canonical builder, preserve old metadata columns temporarily, and enforce “no label columns in feature shards” with runtime assertions.

With those revisions, the plan is safe and should not damage Notebook 2 wiring more than necessary.
