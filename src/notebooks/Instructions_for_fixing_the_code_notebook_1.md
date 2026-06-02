Yes. The split problem and the Notebook 1 problems are connected, but they are not the same problem.

Notebook 2 currently has the obvious problem: it uses random splitting for a temporal AML problem. Notebook 1 has a more subtle problem: it creates graph artifacts in a way that can make proper temporal splitting in Notebook 2 difficult or unsafe.

Below is the deeper analysis.

## 1. The core issue: what is the correct evaluation setting?

For this project, the realistic AML setting is:

```text
Use past transaction behavior to detect or score suspicious behavior in later periods.
```

That means the model should not be evaluated with a random split. It should be evaluated with a chronological split.

The correct data unit should be:

```text
one row = one node in one time window
```

not:

```text
one row = one global node aggregated across all time
```

and not:

```text
one row = one transaction randomly split
```

The clean modeling table should eventually look like:

```text
node
window_key
window_start
window_end
transaction_features
graph_features
motif_features
label
split
```

The split should be assigned by `window_start` or `window_end`:

```text
train      = old windows
validation = middle windows
test       = newest windows
```

The main risk is that if Notebook 1 creates overlapping or label-contaminated graph artifacts, Notebook 2 can accidentally build features that already contain future or target information.

---

## 2. Notebook 1 does not directly train a model, but it can create leaky artifacts

Notebook 1 currently does not perform supervised training. So it does not directly leak through `train_test_split`. Its responsibility is graph artifact generation.

The current flow is:

```text
load raw transactions
normalize columns
globally encode nodes
save transactions.parquet
stream windows
build temporal_edges
build snapshot_edges
build second_order_edges
save graph shards
save windows_meta
save graph_manifest
```

The proof is in the execution section: it loads transactions, encodes nodes globally, saves global artifacts, streams windows, builds temporal, snapshot, and second-order edges, then saves metadata and manifest.  

This is a reasonable graph-building pipeline, but several design choices become dangerous when Notebook 2 uses these artifacts for supervised temporal evaluation.

---

# 3. Problem 1: labels are embedded into graph artifacts

This is the most serious issue in Notebook 1.

`LoaderConfig.keep_cols` explicitly keeps `is_sar` inside the normalized transaction table:

```text
src_node, dst_node, amount, step, is_sar
```



Then `load_transactions()` defaults `is_sar = 0` if absent, casts it, keeps it, and returns it as part of the canonical table. 

This is acceptable for the base transaction table because the model needs labels somewhere. The problem is that downstream graph builders use it as part of graph artifacts.

In the current graph construction, temporal edges include alert columns derived from `is_sar`: `alert_1` and `alert_2`. Earlier reviewed code shows that `build_temporal_edges()` renames `is_sar` into `alert_1` and `alert_2`, then saves these temporal shards. The second-order graph then aggregates these labels into `n_alert`.

That means the graph artifact can contain direct target information.

The dangerous route is:

```text
is_sar
-> alert_1, alert_2 in temporal_edges
-> n_alert in second_order_edges
-> possible feature in Notebook 2
-> target leakage
```

This is not hypothetical. If Notebook 2 later uses `n_alert`, `alert_1`, `alert_2`, or any feature derived from them, model performance becomes invalid.

The correct separation should be:

```text
transactions.parquet may contain is_sar as label source.
feature graph shards must not contain is_sar, alert_1, alert_2, or n_alert.
optional debug shards may contain labels, but must be clearly separated.
```

So Notebook 1 should produce either:

```text
temporal_edges_feature/
second_order_edges_feature/
```

without labels, and optionally:

```text
temporal_edges_labeled_debug/
second_order_edges_labeled_debug/
```

for diagnostics only.

---

# 4. Problem 2: overlapping windows are unsafe for temporal evaluation

Notebook 1 has inconsistent window settings.

At the config level, `GraphConfig` says:

```text
window_size = 7
window_stride = 7
```



But `LoaderConfig` says:

```text
window_size = 30
window_stride = 15
```



And `iter_windows()` defaults to:

```text
window_size = 30
window_stride = 15
```



This is not only a consistency problem. It also matters for leakage.

With:

```text
window_size = 30
window_stride = 15
```

windows overlap. Example:

```text
window 1: step 0 to 29
window 2: step 15 to 44
```

Transactions from step 15 to 29 appear in both windows.

If Notebook 2 later splits by windows, one window near a boundary can share transactions with another window in a different split. This can contaminate validation or test with training-period events.

For final supervised temporal evaluation, the safer setting is:

```text
window_stride = window_size
```

For example:

```text
window_size = 7, window_stride = 7
```

or:

```text
window_size = 30, window_stride = 30
```

If you want overlapping windows for graph exploration, keep them in a separate artifact folder and do not use them for the main supervised test.

---

# 5. Problem 3: global node encoding is mostly safe, but needs a rule

Notebook 1 uses a global `NodeEncoder`, fits it on all source and destination nodes, then transforms the whole transaction table. The encoder learns all nodes from the full dataset before any temporal split. 

This is not automatically target leakage. If `node_id` is only a join key, it is acceptable and useful because it gives stable node identities across all graph shards.

But it becomes leakage or at least an unrealistic evaluation setting if:

```text
node_id is used as a numeric feature,
future-only node existence affects training features,
global node counts are used to normalize graph features,
or the intended setting is inductive evaluation on unseen future accounts.
```

The current design should explicitly state:

```text
node_id is an identifier only.
It must be excluded from model features.
```

For now, I would keep global encoding because it prevents many join problems between Notebook 1 and Notebook 2. But Notebook 2 must not use `node` as a direct feature.

A stricter future design could use a train-only encoder and map future unseen nodes to `UNKNOWN`, but that is a more advanced setting and not necessary for the first temporal evaluation.

---

# 6. Problem 4: Notebook 1 lacks `event_id`, which weakens motif correctness

The current canonical table keeps:

```text
src_node
dst_node
amount
step
is_sar
```

but not `event_id`. 

This is a correctness problem for Notebook 2.

When Notebook 2 reconstructs event-level rows from `temporal_edges`, it has to infer unique events from values like:

```text
src_node, dst_node, step, amount, is_sar
```

That can collapse two real transactions if they share the same tuple.

For graph snapshots, this may not matter much because aggregation is expected. For exact motif mining, it matters because the number of event instances and motif paths can change.

Notebook 1 should create:

```text
event_id
```

immediately after sorting the transaction table, and `temporal_edges` should preserve:

```text
event_id_1
event_id_2
```

This is not only cleaner. It makes Notebook 2 much easier to fix because motif instances can reference real event IDs instead of approximate reconstructed events.

---

# 7. Problem 5: `max_fan` pruning can be unstable

The temporal edge builder prunes high-fan nodes before the self-join. The idea is good because temporal self-joins can explode. But the pruning method is order-dependent unless the dataframe is explicitly sorted before `groupby().head(max_fan)`.

The loader sorts by `step` only.  If many transactions share the same `step`, the retained rows under `head(max_fan)` may depend on input order within that step.

This is not leakage, but it affects reproducibility and can bias which temporal relays survive.

Fix:

```text
sort by relay node, step, event_id before groupby().head(max_fan)
```

For example:

```python
left = left.sort_values(["dst_1", "step_1", "event_id_1"]).groupby("dst_1").head(max_fan)
right = right.sort_values(["src_2", "step_2", "event_id_2"]).groupby("src_2").head(max_fan)
```

This requires adding `event_id` first.

---

# 8. Problem 6: configuration is fragmented

Notebook 1 currently has multiple places where window parameters exist:

```text
GraphConfig.window_size = 7
GraphConfig.window_stride = 7
LoaderConfig.window_size = 30
LoaderConfig.window_stride = 15
iter_windows default = 30 / 15
execution section also defines WINDOW_SIZE / WINDOW_STRIDE separately
```

The proof for the first three is visible in the config and iterator sections.   

This makes it easy to think the notebook is using one temporal design while it is actually using another.

For the current problem, there should be one source of truth:

```python
WINDOW_SIZE = 7
WINDOW_STRIDE = 7
DELTA_W = 2
```

or:

```python
WINDOW_SIZE = 30
WINDOW_STRIDE = 30
DELTA_W = 5
```

My recommendation:

```text
Use 7/7 for motif-focused AML temporal detection.
Use 30/30 only if the goal is slower community-level behavior.
Do not use 30/15 for final supervised temporal evaluation.
```

---

# 9. Problem 7: Notebook structure is fragile

Notebook 1 repeats `from __future__ import annotations` after executable statements. In a real `.py` script this is invalid. It may work in separate notebook cells, but this file is a Markdown code notebook, so it is structurally fragile. The first occurrence appears after imports and runtime logic, then later it appears again.   

The notebook also defines config, loader, encoder, graph builders, and execution in one long file. That is acceptable for Colab, but only if cells are cleanly separated and rerun in order.

The practical fix is not necessarily to move everything into modules immediately. But each function should be defined once, imports should be consolidated, and execution should start only after all definitions are complete.

---

# 10. What type of split is appropriate for the current data?

For this AML graph setting, the appropriate split is:

```text
temporal window split
```

not random row split.

The correct design is:

```text
Notebook 1:
creates non-overlapping window graph artifacts.

Notebook 2:
creates node-window feature rows.
assigns split using window_start/window_end or windows_meta.split.
trains on train windows.
tunes on validation windows.
reports final metrics on test windows.
```

The recommended split:

```text
Train: oldest 70 percent of windows
Validation: next 15 percent of windows
Test: newest 15 percent of windows
```

This is the minimum clean evaluation. Later, you can add walk-forward validation, but it is not necessary for the first correct version.

---

# 11. Concrete solution for Notebook 1

Do not rewrite everything randomly. The fixes should be applied in this order because each step depends on the previous one.

## Step 1. Add `event_id` in `load_transactions`

Current function returns after sorting by `step`. 

Modify it to sort stably and add `event_id`:

```python
df = df.sort_values(["step", "src_node", "dst_node", "amount"]).reset_index(drop=True)
df["event_id"] = np.arange(len(df), dtype=np.int64)

return df[["event_id", "src_node", "dst_node", "amount", "step", "is_sar"]]
```

This will require updating `_expected_cols` in the execution section from:

```python
{"src_node", "dst_node", "amount", "step", "is_sar"}
```

to:

```python
{"event_id", "src_node", "dst_node", "amount", "step", "is_sar"}
```

## Step 2. Modify `NodeEncoder.transform` to preserve `event_id`

This should already happen because it only changes `src_node` and `dst_node`, but after adding `event_id`, check that the column remains.

Add:

```python
assert "event_id" in tx_df.columns
assert tx_df["event_id"].is_unique
```

after encoding.

## Step 3. Modify `build_temporal_edges`

Current version builds from:

```python
["src_node", "dst_node", "step", "amount", "is_sar"]
```

It should build feature edges from:

```python
["event_id", "src_node", "dst_node", "step", "amount"]
```

The feature version should not include `is_sar`.

Output should be:

```text
event_id_1
src_1
dst_1
step_1
amount_1
event_id_2
src_2
dst_2
step_2
amount_2
_gap
```

If you need labels for diagnostics, create a separate optional debug function. Do not store alert columns inside the main temporal shard.

## Step 4. Modify `build_second_order_edges`

Current second-order builder computes `n_alert`. That should be removed from the feature artifact.

Feature output should be:

```text
src_2nd
dst_2nd
count
weight_src
weight_dst
avg_gap
```

No `n_alert`.

If `n_alert` is needed, create:

```text
second_order_edges_labeled_debug
```

and mark it as forbidden for modeling.

## Step 5. Change window config to non-overlap

Use:

```python
WINDOW_SIZE = 7
WINDOW_STRIDE = 7
DELTA_W = 2
```

or:

```python
WINDOW_SIZE = 30
WINDOW_STRIDE = 30
DELTA_W = 5
```

The key is:

```text
WINDOW_STRIDE == WINDOW_SIZE
```

for supervised temporal evaluation.

## Step 6. Add `window_key` and `split` to `windows_meta`

Current metadata includes:

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



Add:

```text
window_key = f"w_{start}_{end}"
split = train / val / test
```

The split should be assigned after all windows are collected:

```text
first 70 percent windows -> train
next 15 percent -> val
last 15 percent -> test
```

## Step 7. Remove global adjacency dimension as a feature signal

The current metadata stores `adj_sparsity` using `encoder.n_nodes ** 2`, where `encoder.n_nodes` comes from all nodes in the full dataset. 

This is not a major leak if used only as diagnostics, but it is not ideal as a model feature. Better:

```text
keep adj_nnz if needed
remove adj_sparsity, or compute sparsity using n_nodes_window ** 2
```

The safer metadata:

```text
n_nodes_window
adj_nnz
adj_sparsity_window = adj_nnz / n_nodes_window^2
```

But if adjacency is not saved or used, you can remove the CSR matrix construction entirely.

---

# 12. Final diagnosis

Notebook 1 has three serious design problems for temporal AML evaluation.

First, it allows target labels to enter graph artifacts through `alert_1`, `alert_2`, and `n_alert`. This is the most dangerous because it can directly leak the label into Notebook 2 features.

Second, it uses or defaults to overlapping windows, especially `30/15`. This can duplicate transactions across adjacent windows and contaminate temporal splits.

Third, it does not preserve `event_id`, which weakens exact motif reconstruction and can distort motif counts.

The best Notebook 1 policy is:

```text
Use Notebook 1 only to create clean temporal graph artifacts.
Keep labels only in transactions.parquet and optional debug files.
Use non-overlapping windows for final evaluation.
Preserve event_id through temporal edges.
Add window_key and split to windows_meta.
Never let target-derived columns enter feature graph shards.
```

This will make Notebook 2 much easier to fix because the downstream rule becomes simple:

```text
Build features by node + window_key.
Split by windows_meta.split.
Train on train, tune on validation, report on test.
```
