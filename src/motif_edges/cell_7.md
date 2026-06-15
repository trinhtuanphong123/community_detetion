

# ============================================================
# Cell 7: Temporal Index
# ============================================================

from collections import namedtuple

# Lightweight return type for query results — no __dict__, no Python int boxing overhead
_EdgeTuple = namedtuple('EdgeRecord', ['edge_id', 'src', 'dst', 'step', 'amount', 'is_sar'])

@dataclass
class EdgeRecord:
    """
    Lightweight edge record used by temporal matchers.

    Keeping this as a dataclass makes matcher code clearer than using raw dicts.
    """
    __slots__ = ('edge_id', 'src', 'dst', 'step', 'amount', 'is_sar')
    edge_id: int
    src: int
    dst: int
    step: int
    amount: float
    is_sar: int


class EdgeIndexView:
    def __init__(self, index: "TemporalIndex", idx_dict: Dict[Any, np.ndarray]):
        self._index    = index
        self._idx_dict = idx_dict

    def get(self, key, default=None):
        positions = self._idx_dict.get(key)
        if positions is None:
            return [] if default is None else default
        return self._index._materialize_edges(positions)

    def keys(self):
        return self._idx_dict.keys()

    def items(self):
        for key, positions in self._idx_dict.items():
            yield key, self._index._materialize_edges(positions)

    def __len__(self):
        return len(self._idx_dict)


class TemporalIndex:
    """
    NumPy-backed temporal index.

    Edge data stored as parallel NumPy arrays (28 bytes/edge total).
    Index dicts map node/pair keys to NumPy int32 index arrays (4 bytes/entry).
    Step arrays are views into self._steps — zero additional allocation.

    Query interface is identical to the previous version.
    Matchers require NO changes.
    """

    def __init__(self, edges: List[EdgeRecord]):
        self.num_edges = len(edges)
        if self.num_edges == 0:
            self._edge_ids   = np.empty(0, dtype=np.int32)
            self._srcs       = np.empty(0, dtype=np.int64)
            self._dsts       = np.empty(0, dtype=np.int64)
            self._steps      = np.empty(0, dtype=np.int32)
            self._amounts    = np.empty(0, dtype=np.float32)
            self._is_sars    = np.empty(0, dtype=np.int8)
            self._out_idx    = {}
            self._in_idx     = {}
            self._pair_idx   = {}
            self.node_in_degree  = {}
            self.node_out_degree = {}
            self.pair_count      = {}
            self.amount_quantiles = {q: 0.0 for q in [0.1,0.25,0.5,0.75,0.9,0.95,0.99]}
            self.sar_rate_window = 0.0
            self.out_edges = EdgeIndexView(self, self._out_idx)
            self.in_edges  = EdgeIndexView(self, self._in_idx)
            return

        # Sort once by (step, edge_id) — all arrays share this order
        sort_keys = sorted(range(len(edges)), key=lambda i: (edges[i].step, edges[i].edge_id))

        self._edge_ids = np.array([edges[i].edge_id for i in sort_keys], dtype=np.int32)
        self._srcs     = np.array([edges[i].src     for i in sort_keys], dtype=np.int64)
        self._dsts     = np.array([edges[i].dst     for i in sort_keys], dtype=np.int64)
        self._steps    = np.array([edges[i].step    for i in sort_keys], dtype=np.int32)
        self._amounts  = np.array([edges[i].amount  for i in sort_keys], dtype=np.float32)
        self._is_sars  = np.array([edges[i].is_sar  for i in sort_keys], dtype=np.int8)

        # Build index dicts: key -> sorted np.int32 array of positions into above arrays
        out_lists  = defaultdict(list)
        in_lists   = defaultdict(list)
        pair_lists = defaultdict(list)
        in_deg  = defaultdict(int)
        out_deg = defaultdict(int)
        pair_ct = defaultdict(int)

        for pos in range(self.num_edges):
            s = int(self._srcs[pos])
            d = int(self._dsts[pos])
            out_lists[s].append(pos)
            in_lists[d].append(pos)
            pair_lists[(s, d)].append(pos)
            out_deg[s] += 1
            in_deg[d]  += 1
            pair_ct[(s, d)] += 1

        # Convert to NumPy int32 arrays — 4 bytes per entry vs 36 bytes for Python int
        self._out_idx  = {k: np.array(v, dtype=np.int32) for k, v in out_lists.items()}
        self._in_idx   = {k: np.array(v, dtype=np.int32) for k, v in in_lists.items()}
        self._pair_idx = {k: np.array(v, dtype=np.int32) for k, v in pair_lists.items()}

        self.node_in_degree  = dict(in_deg)
        self.node_out_degree = dict(out_deg)
        self.pair_count      = dict(pair_ct)

        self.out_edges = EdgeIndexView(self, self._out_idx)
        self.in_edges  = EdgeIndexView(self, self._in_idx)

        # Window-level stats
        total_sar = int(self._is_sars.sum())
        self.sar_rate_window = float(total_sar / self.num_edges)
        q_vals = np.quantile(self._amounts, [0.1,0.25,0.5,0.75,0.9,0.95,0.99])
        self.amount_quantiles = {
            q: float(v)
            for q, v in zip([0.1,0.25,0.5,0.75,0.9,0.95,0.99], q_vals)
        }

    def _materialize_edges(self, positions: np.ndarray) -> List[_EdgeTuple]:
        """
        Convert a NumPy int32 position array into a list of _EdgeTuple namedtuples.
        Namedtuples have no __dict__ and use ~120 bytes each vs ~416 for dataclass.
        Called only for the small result sets returned by query methods (cap <= 100).
        """
        return [
            _EdgeTuple(
                edge_id = int(self._edge_ids[p]),
                src     = int(self._srcs[p]),
                dst     = int(self._dsts[p]),
                step    = int(self._steps[p]),
                amount  = float(self._amounts[p]),
                is_sar  = int(self._is_sars[p]),
            )
            for p in positions
        ]

    def _range_query(
        self,
        idx_dict: Dict[Any, np.ndarray],
        key: Any,
        t_min: int,
        t_max: int,
        include_left: bool = False,
        max_candidates: Optional[int] = None,
    ) -> List[_EdgeTuple]:
        positions = idx_dict.get(key)
        if positions is None or len(positions) == 0:
            return []

        # np.searchsorted on the step values at these positions
        step_vals = self._steps[positions]   # view, no copy
        if include_left:
            left  = int(np.searchsorted(step_vals, t_min,     side='left'))
        else:
            left  = int(np.searchsorted(step_vals, t_min,     side='right'))
        right = int(np.searchsorted(step_vals, t_max, side='right'))

        result_pos = positions[left:right]   # NumPy slice, no copy
        if max_candidates is not None and len(result_pos) > max_candidates:
            result_pos = result_pos[:max_candidates]

        return self._materialize_edges(result_pos)

    def outgoing(self, src, t_min, t_max, include_left=False, max_candidates=None):
        return self._range_query(self._out_idx, src, t_min, t_max, include_left, max_candidates)

    def incoming(self, dst, t_min, t_max, include_left=False, max_candidates=None):
        return self._range_query(self._in_idx, dst, t_min, t_max, include_left, max_candidates)

    def pair(self, src, dst, t_min, t_max, include_left=False, max_candidates=None):
        return self._range_query(self._pair_idx, (src, dst), t_min, t_max, include_left, max_candidates)

    def _has_any(self, idx_dict, key, t_min, t_max):
        positions = idx_dict.get(key)
        if positions is None or len(positions) == 0:
            return False
        step_vals = self._steps[positions]
        left  = int(np.searchsorted(step_vals, t_min, side='right'))
        right = int(np.searchsorted(step_vals, t_max, side='right'))
        return right > left

    def has_any_outgoing(self, src, t_min, t_max):
        return self._has_any(self._out_idx, src, t_min, t_max)

    def has_any_incoming(self, dst, t_min, t_max):
        return self._has_any(self._in_idx, dst, t_min, t_max)

    def has_any_pair(self, src, dst, t_min, t_max):
        return self._has_any(self._pair_idx, (src, dst), t_min, t_max)

    def stats(self):
        return {
            "num_edges": self.num_edges,
            "num_src_nodes": len(self._out_idx),
            "num_dst_nodes": len(self._in_idx),
            "num_pairs": len(self.pair_count),
        }



def build_temporal_index_from_polars(df_window_edges: pl.DataFrame) -> TemporalIndex:
    required_cols = ["edge_id", "src", "dst", "step", "amount", "is_sar"]
    missing_cols = [c for c in required_cols if c not in df_window_edges.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for TemporalIndex: {missing_cols}")
    # Tối ưu hoá tốc độ: chuyển đổi sang lists và zip thay vì dùng to_dicts() chậm
    edge_ids = df_window_edges["edge_id"].to_list()
    srcs = df_window_edges["src"].to_list()
    dsts = df_window_edges["dst"].to_list()
    steps = df_window_edges["step"].to_list()
    amounts = df_window_edges["amount"].to_list()
    is_sars = df_window_edges["is_sar"].to_list()
    edge_records = [
        EdgeRecord(
            edge_id=eid,
            src=s,
            dst=d,
            step=t,
            amount=a,
            is_sar=sar,
        )
        for eid, s, d, t, a, sar in zip(edge_ids, srcs, dsts, steps, amounts, is_sars)
    ]
    edge_records.sort(key=lambda e: (e.step, e.edge_id))
    return TemporalIndex(edge_records)


# ------------------------------------------------------------
# Smoke test on first non-empty window from Cell 6
# ------------------------------------------------------------

index_build_start = time.time()

test_index = build_temporal_index_from_polars(df_extended_test)

index_build_end = time.time()

print("TemporalIndex smoke test completed.")
print("Index build time:", round(index_build_end - index_build_start, 3), "seconds")
print("Index stats:", test_index.stats())

# Pick one edge from primary test window and query possible next outgoing edges.
if df_primary_test.height > 0:
    anchor = df_primary_test.row(0, named=True)

    anchor_edge = EdgeRecord(
        edge_id=int(anchor["edge_id"]),
        src=int(anchor["src"]),
        dst=int(anchor["dst"]),
        step=int(anchor["step"]),
        amount=float(anchor["amount"]),
        is_sar=int(anchor["is_sar"]),
    )

    t_min = anchor_edge.step
    t_max = anchor_edge.step + DELTA_HOP

    next_candidates = test_index.outgoing(
        src=anchor_edge.dst,
        t_min=t_min,
        t_max=t_max,
        include_left=False,
        max_candidates=MAX_BRANCHING,
    )

    print("\nAnchor edge:")
    print(anchor_edge)

    print(f"\nOutgoing candidates from anchor.dst={anchor_edge.dst} in ({t_min}, {t_max}]:")
    print("num_candidates:", len(next_candidates))

    if len(next_candidates) > 0:
        for cand in next_candidates[:10]:
            print(cand)
else:
    print("df_primary_test is empty. No anchor query performed.")

print("\nCell 7 completed.")



