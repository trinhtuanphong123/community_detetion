


# ============================================================
# Cell 8: Helper Functions for Motif Instance Output
# ============================================================

import json
import hashlib


def edge_record_to_dict(e: EdgeRecord) -> Dict[str, Any]:
    """
    Convert EdgeRecord to plain dict.
    Useful for debugging and JSON-safe serialization.
    """
    return {
        "edge_id": int(e.edge_id),
        "src": int(e.src),
        "dst": int(e.dst),
        "step": int(e.step),
        "amount": float(e.amount),
        "is_sar": int(e.is_sar),
    }


def make_motif_instance_id(
    window_id: int,
    motif_type: str,
    canonical_key: str,
) -> str:
    """
    Create stable motif_instance_id from window_id, motif_type, and canonical key.

    This is better than using a simple counter because it is reproducible.
    """

    digest = hashlib.md5(canonical_key.encode("utf-8")).hexdigest()[:12]
    return f"w{int(window_id):06d}_{motif_type}_{digest}"


def infer_ordered_flag(pattern: MotifPattern) -> bool:
    """
    Decide whether edge order should be meaningful for canonical key.

    fan_in and fan_out use unordered branch combinations.
    Path-like and composite motifs use ordered edge roles.
    """

    unordered_matcher_types = {"fan_in", "fan_out"}

    return pattern.matcher_type not in unordered_matcher_types


def validate_instance_edges(
    edges: List[EdgeRecord],
    pattern: MotifPattern,
) -> None:
    """
    Basic validation for a candidate motif instance.
    This does not fully validate structural correctness.
    Structural correctness belongs to each matcher.
    """

    if len(edges) != len(pattern.edges):
        raise ValueError(
            f"Pattern {pattern.name} expects {len(pattern.edges)} edges, "
            f"but got {len(edges)} edges."
        )

    edge_ids = [e.edge_id for e in edges]

    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError(
            f"Duplicate edge_id inside motif instance for pattern {pattern.name}: {edge_ids}"
        )

    steps = [e.step for e in edges]

    if pattern.time_order == "strict":
        if any(steps[i + 1] <= steps[i] for i in range(len(steps) - 1)):
            raise ValueError(
                f"Strict time order violated for pattern {pattern.name}: steps={steps}"
            )

    elif pattern.time_order == "nondecreasing":
        if any(steps[i + 1] < steps[i] for i in range(len(steps) - 1)):
            raise ValueError(
                f"Nondecreasing time order violated for pattern {pattern.name}: steps={steps}"
            )

    duration = max(steps) - min(steps)

    if duration > pattern.max_duration:
        raise ValueError(
            f"Max duration violated for pattern {pattern.name}: "
            f"duration={duration}, max_duration={pattern.max_duration}"
        )


def make_motif_instance_row(
    window_id: int,
    pattern: MotifPattern,
    edges: List[EdgeRecord],
    node_map: Optional[Dict[str, int]] = None,
    role_map: Optional[Dict[str, int]] = None,
    anchor_edge_id: Optional[int] = None,
    validate: bool = False,
    index: Optional[TemporalIndex] = None,
    candidate_rank: int = -1,
) -> Dict[str, Any]:

    if validate:
        validate_instance_edges(edges, pattern)

    edge_ids = [int(e.edge_id) for e in edges]
    steps = [int(e.step) for e in edges]

    ordered = infer_ordered_flag(pattern)

    canonical_key = make_canonical_key(
        motif_type=pattern.name,
        edge_ids=edge_ids,
        ordered=ordered,
    )

    motif_instance_id = make_motif_instance_id(
        window_id=window_id,
        motif_type=pattern.name,
        canonical_key=canonical_key,
    )

    if anchor_edge_id is None:
        anchor_edge_id = edge_ids[0]

    instance_score_val = score_instance(edges, index, pattern) if index is not None else 0.0

    return {
        "motif_instance_id": motif_instance_id,
        "window_id": int(window_id),
        "motif_type": pattern.name,
        "matcher_type": pattern.matcher_type,
        "canonical_key": canonical_key,
        "anchor_edge_id": int(anchor_edge_id),
        "edge_ids": edge_ids,
        "instance_score": float(instance_score_val),
        "candidate_rank": int(candidate_rank),
        "num_edges": int(len(edges)),
        "start_step": int(min(steps)),
        "end_step": int(max(steps)),
        "duration": int(max(steps) - min(steps)),
    }




def make_edge_motif_membership_rows(
    motif_row: Dict[str, Any],
    pattern: MotifPattern,
    edges: List[EdgeRecord],
) -> List[Dict[str, Any]]:

    pattern_edges_sorted = sorted(pattern.edges, key=lambda x: x.order)

    rows = []

    for p_edge, real_edge in zip(pattern_edges_sorted, edges):
        role = p_edge.role if p_edge.role else p_edge.name

        rows.append({
            "edge_id": int(real_edge.edge_id),
            "motif_instance_id": motif_row["motif_instance_id"],
            "window_id": int(motif_row["window_id"]),
            "motif_type": motif_row["motif_type"],
            "matcher_type": motif_row["matcher_type"],
            "instance_score": float(motif_row["instance_score"]),
            "role_in_motif": role,
        })

    return rows

def motif_instance_rows_to_polars(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "motif_instance_id": pl.String,
        "window_id": pl.Int64,
        "motif_type": pl.String,
        "matcher_type": pl.String,
        "canonical_key": pl.String,
        "anchor_edge_id": pl.Int64,
        "edge_ids": pl.List(pl.Int64),
        "instance_score": pl.Float64,
        "candidate_rank": pl.Int64,
        "num_edges": pl.Int64,
        "start_step": pl.Int64,
        "end_step": pl.Int64,
        "duration": pl.Int64,
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)

    return df.select([
        pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c)
        for c, t in schema.items()
    ])
def membership_rows_to_polars(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "edge_id": pl.Int64,
        "motif_instance_id": pl.String,
        "window_id": pl.Int64,
        "motif_type": pl.String,
        "matcher_type": pl.String,
        "instance_score": pl.Float64,
        "role_in_motif": pl.String,
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)

    return df.select([
        pl.col(c).cast(t) if c in df.columns else pl.lit(None, dtype=t).alias(c)
        for c, t in schema.items()
    ])





# Define new MatcherOutputBuffer
class MatcherOutputBuffer:
    def __init__(
        self,
        motif_dir: str,
        membership_dir: str,
        motif_flush_size: int = 2000,
        membership_flush_size: int = 10000,
        flush_every_instances: Optional[int] = None,
    ):
        self.motif_dir = motif_dir
        self.membership_dir = membership_dir
        if flush_every_instances is not None:
            self.motif_flush_size = flush_every_instances
            self.membership_flush_size = flush_every_instances * 5
        else:
            self.motif_flush_size = motif_flush_size
            self.membership_flush_size = membership_flush_size
        self.motif_rows = []
        self.membership_rows = []
        self.motif_shard_id = 0
        self.membership_shard_id = 0
        os.makedirs(self.motif_dir, exist_ok=True)
        os.makedirs(self.membership_dir, exist_ok=True)

    def add(self, motif_row: Dict[str, Any], membership_rows: List[Dict[str, Any]]):
        self.motif_rows.append(motif_row)
        self.membership_rows.extend(membership_rows)
        if len(self.motif_rows) >= self.motif_flush_size:
            self.flush_motifs()
        if len(self.membership_rows) >= self.membership_flush_size:
            self.flush_memberships()

    def flush_motifs(self):
        if not self.motif_rows:
            return
        path = f"{self.motif_dir}/motif_shard_{self.motif_shard_id:06d}.parquet"
        df = motif_instance_rows_to_polars(self.motif_rows)
        df.write_parquet(path)
        self.motif_rows.clear()
        self.motif_shard_id += 1
        del df
        gc.collect()

    def flush_memberships(self):
        if not self.membership_rows:
            return
        path = f"{self.membership_dir}/membership_shard_{self.membership_shard_id:06d}.parquet"
        df = membership_rows_to_polars(self.membership_rows)
        df.write_parquet(path)
        self.membership_rows.clear()
        self.membership_shard_id += 1
        del df
        gc.collect()

    def close(self, write_empty_outputs: bool = True):
        self.flush_motifs()
        self.flush_memberships()
        if write_empty_outputs:
            if self.motif_shard_id == 0:
                df = motif_instance_rows_to_polars([])
                df.write_parquet(f"{self.motif_dir}/motif_shard_000000.parquet")
            if self.membership_shard_id == 0:
                df = membership_rows_to_polars([])
                df.write_parquet(f"{self.membership_dir}/membership_shard_000000.parquet")

