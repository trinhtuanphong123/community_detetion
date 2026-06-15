


 # ============================================================
# Cell 5a: Window Candidate Summary
# ============================================================

@dataclass
class WindowCandidateSummary:
    """
    Lightweight per-window structural summary.

    Computed once per window, shared across all matchers.
    Allows matchers to skip nodes that cannot satisfy minimum
    degree requirements before touching the temporal index.
    """
    total_edges: int
    node_in_degree:  Dict[int, int]
    node_out_degree: Dict[int, int]

    # Pre-filtered node lists for each matcher type.
    fan_in_candidate_dsts:   Dict[int, List[int]]   # dst -> [min_branches] for each size
    fan_out_candidate_srcs:  Dict[int, List[int]]   # src -> [min_branches] for each size
    center_candidate_nodes:  List[int]               # nodes with in >= 2 and out >= 2

    @property
    def high_in_degree_nodes(self) -> List[int]:
        return [n for n, d in self.node_in_degree.items()  if d >= 3]

    @property
    def high_out_degree_nodes(self) -> List[int]:
        return [n for n, d in self.node_out_degree.items() if d >= 3]


def build_window_candidate_summary(
    df_extended: pl.DataFrame,
    min_fan_branches: int = 3,
    min_center_in:    int = 2,
    min_center_out:   int = 2,
) -> WindowCandidateSummary:
    """
    Build a WindowCandidateSummary from the extended window edge dataframe.

    This is O(|edges|) and should be called once per window before
    any matcher is invoked.
    """
    in_degree:  Dict[int, int] = defaultdict(int)
    out_degree: Dict[int, int] = defaultdict(int)

    for row in df_extended.select(["src", "dst"]).iter_rows():
        src, dst = row
        out_degree[src] += 1
        in_degree[dst]  += 1

    # Pre-build candidate node lists per role.
    fan_in_dsts   = {n for n, d in in_degree.items()  if d >= min_fan_branches}
    fan_out_srcs  = {n for n, d in out_degree.items() if d >= min_fan_branches}
    center_nodes  = [
        n for n in set(in_degree) & set(out_degree)
        if in_degree[n] >= min_center_in and out_degree[n] >= min_center_out
    ]

    return WindowCandidateSummary(
        total_edges      = df_extended.height,
        node_in_degree   = dict(in_degree),
        node_out_degree  = dict(out_degree),
        fan_in_candidate_dsts  = {n: in_degree[n]  for n in fan_in_dsts},
        fan_out_candidate_srcs = {n: out_degree[n] for n in fan_out_srcs},
        center_candidate_nodes = center_nodes,
    )


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

# test_summary = build_window_candidate_summary(df_extended_test, min_fan_branches=3)

# print("WindowCandidateSummary smoke test:")
# print("  total_edges:           ", test_summary.total_edges)
# print("  nodes with in >= 3:    ", len(test_summary.fan_in_candidate_dsts))
# print("  nodes with out >= 3:   ", len(test_summary.fan_out_candidate_srcs))
# print("  center candidates:     ", len(test_summary.center_candidate_nodes))
print("\nCell 5a completed.")


