

# ============================================================
# Cell 2: Config Paths and Parameters
# ============================================================

# ----------------------------
# Input path
# ----------------------------
DATA_PATH = "/content/drive/MyDrive/AML/dataset/subgraph_1hop_edges.parquet"

# ----------------------------
# Output directories
# ----------------------------
OUTPUT_DIR = "/content/drive/MyDrive/AML/outputs/motif_mining"

MOTIF_INSTANCE_DIR = f"{OUTPUT_DIR}/motif_instances"
MEMBERSHIP_DIR = f"{OUTPUT_DIR}/edge_motif_membership"
FEATURE_DIR = f"{OUTPUT_DIR}/edge_features"
LOG_DIR = f"{OUTPUT_DIR}/logs"

for path in [OUTPUT_DIR, MOTIF_INSTANCE_DIR, MEMBERSHIP_DIR, FEATURE_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Windowing parameters
# ----------------------------
WINDOW_SIZE = 30
STRIDE = 30

# Maximum allowed gap between two consecutive edges in a temporal motif.
DELTA_HOP = 5

# Maximum total duration of a motif instance.
# This is also used as lookahead for extended windows.
MAX_MOTIF_DURATION = 20

assert DELTA_HOP <= MAX_MOTIF_DURATION

# ----------------------------
# Candidate / output caps
# ----------------------------
MAX_IN_CANDIDATES = 100
MAX_OUT_CANDIDATES = 100
MAX_BRANCHING = 50

MAX_INSTANCES_PER_ANCHOR = 1_000
MAX_INSTANCES_PER_WINDOW = 1_000_000
MAX_FAN_INSTANCES_PER_WINDOW = 10_000
MAX_FAN_INSTANCES_PER_CENTER = 20
MATCHER_OUTPUT_FLUSH_EVERY = 5_000

# ----------------------------
# Amount constraints
# ----------------------------
AMOUNT_MIN = None

AMOUNT_RATIO_MIN = 0.3
AMOUNT_RATIO_MAX = 2.0

# ----------------------------
# Internal schema names
# ----------------------------
RAW_TO_INTERNAL_COLS = {
    "nameOrig": "src",
    "nameDest": "dst",
    "isSAR": "is_sar",
}

REQUIRED_RAW_COLUMNS = [
    "edge_id",
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isSAR",
]

MATCHER_COLUMNS = [
    "edge_id",
    "src",
    "dst",
    "step",
    "amount",
    "is_sar",
]


# ----------------------------
# Fan-in / fan-out branch sizes
# ----------------------------
FAN_IN_SIZES  = [3, 4, 5, 6, 7]   # number of incoming branches per pattern
FAN_OUT_SIZES = [6, 7, 8, 9, 10]   # loại bỏ 3, 4, 5; thêm 9, 10

# Candidate cap schedule keyed by branch count.
# C(cap, n) is bounded at roughly 250 000 for all n.
FAN_IN_CAP_SCHEDULE  = {3: 100, 4: 50, 5: 25, 6: 15, 7: 10}
FAN_OUT_CAP_SCHEDULE = {6: 15, 7: 10, 8: 8, 9: 6, 10: 5}

# ----------------------------
# Split-merge branch sizes
# ----------------------------
SPLIT_MERGE_SIZES = [3, 4, 5]     # number of branch pairs (split legs = merge legs)

SPLIT_MERGE_OUT_CAP   = {3: 12, 4: 8, 5: 6}
SPLIT_MERGE_MERGE_CAP = {3: 3,  4: 2, 5: 2}

# ----------------------------
# Center-in-out arm configs
# ----------------------------
# Each tuple is (n_in, n_out).
CENTER_INOUT_CONFIGS = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 3)]

CENTER_INOUT_IN_CAP  = {3: 30, 4: 15, 5: 10}
CENTER_INOUT_OUT_CAP = {2: 20, 3: 12,  4: 8}

# ----------------------------
# Cycle sizes
# ----------------------------
CYCLE_SIZES           = list(range(5, 11))   # 5, 6, 7, 8, 9, 10
BIDIRECTIONAL_CYCLE_THRESHOLD = 999            # retained for compatibility; cycle matching uses forward DFS only

CYCLE_MAX_BRANCHING = {5: 8, 6: 6, 7: 5, 8: 4, 9: 4, 10: 4}

# ----------------------------
# Stacked Bipartite layer configs
# ----------------------------
# Each entry is a list of intermediate layer widths.
STACKED_BIPARTITE_CONFIGS = [
    [2],        # equivalent to split_merge_4
    [3],        # equivalent to split_merge_6
    [2, 2],     # two-layer: 1→2→2→1
    [3, 2],     # two-layer: 1→3→2→1
]


print("Config ready.")
print("DATA_PATH:", DATA_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("WINDOW_SIZE:", WINDOW_SIZE)
print("STRIDE:", STRIDE)
print("DELTA_HOP:", DELTA_HOP)
print("MAX_MOTIF_DURATION:", MAX_MOTIF_DURATION)



