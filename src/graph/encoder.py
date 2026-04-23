"""
encoder.py — Global node ID encoder.

Maps arbitrary node identifiers to contiguous integer IDs that remain
consistent across time windows.  This is required because cuGraph
expects integer vertex IDs starting from 0.
"""

from __future__ import annotations

try:
    import cudf as pd_lib
except ImportError:
    import pandas as pd_lib


class NodeEncoder:
    """Encodes heterogeneous node labels to contiguous ints.

    Usage
    -----
    >>> enc = NodeEncoder()
    >>> df = enc.fit_transform(df, src_col="src_node", dst_col="dst_node")
    >>> enc.decode(encoded_ids)
    """

    def __init__(self):
        self._label_to_id: dict = {}
        self._id_to_label: dict = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: "pd_lib.DataFrame", src_col: str = "src_node",
            dst_col: str = "dst_node") -> "NodeEncoder":
        """Learn mapping from all unique nodes in *df*."""
        if hasattr(df[src_col], "to_pandas"):
            src_vals = df[src_col].to_pandas().unique()
            dst_vals = df[dst_col].to_pandas().unique()
        else:
            src_vals = df[src_col].unique()
            dst_vals = df[dst_col].unique()

        import numpy as np
        all_nodes = np.union1d(src_vals, dst_vals)
        for node in all_nodes:
            node_key = int(node) if hasattr(node, "item") else node
            if node_key not in self._label_to_id:
                self._label_to_id[node_key] = self._next_id
                self._id_to_label[self._next_id] = node_key
                self._next_id += 1
        return self

    def transform(self, df: "pd_lib.DataFrame", src_col: str = "src_node",
                  dst_col: str = "dst_node") -> "pd_lib.DataFrame":
        """Replace node labels with encoded integer IDs (in-place safe)."""
        df = df.copy()
        if hasattr(df[src_col], "to_pandas"):
            src_mapped = df[src_col].to_pandas().map(self._label_to_id)
            dst_mapped = df[dst_col].to_pandas().map(self._label_to_id)
            df[src_col] = pd_lib.Series(src_mapped.values)
            df[dst_col] = pd_lib.Series(dst_mapped.values)
        else:
            df[src_col] = df[src_col].map(self._label_to_id)
            df[dst_col] = df[dst_col].map(self._label_to_id)
        return df

    def fit_transform(self, df: "pd_lib.DataFrame", src_col: str = "src_node",
                      dst_col: str = "dst_node") -> "pd_lib.DataFrame":
        """Convenience: fit then transform."""
        self.fit(df, src_col, dst_col)
        return self.transform(df, src_col, dst_col)

    def decode(self, encoded_ids):
        """Map encoded IDs back to original labels."""
        import numpy as np
        if isinstance(encoded_ids, (list, np.ndarray)):
            return [self._id_to_label.get(int(i), i) for i in encoded_ids]
        return self._id_to_label.get(int(encoded_ids), encoded_ids)

    @property
    def n_nodes(self) -> int:
        return self._next_id
