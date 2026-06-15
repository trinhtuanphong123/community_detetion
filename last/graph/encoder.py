"""
encoder.py — Stable node ID encoder for the AML graph pipeline.

Maps raw entity identifiers (int or string) to contiguous integer IDs
that remain consistent across time windows.

Guarantees:
- Direction  : src_node and dst_node are encoded independently; u → v is preserved.
- Time       : encoding is built once (fit) and reused across all windows.
- Memory     : no full DataFrame copy; only the two target columns are replaced.
               Mapping dict grows only with new nodes, never rebuilt.

Required by cuGraph / NetworkX: vertex IDs must be contiguous integers from 0.
"""

from __future__ import annotations

import numpy as np

try:
    import cudf as pd_lib
    _GPU = True
except ImportError:
    import pandas as pd_lib  # type: ignore
    _GPU = False


class NodeEncoder:
    """
    Maps raw node labels → contiguous integer IDs.

    The mapping is incremental: calling fit() multiple times (e.g., once
    per time window) extends the mapping without reassigning existing IDs.
    This guarantees stable encoding across the whole pipeline.

    Usage
    -----
    >>> enc = NodeEncoder()
    >>> df = enc.fit_transform(df, src_col="src_node", dst_col="dst_node")
    >>> enc.encode_column(series)   # encode a single Series
    >>> enc.decode(encoded_ids)     # reverse lookup
    >>> enc.n_nodes                 # total unique nodes seen so far
    """

    def __init__(self) -> None:
        self._label_to_id: dict = {}
        self._id_to_label: dict = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: "pd_lib.DataFrame",
        src_col: str = "src_node",
        dst_col: str = "dst_node",
    ) -> "NodeEncoder":
        """
        Learn mapping from all unique nodes in df[src_col] and df[dst_col].

        Incremental: new nodes are appended; existing IDs are never changed.
        Safe to call multiple times (once per window).

        Parameters
        ----------
        df : DataFrame
            Must contain src_col and dst_col.
        src_col, dst_col : str
            Column names for sender and receiver nodes.
        """
        src_vals, dst_vals = _unique_values(df, src_col, dst_col)
        all_nodes = np.union1d(src_vals, dst_vals)
        self._register(all_nodes)
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        df: "pd_lib.DataFrame",
        src_col: str = "src_node",
        dst_col: str = "dst_node",
    ) -> "pd_lib.DataFrame":
        """
        Replace raw node labels with encoded integer IDs.

        Modifies only src_col and dst_col in-place (no full DataFrame copy).
        Direction (src → dst) is preserved.

        Raises
        ------
        KeyError
            If a node in df has not been seen during fit().
            Call fit() or fit_transform() first.
        """
        df[src_col] = encode_series(df[src_col], self._label_to_id)
        df[dst_col] = encode_series(df[dst_col], self._label_to_id)
        return df

    def fit_transform(
        self,
        df: "pd_lib.DataFrame",
        src_col: str = "src_node",
        dst_col: str = "dst_node",
    ) -> "pd_lib.DataFrame":
        """Fit on df then transform it. Convenience wrapper."""
        self.fit(df, src_col, dst_col)
        return self.transform(df, src_col, dst_col)

    # ------------------------------------------------------------------
    # Single-column helper (used by temporal.py, second_order.py)
    # ------------------------------------------------------------------

    def encode_column(self, series: "pd_lib.Series") -> "pd_lib.Series":
        """
        Encode a single Series of raw node IDs.

        Any unseen node is registered on the fly (incremental fit).
        Useful when later modules pass node lists that may contain new nodes.

        Parameters
        ----------
        series : Series
            Raw node IDs (int or string).

        Returns
        -------
        Series of int64 encoded IDs, same index as input.
        """
        raw = _to_numpy(series)
        new_nodes = np.setdiff1d(raw, np.array(list(self._label_to_id.keys())))
        if len(new_nodes) > 0:
            self._register(new_nodes)
        return encode_series(series, self._label_to_id)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, encoded_ids) -> list:
        """
        Map encoded integer IDs back to original raw labels.

        Parameters
        ----------
        encoded_ids : int, list, or ndarray

        Returns
        -------
        Single label (if int input) or list of labels.
        """
        if np.isscalar(encoded_ids):
            return self._id_to_label.get(int(encoded_ids), encoded_ids)
        return [self._id_to_label.get(int(i), i) for i in encoded_ids]

    # ------------------------------------------------------------------
    # Properties / dunder helpers
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Total number of unique nodes registered so far."""
        return self._next_id

    def __len__(self) -> int:
        return self._next_id

    def __contains__(self, raw_id) -> bool:
        """True if raw_id has been registered."""
        return raw_id in self._label_to_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self, nodes: np.ndarray) -> None:
        """Add new nodes to the mapping. Vectorized; no Python loop."""
        # Filter to nodes not yet registered
        known = np.array(list(self._label_to_id.keys())) if self._label_to_id else np.array([])
        if len(known) > 0:
            new_nodes = np.setdiff1d(nodes, known)
        else:
            new_nodes = nodes

        if len(new_nodes) == 0:
            return

        # Assign contiguous IDs starting from _next_id
        new_ids = np.arange(self._next_id, self._next_id + len(new_nodes))
        for node, nid in zip(new_nodes.tolist(), new_ids.tolist()):
            self._label_to_id[node] = int(nid)
            self._id_to_label[int(nid)] = node

        self._next_id += len(new_nodes)


# ---------------------------------------------------------------------------
# Module-level helpers (reusable by temporal.py / second_order.py)
# ---------------------------------------------------------------------------

def _to_numpy(series: "pd_lib.Series") -> np.ndarray:
    """Return numpy array from a pandas or cuDF Series."""
    if hasattr(series, "to_pandas"):
        return series.to_pandas().to_numpy()
    return series.to_numpy()


def _unique_values(
    df: "pd_lib.DataFrame",
    src_col: str,
    dst_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique values from two columns as numpy arrays."""
    return _to_numpy(df[src_col]), _to_numpy(df[dst_col])


def encode_series(
    series: "pd_lib.Series",
    mapping: dict,
) -> "pd_lib.Series":
    """
    Map a Series of raw IDs through a label→id dict.

    Works for both pandas and cuDF by going through pandas .map(),
    then wrapping back into the correct Series type.

    Parameters
    ----------
    series : Series
        Raw node IDs.
    mapping : dict
        label → encoded int mapping from NodeEncoder.

    Returns
    -------
    Series of int64 encoded IDs.
    """
    if _GPU and hasattr(series, "to_pandas"):
        mapped = series.to_pandas().map(mapping)
        return pd_lib.Series(mapped.values, dtype="int64")
    return series.map(mapping).astype("int64")
