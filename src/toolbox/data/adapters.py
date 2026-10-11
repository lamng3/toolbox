"""Adapters from third-party data containers to toolbox formats (pattern: *Adapter*)."""

from __future__ import annotations

from typing import Any

import torch

GRAPH_KEYS = ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask")


def pyg_to_graph(data: Any) -> dict[str, torch.Tensor]:
    """Convert a ``torch_geometric.data.Data`` object to the toolbox graph dict."""
    return {key: getattr(data, key) for key in GRAPH_KEYS if getattr(data, key, None) is not None}


def hf_rows_to_records(rows: Any, mapping: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Rename columns of a HF ``Dataset`` (or any iterable of dicts) to toolbox field names.

    Example: ``hf_rows_to_records(gsm8k, {"question": "prompt", "answer": "answer"})``.
    """
    mapping = mapping or {}
    return [{str(mapping.get(k, k)): v for k, v in dict(row).items()} for row in rows]
