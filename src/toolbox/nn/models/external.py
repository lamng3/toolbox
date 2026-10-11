"""Bring any model into toolbox training (pattern: *Adapter*).

* ``pyg_model`` — PyTorch Geometric's ready-made GNNs (``GCN``, ``GraphSAGE``, ``GAT``,
  ``GIN``, ``PNA``, ``EdgeCNN``) with their own layer, norm and jumping-knowledge options.
* ``torch_module`` — **any** ``nn.Module`` by import path, e.g. a model you are developing in
  your own package. No registration needed:

.. code-block:: yaml

    model:
      name: torch_module
      target: my_lab.models:SpectralGNN     # "module.path:ClassName"
      hidden: 128

Data-derived sizes (``in_dim``, ``num_classes``, ``vocab_size``, ``obs_dim``, ``num_actions``)
are passed only if the class accepts them, as with any other toolbox model. Registering the
class with ``@MODELS.register("spectral_gnn")`` is the alternative when you want a short name.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from torch import nn

from toolbox.core.decorators import require
from toolbox.core.registry import MODELS

INJECTED_SIZES = frozenset({"in_dim", "num_classes", "vocab_size", "obs_dim", "num_actions"})
TARGET_SEPARATOR = ":"


def import_target(target: str) -> Any:
    """Resolve ``"package.module:Attr"`` (or ``"package.module.Attr"``)."""
    if TARGET_SEPARATOR in target:
        module_name, attr = target.split(TARGET_SEPARATOR, 1)
    else:
        module_name, _, attr = target.rpartition(".")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _named_params(cls: Any) -> set[str]:
    return set(inspect.signature(cls).parameters)


@MODELS.register("torch_module")
def torch_module(target: str, **kwargs: Any) -> nn.Module:
    """Any ``nn.Module`` by import path (``"module.path:ClassName"``), built with ``kwargs``."""
    cls = import_target(target)
    named = _named_params(cls)
    # Injected sizes go only to classes that name them; everything the user wrote is kept.
    kwargs = {k: v for k, v in kwargs.items() if k in named or k not in INJECTED_SIZES}
    module: nn.Module = cls(**kwargs)
    return module


@MODELS.register("pyg_model")
def pyg_model(
    model: str = "GCN",
    in_dim: int | None = None,
    num_classes: int | None = None,
    hidden_channels: int = 64,
    num_layers: int = 2,
    **kwargs: Any,
) -> nn.Module:
    """A ``torch_geometric.nn.models`` GNN (``GCN``, ``GraphSAGE``, ``GAT``, ``GIN`` ...).

    ``in_dim`` / ``num_classes`` come from the dataset; other kwargs (``dropout``, ``norm``,
    ``jk``, ``heads`` ...) go to PyG unchanged.
    """
    models = require("torch_geometric.nn.models", extra="gnn")
    kwargs = {k: v for k, v in kwargs.items() if k not in INJECTED_SIZES}  # e.g. vocab_size
    module: nn.Module = getattr(models, model)(
        in_channels=in_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=num_classes,
        **kwargs,
    )
    return module
