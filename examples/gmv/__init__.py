"""GMV: serving graph-model outputs as views (Fan, Lu, Che & Wu) — built on toolbox.

An independent re-implementation of the paper's core ideas; see ``examples/gmv/README.md``
for the citation and for what is and is not covered.
"""

from examples.gmv.bisim import refine, rounds_for
from examples.gmv.selection import VIEW_SELECTORS, ViewSelector
from examples.gmv.service import GMVService
from examples.gmv.views import GraphModelView, materialize_view, minimize, sample_views

__all__ = [
    "VIEW_SELECTORS",
    "GMVService",
    "GraphModelView",
    "ViewSelector",
    "materialize_view",
    "minimize",
    "refine",
    "rounds_for",
    "sample_views",
]
