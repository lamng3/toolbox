# Examples

Techniques built on top of toolbox. Each folder is self-contained, with its code, configs,
tests and a README that covers the method, citation, how to run it, results, and what isn't
reproduced.

| Technique | Area | Source | Example |
| --- | --- | --- | --- |
| Graph model views (GMV): serve GNN outputs as views via L-bisimulation | Graph inference serving | Fan, Lu, Che & Wu, *Approximating Graph Inference by Serving Graph Model Outputs as Views* (2025) | [`gmv/`](gmv/) |

Run an example from the repository root with `python -m examples.<name>.run`.
Run its tests with `pytest examples/<name>`.

New examples follow the same layout: `examples/<name>/` with `README.md`, `run.py`,
`configs/` and `tests/`.
