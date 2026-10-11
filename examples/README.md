# Examples

Everything here is built on toolbox without changing it. Each folder is self-contained: code,
configs, tests and a README that says how to run it and what it shows. Its README is also a
page on the [docs site](https://lamng3.github.io/toolbox/examples/).

| Example | Area | Source | Folder |
| --- | --- | --- | --- |
| Recipes: RLHF pipeline, self-improvement, agent tracing, custom losses and layers | Workflows | toolbox | [`recipes/`](recipes/) |
| Graph model views (GMV): serve GNN outputs as views via L-bisimulation | Graph inference serving | Fan, Lu, Che & Wu, *Approximating Graph Inference by Serving Graph Model Outputs as Views* (2025) | [`gmv/`](gmv/) |
| Jev decision models: distill debiased or reasoned decisions into one forward pass | LLM decision models | Nokia Applied Research, [AnyJev](https://github.com/nokia-applied-research/AnyJev) (used as a library) | [`jev/`](jev/) |

Run an example from the repository root with `python -m examples.<folder>.<script>`, for
example `python -m examples.jev.run`. Run its tests with `pytest examples/<folder>`.

## Adding an example

Create `examples/<name>/` with `README.md`, `run.py`, `configs/` and `tests/`, and add a row
to the table above. The README becomes a docs page automatically. For a paper, the README
covers the method, the citation, results, and what is and isn't reproduced;
[`gmv/`](gmv/) is a good template.
