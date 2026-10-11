# Examples

The `examples/` folder holds research examples built **on top of** toolbox. Each one takes an
idea, often a paper, and turns it into a runnable, tested experiment without touching library
code. It registers its own components, trains through `ExperimentBuilder` with its own configs,
and reuses toolbox models, trainers, tracing and repositories for the rest.

## GMV: serving graph-model outputs as views

An independent re-implementation of *Approximating Graph Inference by Serving Graph Model
Outputs as Views* (Yangxin Fan, Mingjian Lu, Haolai Che, Yinghui Wu; Case Western Reserve
University, 2025; [code](https://github.com/Yangxin666/GMV)).

Past GNN inference results become queryable views. A new node query is answered from an
L-bisimilar counterpart's cached output, which is exact for deterministic L-layer GNNs.
Otherwise the view gives an answer with a bisimulation-depth certificate, or the service falls
back to local inference.

```bash
python -m examples.gmv.run                                  # GCN, exact answers only
python -m examples.gmv.run model.conv=sage --min-rounds 1   # certified approximations
pytest tests/examples/test_gmv.py
```

| Model | Answers allowed | Served from views (GVSel) | Fidelity | Random selection |
| --- | --- | --- | --- | --- |
| GCN | exact | 11.0% | 100% | 0.3% |
| GCN | ≥ 2 rounds matched | 44.8% | 98.0% | 5.8% |
| GraphSAGE | exact | 52.0% | 100% | 6.2% |
| GraphSAGE | ≥ 1 round matched | 95.5% | 98.5% | 54.9% |

The full write-up covers the method mapping, the exactness checks (including why GCN needs
one more refinement round than GraphSAGE or GAT), the latency caveats at small scale, and what
is not implemented. It is in
[`examples/gmv/README.md`](https://github.com/lamng3/toolbox/tree/main/examples/gmv).
