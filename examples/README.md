# Examples

Worked research examples built **on top of** toolbox. They show how to take an idea, often a
paper, and turn it into an experiment without forking or editing the library. Each one:

1. **registers** what it needs (datasets, strategies) with toolbox registries;
2. **trains** through `ExperimentBuilder` with its own configs (`--config-dir`);
3. **reuses** toolbox models, trainers, tracing and repositories for everything else;
4. ships **tests** in `tests/examples/` that check its central claim;
5. **cites** its source and states what is and isn't reproduced.

| Example | Idea | Reference |
| --- | --- | --- |
| [`gmv/`](gmv/) | Serve GNN outputs as *views*: answer node queries from bisimilar counterparts in cached inference results | Fan, Lu, Che & Wu, *Approximating Graph Inference by Serving Graph Model Outputs as Views* ([code](https://github.com/Yangxin666/GMV)) |

Run any example as a module from the repository root, e.g. `python -m examples.gmv.run`.

## Adding one

```
examples/<name>/
├── README.md        what, why, citation, how to run, results, gaps
├── __init__.py
├── data.py          @DATASETS.register(...) if the example needs its own data
├── <method>.py      the technique, as small registered components
├── run.py           train via ExperimentBuilder, evaluate, print and save results
└── configs/experiment/<name>.yaml
tests/examples/test_<name>.py
```

When an example's components prove generally useful, promote them into `src/toolbox/`.
