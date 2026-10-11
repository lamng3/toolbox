# Graph neural networks

The workflow mirrors LLMs: **self-supervised pretraining** of an encoder, then **post-training**
on the downstream task.

```bash
toolbox train experiment=gnn_pretrain                      # GraphMAE masked-feature reconstruction
toolbox train experiment=gnn_finetune                      # loads runs/gnn_pretrain/checkpoints/last.pt
toolbox train experiment=gnn_finetune model.pretrained_checkpoint=null    # from-scratch baseline
toolbox train experiment=gnn_finetune model.freeze_encoder=true           # linear probe
```

## Pieces

| Component | Name | Notes |
| --- | --- | --- |
| Layers | `GCNConv`, `SAGEConv`, `GATConv` | pure PyTorch (`index_add_`), so PyG is not needed |
| Encoder | `gnn_encoder` | `conv`, `activation`, `norm` are [swappable layers](layers.md); `num_layers`, `hidden_dim` |
| Pretraining model | `gnn_pretrainer` | encoder + feature decoder + projection head + mask token |
| Classifier | `gnn_node_classifier` | encoder + linear head; `pretrained_checkpoint`, `freeze_encoder` |
| Objectives | `masked_feature` (GraphMAE), `graph_contrastive` (GRACE), `node_classification` (trains on `train_mask`, reports `eval/accuracy` on `val_mask`) | |
| Data | `synthetic_graph` (stochastic block model, offline), `planetoid` (Cora/CiteSeer/PubMed via PyG) | |

Graphs are plain dicts (`x`, `edge_index`, `y`, `*_mask`) in the PyG `edge_index` convention.
`pyg_to_graph` adapts PyG `Data` objects.

```bash
uv pip install -e ".[gnn]"
toolbox train experiment=gnn_pretrain data=planetoid data.dataset=CiteSeer
```

`in_dim` and `num_classes` come from the dataset and are injected into the model, so you never
set them by hand.
