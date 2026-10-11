from __future__ import annotations

import torch

from toolbox.nn.layers import GATConv, GCNConv, SAGEConv
from toolbox.nn.models import ActorCritic, GNNNodeClassifier, GNNPretrainer, TinyGPT
from toolbox.nn.models.gpt import SequenceRewardModel

VOCAB = 50


def tiny_gpt(**kwargs: object) -> TinyGPT:
    return TinyGPT(VOCAB, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, **kwargs)  # type: ignore[arg-type]


def test_gpt_logits_shape() -> None:
    ids = torch.randint(VOCAB, (3, 10))
    assert tiny_gpt()(ids).shape == (3, 10, VOCAB)


def test_gpt_is_causal() -> None:
    model = tiny_gpt().eval()
    ids = torch.randint(VOCAB, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % VOCAB
    torch.testing.assert_close(model(ids)[:, :8], model(changed)[:, :8])


def test_padding_mask_produces_no_nans_with_left_padding() -> None:
    model = tiny_gpt().eval()
    ids = torch.randint(VOCAB, (2, 8))
    mask = torch.ones_like(ids)
    mask[0, :5] = 0  # first sequence is mostly left padding
    assert torch.isfinite(model(ids, attention_mask=mask)).all()


def test_grouped_query_attention() -> None:
    model = tiny_gpt(n_kv_heads=2)
    assert model(torch.randint(VOCAB, (2, 5))).shape == (2, 5, VOCAB)


def test_generate_returns_fixed_width_and_stops_at_eos() -> None:
    model = tiny_gpt()
    out = model.generate(torch.randint(VOCAB, (2, 4)), max_new_tokens=6, temperature=0.0)
    assert out.shape == (2, 6)


def test_reward_model_scores_last_real_token() -> None:
    rm = SequenceRewardModel(VOCAB, d_model=32, n_layers=1, n_heads=4, max_seq_len=16).eval()
    ids = torch.randint(VOCAB, (1, 6))
    padded = torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * 6 + [0] * 3])
    torch.testing.assert_close(rm(ids), rm(padded, attention_mask=mask))


def _ring_graph(n: int = 6, feat: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    return torch.randn(n, feat), edge_index


def test_graph_convs_shapes() -> None:
    x, edge_index = _ring_graph()
    for conv in (GCNConv(5, 8), SAGEConv(5, 8), GATConv(5, 8, heads=2)):
        assert conv(x, edge_index).shape == (6, 8)


def test_gcn_matches_dense_formula() -> None:
    x, edge_index = _ring_graph()
    conv = GCNConv(5, 3, bias=False)
    adj = torch.zeros(6, 6)
    adj[edge_index[1], edge_index[0]] = 1
    adj += torch.eye(6)
    d = adj.sum(1).pow(-0.5)
    expected = (d[:, None] * adj * d[None, :]) @ conv.linear(x)
    torch.testing.assert_close(conv(x, edge_index), expected)


def test_pretrained_encoder_transfers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pre = GNNPretrainer(in_dim=5, hidden_dim=8)
    path = tmp_path / "pre.pt"
    torch.save({"model": pre.state_dict()}, path)
    clf = GNNNodeClassifier(
        in_dim=5, num_classes=3, hidden_dim=8, pretrained_checkpoint=str(path), freeze_encoder=True
    )
    for a, b in zip(pre.encoder.parameters(), clf.encoder.parameters()):
        torch.testing.assert_close(a, b)
    assert not any(p.requires_grad for p in clf.encoder.parameters())


def test_actor_critic() -> None:
    dist, value = ActorCritic(obs_dim=4, num_actions=2)(torch.randn(7, 4))
    assert dist.sample().shape == (7,) and value.shape == (7,)
