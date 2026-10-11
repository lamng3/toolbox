from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from toolbox.core.constants import IGNORE_INDEX
from toolbox.objectives import (
    ClippedPolicyLoss,
    DPOLoss,
    GRPOLoss,
    IPOLoss,
    ReinforceLoss,
    kl_k3,
    token_logprobs,
)
from toolbox.objectives.policy import SEQ_MEAN_TOKEN_MEAN, aggregate
from toolbox.rl import GroupRelativeAdvantage, LeaveOneOutAdvantage, compute_gae


def test_dpo_matches_closed_form() -> None:
    pc, pr, rc, rr = (torch.tensor([v]) for v in (-1.0, -3.0, -2.0, -2.5))
    beta = 0.5
    margin = (pc - rc) - (pr - rr)  # 1.0 - (-0.5) = 1.5
    out = DPOLoss(beta=beta)(pc, pr, rc, rr)
    assert out.loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-beta * 1.5))))
    assert out.metrics["reward_accuracy"] == 1.0
    assert float(margin) == pytest.approx(1.5)


def test_dpo_zero_margin_is_log2() -> None:
    zeros = torch.zeros(4)
    assert DPOLoss()(zeros, zeros, zeros, zeros).loss.item() == pytest.approx(math.log(2))


def test_ipo_minimum_at_target_margin() -> None:
    beta = 0.1
    target = torch.tensor([1 / (2 * beta)])
    zeros = torch.zeros(1)
    assert IPOLoss(beta)(target, zeros, zeros, zeros).loss.item() == pytest.approx(0.0)


def test_token_logprobs_ignores_masked_labels() -> None:
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[1, IGNORE_INDEX, 4]])
    out = token_logprobs(logits, labels)
    expected = F.log_softmax(logits, -1)[0, [0, 2], [1, 4]]
    assert out[0, 1] == 0
    torch.testing.assert_close(out[0, [0, 2]], expected)


def test_clipped_loss_equals_pg_when_on_policy() -> None:
    logprobs = torch.randn(2, 4, requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 4)
    out = ClippedPolicyLoss()(logprobs, logprobs.detach(), adv, mask)
    out.loss.backward()
    # d/dθ of -A·ratio at ratio=1 is -A/num_tokens per token
    expected = -adv[:, None].expand(2, 4) / 8
    torch.testing.assert_close(logprobs.grad, expected)
    assert out.metrics["clip_fraction"] == 0.0


def test_clipping_blocks_large_ratios() -> None:
    old = torch.zeros(1, 1)
    new = torch.full((1, 1), math.log(2.0), requires_grad=True)  # ratio 2 > 1.2
    out = ClippedPolicyLoss(clip_eps=0.2)(new, old, torch.tensor([1.0]), torch.ones(1, 1))
    out.loss.backward()
    assert out.loss.item() == pytest.approx(-1.2)
    assert new.grad is not None and new.grad.abs().sum() == 0


def test_grpo_requires_reference_and_adds_kl() -> None:
    lp = torch.zeros(1, 2)
    loss = GRPOLoss(kl_coef=0.1)
    assert loss.requires_reference
    out = loss(lp, lp, torch.tensor([0.0]), torch.ones(1, 2), ref_logprobs=lp - 1.0)
    assert out.metrics["kl"] > 0
    with pytest.raises(ValueError):
        loss(lp, lp, torch.tensor([0.0]), torch.ones(1, 2))


def test_kl_k3_is_zero_iff_equal() -> None:
    lp = torch.randn(10)
    assert torch.allclose(kl_k3(lp, lp), torch.zeros(10))
    assert (kl_k3(lp, lp + 0.3) > 0).all()


def test_seq_mean_token_mean_weights_sequences_equally() -> None:
    values = torch.tensor([[1.0, 1.0, 1.0, 1.0], [5.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1, 1, 1], [1, 0, 0, 0]])
    assert aggregate(values, mask, SEQ_MEAN_TOKEN_MEAN).item() == pytest.approx(3.0)


def test_reinforce() -> None:
    lp = torch.tensor([[-1.0]], requires_grad=True)
    ReinforceLoss()(lp, lp.detach(), torch.tensor([2.0]), torch.ones(1, 1)).loss.backward()
    assert lp.grad is not None and lp.grad.item() == pytest.approx(-2.0)


def test_group_relative_advantage() -> None:
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    adv = GroupRelativeAdvantage()(rewards, group_size=3)
    assert adv[:3].sum().item() == pytest.approx(0.0, abs=1e-5)
    assert adv[0] > 0 > adv[1]
    unscaled = GroupRelativeAdvantage(scale=False)(rewards, 3)
    assert unscaled[0].item() == pytest.approx(2 / 3)


def test_rloo_baseline_excludes_self() -> None:
    adv = LeaveOneOutAdvantage()(torch.tensor([1.0, 0.0, 0.0]), group_size=3)
    torch.testing.assert_close(adv, torch.tensor([1.0, -0.5, -0.5]))


def test_gae_with_lambda_one_is_discounted_return_minus_value() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    adv, ret = compute_gae(rewards, values, dones, torch.tensor(9.0), gamma=0.9, lam=1.0)
    expected_returns = torch.tensor([1 + 0.9 + 0.81, 1 + 0.9, 1.0])
    torch.testing.assert_close(ret, expected_returns)
    torch.testing.assert_close(adv, expected_returns - values)
