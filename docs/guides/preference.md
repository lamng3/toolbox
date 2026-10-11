# Preference optimisation and reward models

## Data

```json
{"prompt": "Q: 3+4=? A:", "chosen": " 7", "rejected": " 8"}
```

## DPO family (offline)

The `preference` trainer computes summed log-probs of the chosen ($y_w$) and rejected ($y_l$)
responses under the policy and a **frozen copy** of the initial model (the reference). It then
applies a loss to the implicit reward margin:

$$
h = \beta\Big[\big(\log\pi_\theta(y_w) - \log\pi_{\text{ref}}(y_w)\big) - \big(\log\pi_\theta(y_l) - \log\pi_{\text{ref}}(y_l)\big)\Big]
$$

| `objective=` | Loss | Notes |
| --- | --- | --- |
| `dpo` | $-\log\sigma(h)$ | `label_smoothing` = conservative DPO |
| `ipo` | $(h/\beta - \tfrac{1}{2\beta})^2$ | robust to deterministic preferences |
| `slic` | $\max(0, 1 - h)$ | hinge |

```bash
toolbox train experiment=dpo_tiny model.init_from=runs/sft_tiny/checkpoints/last.pt
toolbox train experiment=dpo_qwen objective=ipo objective.beta=0.5
toolbox train experiment=dpo_tiny trainer.length_normalize=true   # mean instead of sum log-probs
```

Logged metrics: `reward_margin`, `reward_accuracy` (fraction with $h>0$), `chosen_reward`,
`rejected_reward`.

!!! tip
    Start DPO from an SFT checkpoint (`model.init_from`). The reference model is a copy of
    whatever the policy starts from.

## Reward models (RLHF stage 2)

`experiment=reward_model_tiny` trains a scalar head on a transformer backbone with the
Bradley–Terry loss $-\log\sigma(r(x,y_w) - r(x,y_l))$. Use the result as a learned reward
for online RL:

```bash
toolbox train experiment=reward_model_tiny
toolbox train experiment=grpo_tiny reward=learned \
    reward.checkpoint=runs/reward_model_tiny/checkpoints/last.pt
```
