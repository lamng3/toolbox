# Recipes

End-to-end workflows built only from the public API and configs. Copy one, change it, run it.

| Recipe | What it shows |
| --- | --- |
| `rlhf_pipeline.sh` | SFT → reward model → DPO and GRPO (verifiable *and* learned reward), each stage initialised from the previous one. |
| `self_improve.py` | Expert iteration: alternate GRPO rounds with SFT on the policy's own successful, traced rollouts. |
| `agent_tracing.py` | Log an agent's decisions as atomic actions, attach delayed rewards, then train a decision model on the trace. |
| `custom_objective.py` | Register a new loss as a plugin and train with it, without touching toolbox code. |
| `custom_layer.py` | Replace model internals (a transformer feed-forward, a GNN head) with your own layers and compare against the built-ins. |

All of them run on a laptop CPU in a minute or two with the tiny models; switch to
`model=qwen2_5_0_5b_lora tokenizer=qwen2_5` on a GPU for real results.
