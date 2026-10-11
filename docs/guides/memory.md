# Memory and self-improvement

There are two ways for a model to improve over time from experience.

## In context: episodic memory

`MemoryStore` keeps `(state, action, reward)` experiences and retrieves similar ones. The best
ones go into the prompt, and no weights change.

```python
from toolbox.memory import MemoryItem, MemoryStore, augment_prompt

memory = MemoryStore("runs/agent/memory", min_reward=1.0)
memory.add([MemoryItem("Q: 12+7=? A:", " 19", reward=1.0)])
prompt = augment_prompt(question, memory.search(question, k=3))
```

The embedder is a strategy. The default `HashingEmbedder` (word n-grams) needs no model. You
can plug in a sentence-transformer or the policy's hidden states by implementing
`Embedder.__call__(texts) -> (N, dim)`.

During RL the `memory_writer` callback stores every rollout with `reward ≥ min_reward`.

## In weights: expert iteration

1. Run GRPO with the `trace_rollouts` callback, which logs every sample and its reward.
2. Run SFT on the successful samples only (`data=traces mode=sft min_reward=1`).
3. Repeat from the new checkpoint.

```bash
python -m examples.recipes.self_improve --rounds 3
```

This is the STaR / ReST / rejection-sampling fine-tuning family. The same trace log also
gives DPO pairs (`mode=preference`) for free.
