# Pretraining

Pretraining is included as a **minimal, readable example**:

```bash
toolbox train experiment=lm_pretrain_tiny
```

* `text_lm` packs a corpus into fixed-length blocks, with BOS/EOS around each document.
* `tiny_gpt` is a Llama-style decoder (RoPE, RMSNorm, SwiGLU, grouped-query attention, tied
  embeddings) of about 0.3M parameters. It is small enough to read in one sitting and to test
  on CPU.

For real pretraining runs, use frameworks built for that scale:

| Framework | Notes |
| --- | --- |
| [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch-native, FSDP2/TP/PP, Llama-style models |
| [Megatron-LM / NeMo](https://github.com/NVIDIA/Megatron-LM) | the reference for very large-scale training |
| [nanotron](https://github.com/huggingface/nanotron) | Hugging Face's minimalistic 3D-parallel trainer |
| [OLMo](https://github.com/allenai/OLMo) | fully open data + training code |
| [nanoGPT](https://github.com/karpathy/nanoGPT) / modded-nanoGPT | learning and speedrunning |

Checkpoints from any of them load into toolbox post-training through Hugging Face
(`model.model_name_or_path=...`).
