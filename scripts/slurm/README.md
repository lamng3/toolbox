# SLURM

You normally never write an sbatch file by hand: `toolbox launch` renders one from
`src/toolbox/launch/templates/slurm.sbatch` using the `launcher=slurm` config.

```bash
# 1. Inspect the script that would be submitted
toolbox launch experiment=grpo_qwen_gsm8k launcher=slurm \
    launcher.partition=gpu launcher.account=myproj launcher.gpus_per_node=4 --dry-run

# 2. Submit it (script is saved to <output_dir>/<job_name>.sbatch, logs to <output_dir>/logs)
toolbox launch experiment=grpo_qwen_gsm8k launcher=slurm launcher.partition=gpu
```

What the generated job does:

* `srun torchrun` across `nodes × gpus_per_node` processes with a c10d rendezvous on the first node;
* forces `strategy=ddp` and `trainer.resume_from=auto`;
* traps `USR1` (sent 120 s before the time limit) and requeues itself, so long runs survive
  preemption and wall-time limits by resuming from `checkpoints/last.pt`.

Cluster-specific setup (modules, conda, scratch dirs) goes in `launcher.setup` and `launcher.env`;
put them in your own `configs/launcher/<cluster>.yaml` and pass `--config-dir configs`.
