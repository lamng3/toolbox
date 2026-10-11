# GPU VMs and SLURM

## A fresh GPU VM

```bash
git clone https://github.com/lamng3/toolbox.git && cd toolbox
bash scripts/setup_vm.sh            # uv venv + `.[llm,tracking]` + env report + 5-step smoke test
.venv/bin/toolbox train experiment=grpo_qwen_gsm8k
```

Multiple GPUs on one machine:

```bash
toolbox launch experiment=sft_qwen launcher=torchrun launcher.nproc_per_node=8
# runs: python -m torch.distributed.run --standalone --nproc-per-node=8 -m toolbox.cli train ... strategy=ddp
```

`strategy=fsdp` shards parameters, gradients and optimizer state for models that do not fit on
one GPU.

## SLURM

`toolbox launch` turns the experiment config into an `sbatch` script and submits it. Always look
at the script first:

```bash
toolbox launch experiment=grpo_qwen_gsm8k launcher=slurm \
    launcher.partition=gpu launcher.account=myproj \
    launcher.nodes=2 launcher.gpus_per_node=4 launcher.time=24:00:00 --dry-run
```

Drop `--dry-run` to submit. The script is saved to `<output_dir>/<job_name>.sbatch` and logs
go to `<output_dir>/logs/`.

What the generated job does:

1. `srun torchrun` launches `nodes × gpus_per_node` processes with a c10d rendezvous on the
   first node, and forces `strategy=ddp`.
2. It adds `trainer.resume_from=auto`, so every (re)start continues from
   `checkpoints/last.pt`.
3. It traps `USR1`, which SLURM sends 120 s before the time limit or on preemption, and
   requeues itself. Long runs survive wall-time limits without manual babysitting.
4. Checkpoints are written atomically (temp file + rename), so a killed job never leaves a
   corrupt `last.pt`.

### Cluster presets

Keep cluster details (modules, accounts, scratch directories) in your own config directory
instead of passing flags every time:

```yaml title="configs/launcher/mycluster.yaml"
name: slurm
job_name: ${name}
partition: gpu
account: my_account
gpus_per_node: 4
cpus_per_task: 32
mem: 256G
time: "24:00:00"
setup:
  - "module load cuda/12.4"
  - "cd ${oc.env:PWD}"
  - "source .venv/bin/activate"
env:
  HF_HOME: /scratch/${oc.env:USER}/hf
  WANDB_MODE: offline
```

```bash
toolbox launch --config-dir configs experiment=grpo_qwen_gsm8k launcher=mycluster
```

### Sweeps

Each `launch` call is a `Command` object. A loop of submissions is a sweep, and
`CommandInvoker.undo_all()` cancels them all:

```python
from toolbox.launch import CommandInvoker, SlurmSubmitCommand

invoker = CommandInvoker()
for lr in (1e-6, 3e-6, 1e-5):
    for seed in (0, 1, 2):
        invoker.run(SlurmSubmitCommand(
            [f"experiment=grpo_qwen_gsm8k", f"optimizer.lr={lr}", f"seed={seed}",
             f"output_dir=runs/sweep/lr{lr}_s{seed}"],
            output_dir=f"runs/sweep/lr{lr}_s{seed}", job_name=f"grpo-{lr}-{seed}",
            partition="gpu", gpus_per_node=1,
        ))
# changed your mind?
invoker.undo_all()   # scancel every submitted job
```
