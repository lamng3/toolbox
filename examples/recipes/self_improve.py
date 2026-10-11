"""Expert iteration: the model improves from its own (traced) successes.

Each round:
  1. GRPO from the current checkpoint; ``trace_rollouts`` logs every sample with its reward.
  2. SFT on the successful samples only (``data=traces mode=sft min_reward=1``).
  3. The SFT checkpoint seeds the next round.

    python -m examples.recipes.self_improve --rounds 2 --out runs/self_improve
"""

from __future__ import annotations

import argparse
from pathlib import Path

from toolbox.experiment import load_config, run_experiment


def checkpoint(run: Path) -> Path:
    return run / "checkpoints" / "last.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/self_improve"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    common = [f"trainer.device={args.device}"]

    warmup = args.out / "round0_sft"
    run_experiment(load_config(["experiment=sft_tiny", f"output_dir={warmup}", *common]))
    current = checkpoint(warmup)

    for round_ in range(1, args.rounds + 1):
        rl_dir = args.out / f"round{round_}_grpo"
        metrics = run_experiment(
            load_config(
                [
                    "experiment=grpo_tiny",
                    f"output_dir={rl_dir}",
                    f"model.init_from={current}",
                    *common,
                ]
            )
        )
        print(f"round {round_}: GRPO eval reward = {metrics.get('eval/reward')}")

        sft_dir = args.out / f"round{round_}_sft"
        try:
            run_experiment(
                load_config(
                    [
                        "experiment=sft_from_traces",
                        f"output_dir={sft_dir}",
                        f"data.path={rl_dir / 'traces'}",
                        f"model.init_from={checkpoint(rl_dir)}",
                        *common,
                    ]
                )
            )
            current = checkpoint(sft_dir)
        except ValueError as exc:  # no successful samples this round
            print(f"round {round_}: skipping SFT ({exc})")
            current = checkpoint(rl_dir)


if __name__ == "__main__":
    main()
