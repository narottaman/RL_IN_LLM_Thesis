#!/usr/bin/env python3
"""
training_without_peft.py
Full finetune (no PEFT/LoRA) PPO on GSM8K via VERL.

- Prepares PPO-ready parquet using your utils:
    utils/load_gsm8k_local.py  -> loads/normalizes to {prompt, label} and raw 'answer'
    utils/response_format.py   -> enforces <reasoning>/<answer> style prompting, if you use it
- Uses your merged reward at utils/reward.py (compute_score), or VERL's built-in GSM8K scorer.

Example:
  python training_without_peft.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --variant main \
      --use-reward custom \
      --gpus 2 \
      --train-bsz 256 \
      --ppo-mb 64 \
      --ppo-micro-bsz 2
"""

import argparse
import os
import subprocess
from pathlib import Path

from utils.load_gsm8k_local import load_gsm8k_parquet

def _prep_split(variant: str, split: str) -> str:
    """
    Build a VERL-friendly parquet with columns:
      - prompt        : shown to policy during rollout
      - ground_truth  : numeric (string) final answer for reward
      - data_source   : tag string (e.g., 'openai/gsm8k')
    """
    ds = load_gsm8k_parquet(variant=variant, split=split)
    ds = ds.rename_columns({"label": "ground_truth"})
    ds = ds.add_column("data_source", ["openai/gsm8k"] * len(ds))

    out_dir = Path(f"data/ppo_data/gsm8k_prep/{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.parquet"
    ds.to_parquet(str(out_path))
    return str(out_path)

def build_verl_cmd(
    train_parquet: str,
    val_parquet: str,
    model_path: str,
    gpus: int,
    use_reward: str,
    exp_name: str,
    max_prompt: int,
    max_response: int,
    train_bsz: int,
    ppo_mb: int,
    ppo_micro_bsz: int,
):
    """
    Compose the VERL PPO command WITHOUT any LoRA/PEFT settings.
    """
    overrides = [
        # --- Data ---
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.max_prompt_length={max_prompt}",
        f"data.max_response_length={max_response}",
        f"data.train_batch_size={train_bsz}",

        # --- Models ---
        # Use same HF model for actor/rollout/ref and critic backbone
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",

        # --- Strategies & rollout engine ---
        # Full finetune commonly uses FSDP; adjust to your infra (fsdp/fsdp2/ddp)
        "actor_rollout_ref.actor.strategy=fsdp",
        "critic.strategy=fsdp",
        # vLLM is fine for rollouts here as well (sampling only)
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",

        # --- PPO batch/microbatch ---
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mb}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_bsz}",
        f"critic.ppo_micro_batch_size_per_gpu={ppo_micro_bsz}",

        # --- Optimization / regularization (conservative defaults) ---
        # Feel free to tune LR upward for full finetune; start small to stabilize.
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "critic.optim.lr=1e-5",
        "algorithm.kl_ctrl.kl_coef=0.001",

        # --- Misc (ref logprob, logging, saving, trainer) ---
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "trainer.logger=['console']",
        "trainer.val_before_train=False",
        "trainer.default_hdfs_dir=null",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.nnodes=1",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        "trainer.total_epochs=15",
        f"trainer.experiment_name={exp_name}",
    ]
    overrides += [
        "trainer.project_name=verl_gsm8k",
        "trainer.default_local_dir=checkpoints",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.resume_mode=auto",
        "checkpoint.contents=['model','optimizer','extra','hf_model']",
        "trainer.remove_previous_ckpt_in_save=False",
    ]
    # Reward selection
    if use_reward == "custom":
        overrides += [
            "custom_reward_function.path=utils/reward.py",
            "custom_reward_function.name=compute_score",
        ]
    elif use_reward == "builtin":
        # Use VERL’s built-in GSM8K reward (expects a parsable final numeric answer)
        pass
    else:
        raise ValueError("--use-reward must be one of: builtin, custom")

    cmd = ["python", "-u", "-m", "verl.trainer.main_ppo"] + overrides
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", default="main", choices=["main", "socratic"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--gpus", type=int, default=1)

    # Sequence lengths & batches
    parser.add_argument("--max-prompt", type=int, default=512)
    parser.add_argument("--max-response", type=int, default=256)
    parser.add_argument("--train-bsz", type=int, default=256)
    parser.add_argument("--ppo-mb", type=int, default=64)
    parser.add_argument("--ppo-micro-bsz", type=int, default=4)

    # Reward: builtin (VERL GSM8K) or custom (utils/reward.py::compute_score)
    parser.add_argument("--use-reward", default="custom", choices=["builtin", "custom"])

    parser.add_argument("--exp-name", default="qwen2.5-7b_gsm8k_ppo_fullfinetune")
    args = parser.parse_args()

    # 1) Prepare PPO-ready parquet files
    print(f"[prep] Building PPO parquet for {args.variant}:{args.train_split}/{args.val_split} ...")
    train_parquet = _prep_split(args.variant, args.train_split)
    val_parquet   = _prep_split(args.variant, args.val_split)

    # 2) Build the VERL command (NO PEFT)
    cmd = build_verl_cmd(
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        model_path=args.model,
        gpus=args.gpus,
        use_reward=args.use_reward,
        exp_name=args.exp_name,
        max_prompt=args.max_prompt,
        max_response=args.max_response,
        train_bsz=args.train_bsz,
        ppo_mb=args.ppo_mb,
        ppo_micro_bsz=args.ppo_micro_bsz,
    )

    print("[run] Executing:\n  ", " \\\n   ".join(cmd))
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    # if you pin devices: env["CUDA_VISIBLE_DEVICES"] = "0,1"
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()
