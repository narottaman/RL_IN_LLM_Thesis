#!/usr/bin/env python3
"""
PPO + PEFT LoRA training on GSM8K with VERL.

- Prepares PPO-ready parquet files using your utils:
    utils/load_gsm8k_local.py      -> loads local parquet into HF Dataset
    utils/response_format.py       -> builds prompts with <reasoning>/<answer>
- Runs verl.trainer.main_ppo with LoRA enabled.
- Lets you pick VERL's built-in GSM8K reward OR your adapter in utils/reward_adapter.py.

Usage:
  python training_with_peft.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --variant main \
      --train-split train \
      --val-split test \
      --use-reward adapter \
      --lora-rank 32 \
      --gpus 1

Notes:
- If you choose --use-reward adapter, ensure utils/reward_adapter.py
  exposes a function with signature:
      def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
  (It can internally call your gsm8k_with_tags.)
"""

import argparse
import os
import subprocess
from pathlib import Path

from datasets import DatasetDict
from utils.load_gsm8k_local import load_gsm8k_parquet

def _prep_split(variant: str, split: str) -> str:
    """
    Loads local GSM8K parquet via your utils, converts to PPO-ready structure,
    and writes a single parquet the VERL loader expects:
      - 'prompt'        (string shown to policy)
      - 'ground_truth'  (gold final numeric answer as string)
      - 'data_source'   (dataset name, used by VERL metrics; we set 'openai/gsm8k')
    Returns path to the written parquet file.
    """
    ds = load_gsm8k_parquet(variant=variant, split=split)  # has columns: prompt, label, answer (debug)
    # Rename/augment to VERL-friendly columns
    ds = ds.rename_columns({"label": "ground_truth"})
    ds = ds.add_column("data_source", ["openai/gsm8k"] * len(ds))  # stable name used in examples

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
    lora_rank: int,
    lora_alpha: int,
    use_reward: str,
    exp_name: str,
    max_prompt: int,
    max_response: int,
    train_bsz: int,
    ppo_mb: int,
    ppo_micro_bsz: int,
):
    """
    Compose a `python -m verl.trainer.main_ppo` command with LoRA & PPO flags.
    """
    overrides = [
        # --- data ---
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.max_prompt_length={max_prompt}",
        f"data.max_response_length={max_response}",
        f"data.train_batch_size={train_bsz}",

        # --- models (actor/rollout/ref share the same base hf model) ---
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",

        # --- FSDP + vLLM rollout (required for LoRA in VERL) ---
        "actor_rollout_ref.actor.strategy=fsdp",
        "critic.strategy=fsdp",
        "actor_rollout_ref.rollout.name=vllm",

        # --- LoRA knobs (must set for LoRA) ---
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        # VERL LoRA note: vLLM must load the *base* model; set load_format to safetensors
        "actor_rollout_ref.rollout.load_format=safetensors",

        # --- PPO scales (safe small defaults; adjust to your GPU) ---
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mb}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_bsz}",
        f"critic.ppo_micro_batch_size_per_gpu={ppo_micro_bsz}",

        # --- misc rollout/ref compute ---
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",

        # --- optimization / regularization ---
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "critic.optim.lr=1e-5",
        "algorithm.kl_ctrl.kl_coef=0.001",

        # --- trainer ---
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
        "trainer.save_freq=1",   # save every epoch
        "trainer.test_freq=1",   # optional
        "trainer.resume_mode=auto",
        "checkpoint.contents=['model','optimizer','extra','hf_model']",
        "trainer.remove_previous_ckpt_in_save=False",
    ]

    # Reward selection:
    #   - "builtin": VERL's GSM8K rule-based reward (expects #### <number>).
    #   - "adapter": your utils/reward_adapter.compute_score (parses <answer>...</answer>).
    if use_reward == "adapter":
        overrides += [
            "custom_reward_function.path=utils/reward.py.py",
            # Your file should define `compute_score(data_source, solution_str, ground_truth, extra_info=None)`
            # If your function name differs, set it here:
            "custom_reward_function.name=compute_score",
        ]
    elif use_reward == "builtin":
        # No custom reward override -> built-in GSM8K reward is used.
        pass
    else:
        raise ValueError("--use-reward must be one of: builtin, adapter")

    cmd = ["python", "-u", "-m", "verl.trainer.main_ppo"] + overrides
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variant", default="main", choices=["main", "socratic"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-prompt", type=int, default=512)
    parser.add_argument("--max-response", type=int, default=256)
    parser.add_argument("--train-bsz", type=int, default=256)
    parser.add_argument("--ppo-mb", type=int, default=64)
    parser.add_argument("--ppo-micro-bsz", type=int, default=4)
    parser.add_argument("--use-reward", default="adapter", choices=["builtin", "adapter"])
    parser.add_argument("--exp-name", default="qwen2.5-7b_gsm8k_ppo_lora")
    args = parser.parse_args()

    # 1) Prepare PPO-ready parquet (prompt / ground_truth / data_source)
    print(f"[prep] Building PPO parquet for {args.variant}:{args.train_split}/{args.val_split} ...")
    train_parquet = _prep_split(args.variant, args.train_split)
    val_parquet   = _prep_split(args.variant, args.val_split)

    # 2) Build VERL command with LoRA + PPO
    cmd = build_verl_cmd(
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        model_path=args.model,
        gpus=args.gpus,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
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
    # If your environment needs this to avoid vLLM / Ray device issues:
    env.setdefault("ENSURE_CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", ""))

    # 3) Launch training (streams logs to stdout)
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    main()
