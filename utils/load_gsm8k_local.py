# load_gsm8k_local.py
from pathlib import Path
from datasets import load_dataset, Dataset
from utils.response_format import verl_format_record

def _load_local_parquet(variant: str, split: str) -> Dataset | None:
    p = Path(f"data/ppo_data/gsm8k/{variant}/{split}-00000-of-00001.parquet")
    if p.exists() and p.is_file() and p.stat().st_size > 0:
        return load_dataset("parquet", data_files=str(p))["train"]
    return None

def _load_hf_backup(variant: str, split: str) -> Dataset:
    # HF repo mirroring GSM8K. You can change to 'openai/gsm8k' or your mirror.
    # Note: HF fallback is optional—remove this if you are strictly offline.
    name = "openai/gsm8k"
    subset = variant  # "main" or "socratic"
    return load_dataset(name, subset, split=split)

def load_gsm8k_parquet(variant: str = "main", split: str = "train") -> Dataset:
    """
    Returns a Dataset with columns:
      - prompt : str  (used for PPO rollouts)
      - label  : str  (ground-truth numeric answer for the verifier)
    """
    ds = _load_local_parquet(variant, split)
    if ds is None:
        ds = _load_hf_backup(variant, split)

    def _map(x):
        rec = verl_format_record(x["question"], x["answer"])
        return {"prompt": rec["prompt"], "label": rec["label"], "answer": x["answer"]}  # keep raw for debugging

    keep = {"prompt", "label", "answer"}  # answer kept only for offline checks
    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in keep])

if __name__ == "__main__":
    train_ds = load_gsm8k_parquet("main", "train")
    test_ds  = load_gsm8k_parquet("main", "test")
    print("Train sample:", train_ds[0].keys())
    print(train_ds[0]["prompt"][:300], "...\n<label>:", train_ds[0]["label"])
