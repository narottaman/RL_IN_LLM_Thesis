# rewards_adapter.py
import re
from typing import Optional
from verl.utils.reward_score import gsm8k as gsm8k_score  # type: ignore # builtin scorer

ANSWER_RX = re.compile(r"<answer>\s*([\-]?\d+(?:\.\d+)?)\s*</answer>", re.S)

def extract_number_from_tagged(response: str) -> Optional[str]:
    m = ANSWER_RX.search(response)
    return m.group(1).strip() if m else None

def gsm8k_with_tags(response: str, gold_label: str) -> float:
    """
    Use VERL's built-in GSM8K scorer, but allow <answer>123</answer> format.
    Falls back to raw response if tags weren't present.
    """
    num = extract_number_from_tagged(response)
    candidate = num if num is not None else response  # let builtin try its regexes
    # builtin returns 1.0 for correct, 0.0 otherwise
    return float(gsm8k_score(solution_str=candidate, ground_truth=gold_label))
