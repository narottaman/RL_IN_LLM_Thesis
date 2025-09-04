# rewards.py
import re
from typing import List, Dict, Tuple

# If VERL exposes a ready-made verifier/reward for GSM8K, import it here.
# The exact path may vary depending on your setup/package name.
# Example (adjust to your install):
# from verl.tasks.gsm8k import gsm8k_verifier  # callable: (pred, label) -> float in [0,1]

NUM_RX = re.compile(r"^-?\d+(\.\d+)?$")

def extract_blocks(text: str) -> Tuple[str, str, Dict[str, bool]]:
    """
    Extract <reasoning>...</reasoning> and <answer>...</answer> blocks.
    Returns (reasoning_text, answer_text, flags).
    Flags include presence, multiplicity, and stray text checks.
    """
    flags = {
        "has_reasoning": False,
        "has_answer": False,
        "single_reasoning": False,
        "single_answer": False,
        "no_extraneous_outside": False,
    }

    # Find blocks (non-greedy, dotall)
    r_blocks = re.findall(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.DOTALL)
    a_blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)

    flags["has_reasoning"] = len(r_blocks) > 0
    flags["has_answer"] = len(a_blocks) > 0
    flags["single_reasoning"] = len(r_blocks) == 1
    flags["single_answer"] = len(a_blocks) == 1

    reasoning = r_blocks[0] if r_blocks else ""
    answer = a_blocks[0] if a_blocks else ""

    # Check for extraneous content outside the two blocks (tolerate whitespace)
    rebuilt = ""
    if r_blocks:
        rebuilt += f"<reasoning>\n{reasoning}\n</reasoning>"
    if a_blocks:
        rebuilt += f"<answer>\n{answer}\n</answer>"
    outside = text.replace(rebuilt, "")
    flags["no_extraneous_outside"] = outside.strip() == ""

    return reasoning, answer, flags

def is_numeric_single_line(s: str) -> bool:
    # Accepts a single number possibly surrounded by whitespace/newlines
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip() != ""]
    if len(lines) != 1:
        return False
    return bool(NUM_RX.match(lines[0]))

def safe_float(s: str):
    try:
        return float(s.strip())
    except Exception:
        return None

def format_reward(response: str, max_bonus: float = 0.3) -> float:
    """
    Tiny shaping reward (0..max_bonus) to encourage good formatting:
      +0.10 has both blocks
      +0.05 single (not repeated) blocks
      +0.10 numeric-only inside <answer>
      +0.05 nothing outside the blocks
    """
    _, answer, flags = extract_blocks(response)
    reward = 0.0
    if flags["has_reasoning"] and flags["has_answer"]:
        reward += 0.10
    if flags["single_reasoning"] and flags["single_answer"]:
        reward += 0.05
    if is_numeric_single_line(answer):
        reward += 0.10
    if flags["no_extraneous_outside"]:
        reward += 0.05
    return min(reward, max_bonus)

def exact_match_numeric(pred_num: float, gold_num: float) -> float:
    return 1.0 if pred_num == gold_num else 0.0

def tolerant_relative_match(pred_num: float, gold_num: float, rel_tol=1e-9, abs_tol=0.0) -> float:
    # Optional: a softer numeric equality (use only if you want tolerance)
    return 1.0 if abs(pred_num - gold_num) <= max(abs_tol, rel_tol * max(1.0, abs(gold_num))) else 0.0

def parse_numeric_from_answer_block(response: str):
    _, answer, _ = extract_blocks(response)
    if not is_numeric_single_line(answer):
        return None
    return safe_float(answer)

def gsm8k_correctness_reward(response: str, gold_label: str) -> float:
    """
    If you prefer using VERL’s built-in GSM8K verifier, call it instead of this function.
    This local version does strict numeric match on the <answer> number.
    """
    pred = parse_numeric_from_answer_block(response)
    gold = safe_float(gold_label)
    if pred is None or gold is None:
        return 0.0
    return exact_match_numeric(pred, gold)  # or tolerant_relative_match(pred, gold)

def combined_reward(
    responses: List[str],
    labels: List[str],
    correctness_weight: float = 1.0,
    format_weight: float = 1.0,
    use_verl_verifier: bool = True,
) -> List[float]:
    """
    Combine VERL GSM8K correctness reward with format shaping.
    - correctness_weight is usually >> format_weight (e.g., 1.0 vs 0.2–0.3).
    - If use_verl_verifier is True, call VERL’s verifier; otherwise use local strict check.
    """
    rewards = []
    for resp, gold in zip(responses, labels):
        # 1) Correctness (primary)
        if use_verl_verifier:
            # Example API; adjust to your actual import:
            # corr = float(gsm8k_verifier(resp, gold))  # expects 0/1 (or a score)
            # If your API expects plain numbers (no tags), feed only the extracted number:
            pred_num = parse_numeric_from_answer_block(resp)
            # corr = float(gsm8k_verifier(str(pred_num) if pred_num is not None else "", gold))
            # For illustration, fall back to local:
            corr = gsm8k_correctness_reward(resp, gold)
        else:
            corr = gsm8k_correctness_reward(resp, gold)

        # 2) Formatting (shaping)
        fmt = format_reward(resp)

        rewards.append(correctness_weight * corr + format_weight * fmt)
    return rewards
