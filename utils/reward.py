# utils/reward.py
# One-file reward for VERL PPO on GSM8K.
# - Entry point: compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
# - Combines correctness reward (dominant) + small format-shaping bonus
# - Works with <reasoning>...</reasoning><answer>...</answer> OR '#### 123' style
# - Optionally calls VERL's builtin GSM8K scorer if available

from __future__ import annotations
import os
import re
from typing import Optional, Tuple, Dict

# ----------------------------
# Config (env overridable)
# ----------------------------
CORRECTNESS_WEIGHT = float(os.getenv("REWARD_CORRECTNESS_WEIGHT", "1.0"))
FORMAT_WEIGHT      = float(os.getenv("REWARD_FORMAT_WEIGHT", "0.2"))
MAX_FORMAT_BONUS   = float(os.getenv("REWARD_MAX_FORMAT_BONUS", "0.3"))
USE_TOLERANCE      = os.getenv("REWARD_USE_TOLERANCE", "0").strip() in {"1", "true", "True"}
REL_TOL            = float(os.getenv("REWARD_REL_TOL", "1e-9"))
ABS_TOL            = float(os.getenv("REWARD_ABS_TOL", "0.0"))

# Try to import VERL builtin scorer (optional)
_VERL_SCORER = None
try:
    from verl.utils.reward_score import gsm8k as _VERL_GSM8K_SCORER  # type: ignore
    _VERL_SCORER = _VERL_GSM8K_SCORER
except Exception:
    _VERL_SCORER = None

# ----------------------------
# Regex helpers
# ----------------------------
ANSWER_TAG_RX = re.compile(r"<answer>\s*([\-]?\d+(?:\.\d+)?)\s*</answer>", re.S)
REASONING_BLOCK_RX = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.S)
ANSWER_BLOCK_RX = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
HASH_RX = re.compile(r"####\s*([\-]?\d+(?:\.\d+)?)\s*$")
NUM_RX = re.compile(r"^-?\d+(\.\d+)?$")

# ----------------------------
# Parsing / extraction
# ----------------------------
def _extract_number_from_answer_tag(text: str) -> Optional[str]:
    m = ANSWER_TAG_RX.search(text or "")
    return m.group(1).strip() if m else None

def _extract_number_from_hash(text: str) -> Optional[str]:
    m = HASH_RX.search((text or "").strip())
    return m.group(1).strip() if m else None

def _safe_float(s: Optional[str]) -> Optional[float]:
    try:
        if s is None: return None
        return float(s.strip())
    except Exception:
        return None

def _extract_blocks(text: str) -> Tuple[str, str, Dict[str, bool]]:
    flags = {
        "has_reasoning": False,
        "has_answer": False,
        "single_reasoning": False,
        "single_answer": False,
        "no_extraneous_outside": False,
    }
    r_blocks = REASONING_BLOCK_RX.findall(text or "")
    a_blocks = ANSWER_BLOCK_RX.findall(text or "")

    flags["has_reasoning"] = len(r_blocks) > 0
    flags["has_answer"] = len(a_blocks) > 0
    flags["single_reasoning"] = len(r_blocks) == 1
    flags["single_answer"] = len(a_blocks) == 1

    reasoning = r_blocks[0] if r_blocks else ""
    answer = a_blocks[0] if a_blocks else ""

    rebuilt = ""
    if r_blocks:
        rebuilt += f"<reasoning>\n{reasoning}\n</reasoning>"
    if a_blocks:
        rebuilt += f"<answer>\n{answer}\n</answer>"
    outside = (text or "").replace(rebuilt, "")
    flags["no_extraneous_outside"] = outside.strip() == ""
    return reasoning, answer, flags

def _is_numeric_single_line(s: str) -> bool:
    lines = [ln.strip() for ln in (s or "").strip().splitlines() if ln.strip()]
    return len(lines) == 1 and bool(NUM_RX.match(lines[0]))

# ----------------------------
# Correctness reward
# ----------------------------
def _exact_match(a: float, b: float) -> float:
    return 1.0 if a == b else 0.0

def _tolerant_match(a: float, b: float, rel: float, ab: float) -> float:
    return 1.0 if abs(a - b) <= max(ab, rel * max(1.0, abs(b))) else 0.0

def _correctness_reward(solution_str: str, gold_label: str) -> float:
    """
    Prefer VERL's builtin scorer if available; otherwise:
    - parse <answer>number</answer> OR #### number
    - strict (or tolerant) numeric equality
    """
    # If VERL scorer exists, try to pass something it can parse:
    if _VERL_SCORER is not None:
        # Prefer sending just the number if we can extract it
        num = _extract_number_from_answer_tag(solution_str) or _extract_number_from_hash(solution_str)
        candidate = num if num is not None else solution_str
        try:
            return float(_VERL_SCORER(solution_str=candidate, ground_truth=gold_label))
        except Exception:
            # Fall through to local check if builtin raised
            pass

    # Local numeric check
    pred_num = _safe_float(_extract_number_from_answer_tag(solution_str) or _extract_number_from_hash(solution_str))
    gold_num = _safe_float(gold_label)
    if pred_num is None or gold_num is None:
        return 0.0
    return _tolerant_match(pred_num, gold_num, REL_TOL, ABS_TOL) if USE_TOLERANCE else _exact_match(pred_num, gold_num)

# ----------------------------
# Format shaping (small bonus)
# ----------------------------
def _format_reward(solution_str: str, max_bonus: float = MAX_FORMAT_BONUS) -> float:
    """
    0 .. max_bonus
      +0.10 has both <reasoning> and <answer>
      +0.05 single (not repeated) blocks
      +0.10 numeric-only inside <answer>
      +0.05 nothing outside the two blocks
    """
    _, answer, flags = _extract_blocks(solution_str or "")
    reward = 0.0
    if flags["has_reasoning"] and flags["has_answer"]:
        reward += 0.10
    if flags["single_reasoning"] and flags["single_answer"]:
        reward += 0.05
    if _is_numeric_single_line(answer):
        reward += 0.10
    if flags["no_extraneous_outside"]:
        reward += 0.05
    return min(reward, max_bonus)

# ----------------------------
# VERL entrypoint
# ----------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None
) -> float:
    """
    Called by VERL once per sample.
    Returns a scalar reward (float).
    """
    corr = _correctness_reward(solution_str, ground_truth)
    fmt  = _format_reward(solution_str)
    return CORRECTNESS_WEIGHT * corr + FORMAT_WEIGHT * fmt
