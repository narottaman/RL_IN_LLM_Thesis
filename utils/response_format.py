# utils/response_format.py
import re
from typing import Dict

# System instruction for VERL PPO runs on GSM8K
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
Explain your steps concisely.
</reasoning>
<answer>
THE_FINAL_NUMERIC_ANSWER_ONLY
</answer>"""

def extract_hash_answer(gsm8k_solution_text: str) -> str | None:
    """
    GSM8K solutions end with a line like: '#### 1234'
    Returns the text after '####' trimmed, or None if not found.
    """
    m = re.search(r"####\s*(.+)$", gsm8k_solution_text.strip())
    return m.group(1).strip() if m else None

def build_prompt(question: str) -> str:
    """
    Build the full prompt that instructs the model to answer
    using <reasoning> and <answer> blocks.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        "Give your solution now."
    )

def verl_format_record(question: str, solution_text: str) -> Dict[str, str]:
    """
    Prepare a PPO sample:
    - 'prompt' is shown to the policy for generation.
    - 'label' is the ground-truth final numeric answer (for the verifier).
    DO NOT include any gold completion for PPO.
    """
    final = extract_hash_answer(solution_text)
    return {
        "prompt": build_prompt(question),
        "label": final if final is not None else ""
    }
