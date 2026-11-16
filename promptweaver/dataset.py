"""
Reads the ground-truth plans of the TinyAgent dataset
(https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset).
"""

import json
from typing import List


def load_plans(path: str) -> List[List[str]]:
    """
    Returns the tool names called in the ground-truth plan of each sample
    (excluding the final `join`). Samples whose answer is not a plan are skipped.
    """
    with open(path) as f:
        data = json.load(f)
    plans = []
    for sample in data.values():
        output = sample["output"][0]
        if output["type"] != "plan":
            continue
        tools = [
            step["tool_name"]
            for step in output["parsed_output"]
            if step["tool_name"] != "join"
        ]
        if tools:
            plans.append(tools)
    return plans
