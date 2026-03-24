"""Fallback reward function for non-tool responses.

This is called by Verl's reward_manager when the model doesn't produce
a valid tool call. It just checks if <tool_call> exists in the response.
The REAL reward comes from FactorTool.execute() during multi-turn generation.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def compute_score(solution_str, ground_truth=None, method="flexible", format_score=0.0, score=1.0, **kwargs):
    """Score the response string. Returns 0-1."""
    response = solution_str

    # Check if response contains a valid tool call
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
    if tool_call_match:
        try:
            json.loads(tool_call_match.group(1))
            return 1.0  # Valid tool call structure
        except json.JSONDecodeError:
            try:
                fixed = tool_call_match.group(1).replace("'", '"')
                json.loads(fixed)
                return 0.2  # Fixable tool call
            except Exception:
                return 0.0
    return 0.0  # No tool call
