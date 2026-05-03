print("=== Test 2: tool-use SFT message formatting (5-turn conversation structure) ===")
print("Objective : Verify _format_data_AngleDistanceTask_tooluse produces a correctly structured")
print("            5-turn conversation for a distance sample. Uses mock patches so no real")
print("            dataset or model download is required.")
print("Expected  :")
print("  turn count  -> 5 (system, user, assistant[tool_call], tool, assistant[answer])")
print("  role order  -> system -> user -> assistant -> tool -> assistant")
print("  turn 3 text -> contains <tool_call>")
print("  turn 5 text -> contains <answer>")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("src").resolve()))

from unittest.mock import patch

from medvision_bm.sft.sft_utils import _format_data_AngleDistanceTask_tooluse

VALUES_DICT = {
    "metric_type": "distance",
    "<landmark 1>": "A", "<landmark 2>": "B",
    "<x1>": "0.100", "<y1>": "0.200",
    "<x2>": "0.500", "<y2>": "0.800",
    "<pixel_width>": "0.500", "<pixel_height>": "0.500",
    "<image_width>": "512", "<image_height>": "512",
    "<distance>": "102.400",
}
EXAMPLE = {"biometric_profile": {"metric_type": "distance"}}
MOCK_PROMPT = "mock prompt. Report the reasoning process in <think> </think> tags."

with patch("medvision_bm.sft.sft_utils._doc_to_text_AngleDistanceTask_CoT",
           return_value=(MOCK_PROMPT, VALUES_DICT)), \
     patch("medvision_bm.sft.sft_utils.safe_exec_python", return_value="102.4"):
    result = _format_data_AngleDistanceTask_tooluse(
        EXAMPLE,
        model_name="Qwen2.5-VL-7B-Instruct",
        model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
    )

msgs = result["messages"]
roles = [m["role"] for m in msgs]

# --- turn count ---
assert len(msgs) == 5, f"expected 5 turns, got {len(msgs)}: {roles}"
print(f"  turn count           : {len(msgs)}  PASS")

# --- role order ---
expected_roles = ["system", "user", "assistant", "tool", "assistant"]
assert roles == expected_roles, f"role order wrong: {roles}"
print(f"  role order           : {roles}  PASS")

# --- turn 3: assistant emits tool call ---
turn3_text = msgs[2]["content"][0]["text"]
assert "<tool_call>" in turn3_text, f"turn 3 missing <tool_call>: {turn3_text[:80]!r}"
print(f"  turn 3 has <tool_call>: {turn3_text[:60]!r}...  PASS")

# --- turn 5: assistant emits final answer ---
turn5_text = msgs[4]["content"][0]["text"]
assert "<answer>" in turn5_text, f"turn 5 missing <answer>: {turn5_text[:80]!r}"
print(f"  turn 5 has <answer>  : {turn5_text[:60]!r}...  PASS")

print("OK")
