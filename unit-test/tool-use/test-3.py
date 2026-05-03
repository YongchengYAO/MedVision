print("=== Test 3: VLLM_Qwen25VL_ToolUse helper methods ===")
print("Objective : Verify _pick_instruct, _transform_prompt, and _extract_code static/class")
print("            methods on VLLM_Qwen25VL_ToolUse. vllm/transformers/decord are mocked so")
print("            no GPU or model download is needed.")
print("Expected  :")
print("  _pick_instruct      -> routes TL / angle / distance to the correct instruct constant")
print("  _transform_prompt   -> strips CoT format suffix, appends tool-use instruct, no double space")
print("  _transform_prompt   -> raises ValueError when prompt lacks the sentinel phrase")
print("  _extract_code       -> parses code from <tool_call> JSON")
print("  _extract_code       -> returns None for missing tag, malformed JSON, null arguments")
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("src").resolve()))

import unittest.mock as mock
sys.modules["vllm"] = mock.MagicMock()
sys.modules["vllm.lora"] = mock.MagicMock()
sys.modules["vllm.lora.request"] = mock.MagicMock()
sys.modules["decord"] = mock.MagicMock()
sys.modules["transformers"] = mock.MagicMock()

from medvision_bm.medvision_lmms_eval.lmms_eval.models.vllm_qwen25vl_tooluse import (
    VLLM_Qwen25VL_ToolUse,
)
from medvision_bm.sft.sft_prompts_tooluse import (
    COT_INSTRUCT_TL_TOOLUSE,
    COT_INSTRUCT_ANGLE_TOOLUSE,
    COT_INSTRUCT_DISTANCE_TOOLUSE,
)

def _make_obj():
    return VLLM_Qwen25VL_ToolUse.__new__(VLLM_Qwen25VL_ToolUse)

# --- _pick_instruct: TL task ---
obj = _make_obj()
doc = {"taskType": "Tumor-Lesion-Size", "biometric_profile": {}}
got = obj._pick_instruct(doc)
assert got is COT_INSTRUCT_TL_TOOLUSE, f"TL: got {got!r}"
print(f"  _pick_instruct TL        : COT_INSTRUCT_TL_TOOLUSE  PASS")

# --- _pick_instruct: angle task ---
doc = {"taskType": "Biometrics-From-Landmarks-Angle", "biometric_profile": {"metric_type": "angle"}}
got = obj._pick_instruct(doc)
assert got is COT_INSTRUCT_ANGLE_TOOLUSE, f"angle: got {got!r}"
print(f"  _pick_instruct angle     : COT_INSTRUCT_ANGLE_TOOLUSE  PASS")

# --- _pick_instruct: distance task ---
doc = {"taskType": "Biometrics-From-Landmarks-Distance", "biometric_profile": {"metric_type": "distance"}}
got = obj._pick_instruct(doc)
assert got is COT_INSTRUCT_DISTANCE_TOOLUSE, f"distance: got {got!r}"
print(f"  _pick_instruct distance  : COT_INSTRUCT_DISTANCE_TOOLUSE  PASS")

# --- _transform_prompt: strips CoT format suffix, appends tool instruct ---
SAMPLE_PROMPT = (
    "Task:\nGiven the medical image, estimate the distance.\n"
    "Format requirement:\nReport the reasoning process in <think> tags and "
    "the final answer in <answer> tags."
)
INSTRUCT = "TOOL_INSTRUCT"
result = VLLM_Qwen25VL_ToolUse._transform_prompt(SAMPLE_PROMPT, INSTRUCT)
assert "Report the reasoning process" not in result, "suffix not stripped"
assert result.endswith(INSTRUCT), "tool instruct not appended"
assert "estimate the distance" in result, "task body lost"
assert "  " not in result.split(INSTRUCT)[0], "double space found in body"
print(f"  _transform_prompt strips : suffix removed, TOOL_INSTRUCT appended  PASS")
print(f"  _transform_prompt body   : task body preserved, no double spaces  PASS")

# --- _transform_prompt: raises ValueError when sentinel is absent ---
try:
    VLLM_Qwen25VL_ToolUse._transform_prompt("No sentinel here.", INSTRUCT)
    assert False, "should have raised ValueError"
except ValueError as e:
    assert "sentinel" in str(e).lower(), f"unexpected error message: {e}"
    print(f"  _transform_prompt sentinel: ValueError raised  PASS")

# --- _extract_code: valid <tool_call> JSON ---
PHASE1 = (
    "<think> <step-1-answer> (0.5, 0.6) </step-1-answer> </think>"
    '<tool_call>{"name": "execute_python", "arguments": {"code": "import math\\nprint(42.0)"}}</tool_call>'
)
code = VLLM_Qwen25VL_ToolUse._extract_code(PHASE1)
assert code == "import math\nprint(42.0)", f"code mismatch: {code!r}"
print(f"  _extract_code valid      : {code!r}  PASS")

# --- _extract_code: no <tool_call> tag -> None ---
result = VLLM_Qwen25VL_ToolUse._extract_code("no tool call here")
assert result is None, f"expected None, got {result!r}"
print(f"  _extract_code no tag     : None  PASS")

# --- _extract_code: malformed JSON -> None ---
result = VLLM_Qwen25VL_ToolUse._extract_code("<tool_call>not json</tool_call>")
assert result is None, f"expected None, got {result!r}"
print(f"  _extract_code bad JSON   : None  PASS")

# --- _extract_code: null arguments -> None ---
result = VLLM_Qwen25VL_ToolUse._extract_code(
    '<tool_call>{"name": "execute_python", "arguments": null}</tool_call>'
)
assert result is None, f"expected None, got {result!r}"
print(f"  _extract_code null args  : None  PASS")

print("OK")
