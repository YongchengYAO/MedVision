print("=== Test 4: label masking spot-check with real Qwen2.5-VL tokenizer ===")
print("Objective : Verify mask_non_assistant_turns sets labels=-100 for system/user/tool turns")
print("            and preserves labels for assistant turns (turns 3 and 5), using the actual")
print("            Qwen2.5-VL tokenizer. Requires the model processor to be downloaded.")
print("Expected  :")
print("  trained tokens contain <tool_call> (turn 3 trained correctly)")
print("  trained tokens contain <answer>    (turn 5 trained correctly)")
print("  system content NOT in trained tokens")
print("  tool-response content NOT in trained tokens")
print("  at least some tokens are masked (labels == -100)")
print("  assistant turn HEADERS masked (completion-only: header not trained)")
print("  exactly 2 of 5 <|im_end|> kept in loss (the two assistant-turn closers)")
print("NOTE: skips gracefully if transformers or the processor is not available.")
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path("src").resolve()))

try:
    import torch
    from transformers import AutoProcessor
except ImportError:
    print("Skipping: transformers not installed.")
    sys.exit(0)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"\nLoading processor: {MODEL_ID}")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception as e:
    print(f"Skipping: processor not available ({e})")
    sys.exit(0)

tokenizer = processor.tokenizer

from medvision_bm.sft.sft_prompts_tooluse import TOOL_DEF
from medvision_bm.sft.sft_utils import mask_non_assistant_turns

# Synthetic 5-turn tool-use sample (no real dataset needed)
FAKE_CODE = (
    "import math\n"
    "x1,y1=0.1,0.2\n"
    "x2,y2=0.5,0.8\n"
    "W,H=512,512\n"
    "pw,ph=0.5,0.5\n"
    "print(round(math.sqrt(((x2-x1)*W*pw)**2+((y2-y1)*H*ph)**2),3))"
)
TOOL_CALL_JSON = json.dumps({"name": "execute_python", "arguments": {"code": FAKE_CODE}})
TOOL_RESULT = "102.4"
THINK_TEXT = (
    "<step-1-answer> Landmark A: (0.1, 0.2). </step-1-answer> "
    "<step-2-answer> Landmark B: (0.5, 0.8). </step-2-answer> "
    "<step-3-reasoning> Calling execute_python. </step-3-reasoning>"
)

messages = [
    {"role": "system",    "content": [{"type": "text", "text": json.dumps(TOOL_DEF)}]},
    {"role": "user",      "content": [{"type": "text", "text": "What is the distance between landmark A and B?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": f"<think> {THINK_TEXT} </think><tool_call>{TOOL_CALL_JSON}</tool_call>"}]},
    {"role": "tool",      "content": [{"type": "text", "text": f"<tool_response>{TOOL_RESULT}</tool_response>"}]},
    {"role": "assistant", "content": [{"type": "text", "text": f"<answer> {TOOL_RESULT} </answer>"}]},
]

text = processor.apply_chat_template(
    messages, tools=[TOOL_DEF], add_generation_prompt=False, tokenize=False,
)
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"][0]
labels = input_ids.clone()
labels = mask_non_assistant_turns(input_ids, labels, tokenizer)

trained_ids = input_ids[labels != -100]
decoded = tokenizer.decode(trained_ids, skip_special_tokens=False)
full_decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

total = len(input_ids)
trained = (labels != -100).sum().item()
print(f"\nTotal tokens: {total}  |  Trained: {trained}  |  Masked: {total - trained}")

# --- assertions ---
SYSTEM_SUBSTRING = "Execute Python code and return printed output"
assert SYSTEM_SUBSTRING in full_decoded, "SETUP ERROR: system substring not in full sequence"

assert "<tool_call>" in decoded, "FAIL: <tool_call> missing from trained tokens"
print(f"  <tool_call> in trained   : present (turn 3 trained correctly)  PASS")

assert "<answer>" in decoded, "FAIL: <answer> missing from trained tokens"
print(f"  <answer> in trained      : present (turn 5 trained correctly)  PASS")

assert (labels != -100).sum().item() < total, "FAIL: no tokens masked at all"
print(f"  some tokens masked       : {total - trained}/{total} tokens masked  PASS")

assert SYSTEM_SUBSTRING not in decoded, f"FAIL: system content leaked into trained tokens"
print(f"  system content masked    : not in trained tokens  PASS")

assert "<tool_response>" not in decoded, "FAIL: tool-response leaked into trained tokens"
print(f"  tool-response masked     : not in trained tokens  PASS")

# --- completion-only masking: lock in header-masking + exact EOS-in-loss count ---
# (these would FAIL on the old state-machine helper that kept assistant headers in loss)
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")
nl_enc = tokenizer.encode("\n", add_special_tokens=False)
newline_id = nl_enc[0] if len(nl_enc) == 1 else None

assistant_turns = 0
for i in range(total - 1):
    if input_ids[i].item() == im_start_id and input_ids[i + 1].item() == assistant_id:
        assistant_turns += 1
        assert labels[i].item() == -100, "FAIL: assistant <|im_start|> not masked"
        assert labels[i + 1].item() == -100, "FAIL: assistant role token not masked"
        if (
            newline_id is not None
            and i + 2 < total
            and input_ids[i + 2].item() == newline_id
        ):
            assert labels[i + 2].item() == -100, "FAIL: assistant header newline not masked"
assert assistant_turns == 2, f"SETUP ERROR: expected 2 assistant turns, got {assistant_turns}"
assert "<|im_start|>" not in decoded, "FAIL: assistant header leaked into trained tokens"
print(f"  assistant headers masked : {assistant_turns} headers masked, none in trained  PASS")

imend_total = int((input_ids == im_end_id).sum().item())
imend_in_loss = int(((input_ids == im_end_id) & (labels != -100)).sum().item())
assert imend_in_loss == 2, f"FAIL: expected 2 <|im_end|> in loss, got {imend_in_loss}"
print(f"  closing <|im_end| in loss: {imend_in_loss}/{imend_total} (the 2 assistant closers)  PASS")

print("OK")
