print("=== SFT loss masking: Gemma-family completion-only masking (flag ON) ===")
print("Objective : With MEDVISION_SFT_COMPLETION_ONLY=1, verify make_collate_fn_MedGemma and")
print("            make_collate_fn_Gemma4 compute loss ONLY on the model turn's response")
print("            content + its closing end-of-turn marker.")
print("Expected  :")
print("  <think> and <answer> + the answer value in LOSS")
print("  closing end-of-turn marker in LOSS; exactly 1 end-of-turn token in LOSS")
print("  model-turn header (<START>, 'model', role newline) masked")
print("  user prompt boilerplate masked; pad + image tokens masked")
print("  the ChatML masker (mask_non_assistant_turns) zeroes the ENTIRE loss on a Gemma")
print("  tokenizer -- this is why a separate mask_non_assistant_turns_gemma exists")
print("NOTE: MedGemma (Gemma 3) is pinned to transformers 4.54.0 and Gemma 4 to 5.5.0, so one")
print("      process usually loads only one; each case is attempted independently and skipped on")
print("      any load failure (wrong tf version, gated repo, no HF_TOKEN). Exits 0 if neither")
print("      loads. AutoProcessor.from_pretrained downloads only config/tokenizer/processor JSON,")
print("      never the 27B/31B weights. A placeholder image is used (irrelevant to label masking).")
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

# The flag is read at collate-factory construction time, so it MUST be set before the factory
# is called. Set it up-front for the whole test.
os.environ["MEDVISION_SFT_COMPLETION_ONLY"] = "1"

# --- Real MedVision TL-CoT sample, copied verbatim from test-1.py (no file I/O) ---
USER_TEXT = """Task:
Given the input medical image: T2 Fluid Attenuated Inversion Recovery (FLAIR) brain magnetic resonance imaging (MRI) scan, estimate the major and minor axis lengths of the ellipse enclosing the resection cavity of brain, in millimeters.
Additional information:
The image size is 504 pixels (width) x 504 pixels (height).
The pixel size for this image is 0.433 millimeters (width) x 0.361 millimeters (height).
Format requirement:
The final answer must be enclosed within <answer> </answer> tags. The answer should consist of two decimal numbers separated by a comma, without units or extra text. The first number is the major axis length, and the second is the minor axis length.
Reasoning steps:
Step 1: Identify the major axis (the longest diameter) of the ellipse enclosing the target region. Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively."""

ASSISTANT_TEXT = """<think>
<step-1-reasoning>
The resection cavity spans roughly x from 0.407 to 0.694 and y from 0.635 to 0.694.
</step-1-reasoning>
<step-1-answer>
major_axis_length ~= 63.54 mm; minor_axis_length ~= 36.38 mm
</step-1-answer>
</think>
<answer>63.54, 36.38</answer>"""

PROMPT_SENTINEL = "The pixel size for this image"

try:
    from PIL import Image
    from transformers import AutoProcessor
except ImportError:
    print("Skipping: transformers/PIL not installed.")
    sys.exit(0)

from medvision_bm.sft.gemma4_utils import make_collate_fn_Gemma4
from medvision_bm.sft.medgemma_utils import make_collate_fn_MedGemma
from medvision_bm.sft.sft_utils import (
    _resolve_special_token_id,
    mask_non_assistant_turns,
)

# (model_id, collate factory, start marker, end marker, the OTHER family's start marker)
CASES = [
    ("google/medgemma-27b-it", make_collate_fn_MedGemma, "<start_of_turn>", "<end_of_turn>", "<|turn>"),
    ("google/gemma-4-31B-it", make_collate_fn_Gemma4, "<|turn>", "<turn|>", "<start_of_turn>"),
]


def _example():
    return {
        "processed_images": [Image.new("RGB", (224, 224), (128, 128, 128))],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]},
            {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TEXT}]},
        ],
    }


bar = "-" * 90
checked = 0
for MODEL_ID, make_collate, START, END, OTHER_START in CASES:
    print(f"\n{bar}\nCASE: {MODEL_ID}\n{bar}")
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Skipping {MODEL_ID}: processor not available ({type(e).__name__}: {e})")
        continue

    tok = proc.tokenizer
    start_id = _resolve_special_token_id(tok, START)
    end_id = _resolve_special_token_id(tok, END)
    role_id = _resolve_special_token_id(tok, "model")
    assert start_id is not None and end_id is not None, f"FAIL: {START}/{END} did not resolve"
    assert role_id is not None, "FAIL: role token 'model' did not resolve"

    # The probe must be unambiguous: the other Gemma generation's markers and ChatML's are absent.
    assert _resolve_special_token_id(tok, OTHER_START) is None, \
        f"FAIL: {OTHER_START} unexpectedly resolves on {MODEL_ID}"
    assert _resolve_special_token_id(tok, "<|im_start|>") is None, \
        "FAIL: ChatML marker unexpectedly resolves on a Gemma tokenizer"
    print(f"  markers resolve: start={start_id} end={end_id} role={role_id}; "
          f"{OTHER_START} + <|im_start|> absent  PASS")

    collate = make_collate(proc)
    batch = collate([_example()])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    trained = tok.decode(input_ids[labels != -100], skip_special_tokens=False)
    masked = tok.decode(input_ids[labels == -100], skip_special_tokens=False)
    total = int(input_ids.shape[0])
    n_trained = int((labels != -100).sum())

    # 1. response content trained (incl. CoT scaffolding -> guards Gemma-4 strip_thinking())
    assert "<answer>" in trained and "63.54, 36.38" in trained, "FAIL: answer missing from LOSS"
    assert "<think>" in trained, "FAIL: <think> stripped from LOSS"
    print("  response content trained : <think> ... <answer>63.54, 36.38</answer> in LOSS  PASS")

    # 2. closing end-of-turn in loss, and ONLY the model turn's (exactly 1)
    assert trained.rstrip().endswith(END), f"FAIL: closing {END} not last in LOSS"
    n_end_in_loss = int(((input_ids == end_id) & (labels != -100)).sum())
    assert n_end_in_loss == 1, f"FAIL: expected exactly 1 end-of-turn in LOSS, got {n_end_in_loss}"
    print(f"  closing marker trained   : {END} terminates LOSS; exactly 1 end-of-turn kept  PASS")

    # 3. model header (start marker, role, role newline) masked
    hdrs = [k for k in range(total - 1)
            if input_ids[k].item() == start_id and input_ids[k + 1].item() == role_id]
    assert len(hdrs) == 1, f"SETUP ERROR: expected 1 model header, got {len(hdrs)}"
    k = hdrs[0]
    assert labels[k] == -100 and labels[k + 1] == -100, "FAIL: model header not masked"
    nl_enc = tok.encode("\n", add_special_tokens=False)
    if len(nl_enc) == 1 and k + 2 < total and input_ids[k + 2].item() == nl_enc[0]:
        assert labels[k + 2] == -100, "FAIL: role-header newline not masked"
    print("  model header masked      : <START> 'model' + role newline not in LOSS  PASS")

    # 4. the bug this fixes: user prompt boilerplate out of the loss
    assert PROMPT_SENTINEL not in trained, "FAIL: user prompt leaked into LOSS"
    assert PROMPT_SENTINEL in masked, "FAIL: prompt sentinel not in MASKED"
    print("  user prompt masked       : pixel-size text in MASKED, not in LOSS  PASS")

    # 5. pad + image tokens fully masked
    assert int(((input_ids == tok.pad_token_id) & (labels != -100)).sum()) == 0, "FAIL: pad in LOSS"
    img_tok = (getattr(tok, "special_tokens_map", {}) or {}).get("image_token")
    if img_tok is not None:
        img_id = tok.convert_tokens_to_ids(img_tok)
        assert int(((input_ids == img_id) & (labels != -100)).sum()) == 0, "FAIL: image token in LOSS"
    print("  pad + image masked       : zero pad/image tokens in LOSS  PASS")

    # 6. non-degenerate
    assert 0 < n_trained < total, f"FAIL: degenerate mask ({n_trained}/{total})"
    print(f"  non-degenerate           : {n_trained}/{total} tokens in LOSS  PASS")

    # 7. the trap: the ChatML masker zeroes the whole loss on a Gemma tokenizer.
    #    mask_non_assistant_turns mutates its labels arg -> pass a clone.
    chatml_labels = mask_non_assistant_turns(input_ids, input_ids.clone(), tok)
    assert int((chatml_labels != -100).sum()) == 0, \
        "FAIL: expected the ChatML masker to zero the whole loss on a Gemma tokenizer"
    print("  ChatML masker is a trap  : zeroes the ENTIRE loss on a Gemma tokenizer  PASS")

    print(f"\n  TOKEN BUDGET: total={total}  loss={n_trained}  masked={total - n_trained}  "
          f"loss_frac={n_trained / total:.1%}")
    checked += 1

if checked == 0:
    print("\nSkipping: neither Gemma processor loaded under the installed transformers.")
    sys.exit(0)

print("\nOK")
