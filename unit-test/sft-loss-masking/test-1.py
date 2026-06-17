print("=== SFT loss masking: make_collate_fn_Qwen25VL prompt-masking (H5) ===")
print("Objective : On a real MedVision TL-CoT sample, verify the Qwen2.5-VL collate")
print("            masks everything up to and including the assistant header and the")
print("            whole user/system turn, and computes loss only on the assistant")
print("            response content + its closing <|im_end|>.")
print("Sample    : copied verbatim from record 0 of")
print("            Results/MedVision-TL-CoT/Claude-Fable-5/")
print("            20260612_013600_samples_BraTS24_TumorLesionSize_Task04_Axial-CoT.jsonl")
print("            (target = [74.0, 36.0]); inlined below so the test reads no files.")
print("NOTE: a placeholder image is used (image content is irrelevant to label")
print("      masking — image-pad tokens are masked regardless). Skips gracefully if")
print("      transformers/PIL/the processor is unavailable.")
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

# --- Real sample, copied verbatim (no file I/O) ---
USER_TEXT = """Task:
Given the input medical image: T2 Fluid Attenuated Inversion Recovery (FLAIR) brain magnetic resonance imaging (MRI) scan, estimate the major and minor axis lengths of the ellipse enclosing the resection cavity of brain, in millimeters.
Additional information:
The image size is 504 pixels (width) x 504 pixels (height).
The pixel size for this image is 0.433 millimeters (width) x 0.361 millimeters (height).
Format requirement:
The final answer must be enclosed within <answer> </answer> tags. The answer should consist of two decimal numbers separated by a comma, without units or extra text. The first number is the major axis length, and the second is the minor axis length.
Reasoning steps:
Step 1: Identify the major axis (the longest diameter) of the ellipse enclosing the target region. Find its two endpoints and record their relative coordinates in the format (x, y) = (relative position in width direction, relative position in height direction). Denote the endpoints as (x1_major, y1_major) and (x2_major, y2_major). Step 2: Identify the minor axis (the shortest diameter) of the ellipse. Find its two endpoints and record their relative coordinates in the same (x, y) format. Denote them as (x1_minor, y1_minor) and (x2_minor, y2_minor). Step 3: Given the pixel dimensions (pixel_width, pixel_height) and image size (image_width, image_height), compute the physical length of the major axis using: major_axis_length = sqrt(((x2_major - x1_major) * image_width * pixel_width)^2 + ((y2_major - y1_major) * image_height * pixel_height)^2). Step 4: Similarly, compute the physical length of the minor axis using: minor_axis_length = sqrt(((x2_minor - x1_minor) * image_width * pixel_width)^2 + ((y2_minor - y1_minor) * image_height * pixel_height)^2). Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags, respectively. Inside <think> </think>, include reasoning and step results using <step-k-reasoning> </step-k-reasoning> and <step-k-answer> </step-k-answer> tags.
Follow the reasoning steps to get the final answer in the required format."""

ASSISTANT_TEXT = """<think>
<step-1-reasoning>
The resection cavity appears as the large dark (hypointense) region in the inferior-posterior portion of the brain, spanning roughly from x≈205 to x≈350 pixels and y≈290 to y≈390 pixels. The major axis runs roughly horizontally across this dark cavity.
</step-1-reasoning>
<step-1-answer>
(x1_major, y1_major) = (0.407, 0.635), (x2_major, y2_major) = (0.694, 0.694)
</step-1-answer>
<step-2-reasoning>
The minor axis is perpendicular to the major axis, running roughly vertically through the cavity from its superior boundary (~y=290 px) to its inferior boundary (~y=390 px).
</step-2-reasoning>
<step-2-answer>
(x1_minor, y1_minor) = (0.540, 0.575), (x2_minor, y2_minor) = (0.556, 0.774)
</step-2-answer>
<step-3-reasoning>
Major axis: Δx = (0.694 − 0.407) × 504 × 0.433 = 0.287 × 504 × 0.433 ≈ 62.63 mm; Δy = (0.694 − 0.635) × 504 × 0.361 = 0.059 × 504 × 0.361 ≈ 10.73 mm. Length = sqrt(62.63² + 10.73²) ≈ sqrt(3922 + 115) ≈ 63.54 mm.
</step-3-reasoning>
<step-3-answer>
major_axis_length ≈ 63.54 mm
</step-3-answer>
<step-4-reasoning>
Minor axis: Δx = (0.556 − 0.540) × 504 × 0.433 ≈ 3.49 mm; Δy = (0.774 − 0.575) × 504 × 0.361 ≈ 36.21 mm. Length = sqrt(3.49² + 36.21²) ≈ sqrt(12.2 + 1311) ≈ 36.38 mm.
</step-4-reasoning>
<step-4-answer>
minor_axis_length ≈ 36.38 mm
</step-4-answer>
</think>
<answer>63.54, 36.38</answer>"""

try:
    from PIL import Image
    from transformers import AutoProcessor
except ImportError:
    print("Skipping: transformers/PIL not installed.")
    sys.exit(0)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"\nLoading processor: {MODEL_ID}")
try:
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception as e:
    print(f"Skipping: processor not available ({e})")
    sys.exit(0)

from medvision_bm.sft.qwen25vl_utils import make_collate_fn_Qwen25VL

example = {
    "processed_images": [Image.new("RGB", (224, 224), (128, 128, 128))],
    "messages": [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TEXT}]},
    ],
}

collate = make_collate_fn_Qwen25VL(proc)
batch = collate([example])

input_ids = batch["input_ids"][0]
labels = batch["labels"][0]
tok = proc.tokenizer
IMG_PAD = "<|image_pad|>"


def _collapse_image_pads(s):
    import re

    return re.sub(
        r"(?:" + re.escape(IMG_PAD) + r"){2,}",
        lambda m: f"{IMG_PAD}...x{m.group(0).count(IMG_PAD)}...{IMG_PAD}",
        s,
    )


trained_ids = input_ids[labels != -100]
masked_ids = input_ids[labels == -100]
decoded_trained = tok.decode(trained_ids, skip_special_tokens=False)
decoded_masked = tok.decode(masked_ids, skip_special_tokens=False)
full_decoded = tok.decode(input_ids, skip_special_tokens=False)

total = len(input_ids)
n_trained = int((labels != -100).sum().item())
n_masked = total - n_trained
n_img_pad = int((input_ids == tok.convert_tokens_to_ids(proc.image_token)).sum().item())

bar = "─" * 90
print(f"\n{bar}")
print("FULL PROMPT  (decoded input_ids, special tokens shown; image pads collapsed)")
print(bar)
print(_collapse_image_pads(full_decoded))

print(f"\n{bar}")
print("MASKED OUT  (labels == -100  →  NOT in loss; image pads collapsed)")
print(bar)
print(_collapse_image_pads(decoded_masked))

print(f"\n{bar}")
print("LOSS TOKENS  (labels != -100  →  the only tokens used for loss)")
print(bar)
print(decoded_trained)

# Per-token MASK/LOSS view around the assistant-header boundary (the key region).
im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
assistant_id = tok.convert_tokens_to_ids("assistant")
hdr = next(
    (
        k
        for k in range(total - 1)
        if input_ids[k].item() == im_start_id
        and input_ids[k + 1].item() == assistant_id
    ),
    None,
)
print(f"\n{bar}")
print("PER-TOKEN MASK at the assistant-header boundary  (idx | id | piece | MASK/LOSS)")
print(bar)
if hdr is not None:
    lo = max(0, hdr - 2)
    hi = min(total, hdr + 12)
    for k in range(lo, hi):
        tag = "LOSS" if labels[k].item() != -100 else "MASK"
        print(
            f"  {k:>4} | {input_ids[k].item():>6} | {tok.decode([input_ids[k].item()])!r:<14} | {tag}"
        )
    print("  ... (response content continues as LOSS) ...")

print(f"\n{bar}")
print("TOKEN BUDGET")
print(bar)
print(f"  total tokens         : {total}")
print(f"  loss tokens          : {n_trained}")
print(f"  masked tokens        : {n_masked}")
print(f"    of which image pad : {n_img_pad}")
print(f"  loss fraction        : {n_trained / total:.1%}")

print(f"\n{bar}")
print("ASSERTIONS")
print(bar)

# --- strategy assertions ---
assert "<answer>" in decoded_trained, "FAIL: <answer> missing from loss tokens"
assert "63.54, 36.38" in decoded_trained, "FAIL: answer value missing from loss tokens"
print("  response content trained : <answer>63.54, 36.38</answer> in LOSS  PASS")

assert decoded_trained.rstrip().endswith(
    "<|im_end|>"
), "FAIL: closing <|im_end|> not in loss tokens"
print("  closing <|im_end| trained: response terminator in LOSS  PASS")

assert (
    "<|im_start|>" not in decoded_trained
), "FAIL: assistant header leaked into loss tokens"
print("  assistant header masked  : <|im_start|>assistant not in LOSS  PASS")

assert (
    "The pixel size for this image" not in decoded_trained
), "FAIL: user prompt leaked into loss tokens"
assert (
    "The pixel size for this image" in decoded_masked
), "FAIL: pixel-size text should be in the masked set"
print("  user prompt masked       : pixel-size text in MASKED, not in LOSS  PASS")

assert n_trained < total, "FAIL: no tokens masked at all"
print(f"  some tokens masked       : {n_masked}/{total} tokens masked  PASS")

print("\nOK")
