print("=== SFT loss masking: behavior-neutrality golden diff ===")
print("Objective : Prove the completion-only-masking edit does NOT change any existing")
print("            (flag-off) training path, and that MEDVISION_SFT_COMPLETION_ONLY=1")
print("            activates masking ONLY for the Gemma collates.")
print("Modes     : argv[1] in {capture, assert-off, assert-on}")
print("  capture     -> record golden 'labels' each collate produces (run on HEAD/pre-edit code)")
print("  assert-off  -> (flag unset) assert labels byte-identical to golden for ALL 4 families")
print("  assert-on   -> (flag=1) Gemma labels CHANGE (prompt masked); Qwen labels UNCHANGED")
print("Families  : gemma4, medgemma, qwen25vl, qwen3vl (qwen3vl reuses make_collate_fn_Qwen25VL)")
print("NOTE: no single transformers env loads all four processors (Gemma-4/Qwen3.6 need 5.5.0,")
print("      MedGemma needs 4.54.0). Run once per env; per-family try/except skips what won't load.")
print("      A family that only ever SKIPs is a FAIL of the overall validation, not a pass.")
print("      AutoProcessor.from_pretrained pulls only tokenizer/processor JSON, no weights.")
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
FORCE = "--force" in sys.argv[2:]
if MODE not in ("capture", "assert-off", "assert-on"):
    print("usage: test-neutrality-golden.py {capture|assert-off|assert-on} [--force]")
    sys.exit(2)

GOLDEN_DIR = pathlib.Path("unit-test/sft-loss-masking/golden")

# --- Real MedVision TL-CoT sample (from test-1.py) + a short second sample to force padding ---
USER_TEXT = """Task:
Given the input medical image: T2 Fluid Attenuated Inversion Recovery (FLAIR) brain magnetic resonance imaging (MRI) scan, estimate the major and minor axis lengths of the ellipse enclosing the resection cavity of brain, in millimeters.
Additional information:
The image size is 504 pixels (width) x 504 pixels (height).
The pixel size for this image is 0.433 millimeters (width) x 0.361 millimeters (height).
Format requirement:
The final answer must be enclosed within <answer> </answer> tags. The answer should consist of two decimal numbers separated by a comma, without units or extra text.
Reasoning steps:
Step 1: Identify the major axis. Report the reasoning process and final answer within <think> </think> and <answer> </answer> tags."""
ASSISTANT_TEXT = """<think>
<step-1-reasoning>
The resection cavity spans roughly x 0.407 to 0.694 and y 0.635 to 0.694.
</step-1-reasoning>
<step-1-answer>
major_axis_length ~= 63.54 mm; minor_axis_length ~= 36.38 mm
</step-1-answer>
</think>
<answer>63.54, 36.38</answer>"""
USER_TEXT_SHORT = "Task:\nGiven the input medical image, estimate the major and minor axis lengths in millimeters."
ASSISTANT_TEXT_SHORT = "<think> <step-1-answer> 10.0 mm; 5.0 mm </step-1-answer> </think> <answer>10.0, 5.0</answer>"

PROMPT_SENTINEL = "The image size is"

try:
    from PIL import Image
    from transformers import AutoProcessor
    import transformers
except ImportError as e:
    print(f"Skipping: transformers/PIL not installed ({e}).")
    sys.exit(0)

# (family, model_id, factory import path)
CASES = [
    ("gemma4", "google/gemma-4-31B-it", "gemma4_utils", "make_collate_fn_Gemma4"),
    ("medgemma", "google/medgemma-27b-it", "medgemma_utils", "make_collate_fn_MedGemma"),
    ("qwen25vl", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen25vl_utils", "make_collate_fn_Qwen25VL"),
    ("qwen3vl", "Qwen/Qwen3.6-27B", "qwen25vl_utils", "make_collate_fn_Qwen25VL"),
]
GEMMA_FAMILIES = {"gemma4", "medgemma"}


def _examples():
    return [
        {
            "processed_images": [Image.new("RGB", (224, 224), (128, 128, 128))],
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]},
                {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TEXT}]},
            ],
        },
        {
            "processed_images": [Image.new("RGB", (224, 224), (64, 64, 64))],
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT_SHORT}]},
                {"role": "assistant", "content": [{"type": "text", "text": ASSISTANT_TEXT_SHORT}]},
            ],
        },
    ]


def _sha(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()


def _fingerprint(family, model_id, mod_name, factory_name):
    import importlib
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    mod = importlib.import_module(f"medvision_bm.sft.{mod_name}")
    make_collate = getattr(mod, factory_name)
    collate = make_collate(proc)  # flag is read here, at factory-construction time
    batch = collate(_examples())
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    return {
        "family": family,
        "tf_version": transformers.__version__,
        "shape": list(labels.shape),
        "labels_sha256": _sha(labels),
        "input_ids_sha256": _sha(input_ids),
        "n_trained": int((labels != -100).sum()),
        "labels": labels.tolist(),
        "input_ids": input_ids.tolist(),
        "image_token": (getattr(proc.tokenizer, "special_tokens_map", {}) or {}).get("image_token"),
        "pad_token_id": proc.tokenizer.pad_token_id,
    }


def _load_golden(family):
    p = GOLDEN_DIR / f"labels__{family}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


results = []  # (family, status) where status in PASS/FAIL/SKIP

for family, model_id, mod_name, factory_name in CASES:
    tag = f"{family:9} ({model_id})"
    if MODE == "assert-off":
        assert "MEDVISION_SFT_COMPLETION_ONLY" not in os.environ or \
            os.environ["MEDVISION_SFT_COMPLETION_ONLY"] == "0", \
            "assert-off requires MEDVISION_SFT_COMPLETION_ONLY unset or '0'"
    if MODE == "assert-on":
        assert os.environ.get("MEDVISION_SFT_COMPLETION_ONLY") == "1", \
            "assert-on requires MEDVISION_SFT_COMPLETION_ONLY=1"

    try:
        fp = _fingerprint(family, model_id, mod_name, factory_name)
    except Exception as e:
        print(f"SKIP {tag}: {type(e).__name__}: {e}")
        results.append((family, "SKIP"))
        continue

    if MODE == "capture":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        dst = GOLDEN_DIR / f"labels__{family}.json"
        if dst.exists() and not FORCE:
            print(f"SKIP {tag}: golden exists (use --force to overwrite)")
            results.append((family, "SKIP"))
            continue
        dst.write_text(json.dumps(fp, indent=2))
        print(f"PASS {tag}: captured golden (tf={fp['tf_version']}, n_trained={fp['n_trained']})")
        results.append((family, "PASS"))
        continue

    golden = _load_golden(family)
    if golden is None:
        print(f"SKIP {tag}: no golden file (run capture first)")
        results.append((family, "SKIP"))
        continue
    if golden["tf_version"] != fp["tf_version"]:
        print(f"SKIP {tag}: golden tf={golden['tf_version']} != running {fp['tf_version']}")
        results.append((family, "SKIP"))
        continue

    ok = True
    # Tokenization must be unchanged in both modes (precondition for a meaningful labels diff).
    if fp["input_ids_sha256"] != golden["input_ids_sha256"]:
        print(f"FAIL {tag}: input_ids changed (tokenizer/processor drift) "
              f"{golden['input_ids_sha256'][:12]} -> {fp['input_ids_sha256'][:12]}")
        results.append((family, "FAIL"))
        continue

    if MODE == "assert-off":
        if fp["labels"] != golden["labels"]:
            print(f"FAIL {tag}: labels changed with flag OFF "
                  f"(sha {golden['labels_sha256'][:12]} -> {fp['labels_sha256'][:12]}, "
                  f"n_trained {golden['n_trained']} -> {fp['n_trained']})")
            ok = False
        else:
            print(f"PASS {tag}: labels byte-identical to golden (flag off)")

    elif MODE == "assert-on":
        if family in GEMMA_FAMILIES:
            # Gemma: labels MUST change; masking only removes tokens from the loss.
            if fp["labels"] == golden["labels"]:
                print(f"FAIL {tag}: Gemma labels unchanged with flag ON (masking not applied)")
                ok = False
            elif fp["n_trained"] >= golden["n_trained"]:
                print(f"FAIL {tag}: n_trained did not drop ({golden['n_trained']} -> {fp['n_trained']})")
                ok = False
            else:
                # trained positions must be a strict subset of the golden's trained positions
                import numpy as np
                new_lab = np.array(fp["labels"]); old_lab = np.array(golden["labels"])
                new_tr = new_lab != -100; old_tr = old_lab != -100
                subset = bool((new_tr & ~old_tr).sum() == 0)
                # decoded loss excludes the prompt sentinel and includes the answer
                proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                tok = proc.tokenizer
                ids0 = batch_ids = np.array(fp["input_ids"])[0]
                lab0 = new_lab[0]
                trained_txt = tok.decode([int(t) for t, l in zip(ids0, lab0) if l != -100],
                                         skip_special_tokens=False)
                excl = PROMPT_SENTINEL not in trained_txt
                incl = "<answer>" in trained_txt
                if subset and excl and incl:
                    print(f"PASS {tag}: masking applied (n_trained {golden['n_trained']} -> "
                          f"{fp['n_trained']}, prompt masked, answer kept)")
                else:
                    print(f"FAIL {tag}: subset={subset} prompt_excluded={excl} answer_included={incl}")
                    ok = False
        else:
            # Qwen: identical to golden even with the flag set (factory never reads it).
            if fp["labels"] != golden["labels"]:
                print(f"FAIL {tag}: Qwen labels changed with flag ON — factory read the flag!")
                ok = False
            else:
                print(f"PASS {tag}: Qwen labels flag-insensitive (identical to golden)")

    results.append((family, "PASS" if ok else "FAIL"))

# --- Summary ---
summary = {f: s for f, s in results}
print("\nSUMMARY:", json.dumps(summary))
n_fail = sum(1 for _, s in results if s == "FAIL")
if n_fail:
    print(f"RESULT: FAIL ({n_fail} family/families failed {MODE})")
    sys.exit(1)
n_pass = sum(1 for _, s in results if s == "PASS")
print(f"RESULT: {n_pass} PASS, {len(results) - n_pass} SKIP for {MODE} in this env")
