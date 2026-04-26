# Prompt inspection: scaledPS doc_to_text changes exactly the pixel_size line.
print("Test 5: Prompt inspection — doc_to_text_TumorLesionSize_CoT_scaledPS must differ from the base CoT prompt in exactly one line, and that line must mention 'pixel'.")
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset
from lmms_eval.tasks.medvision.medvision_utils import _get_pixel_size_scale_factor
from lmms_eval.tasks.BraTS24.utils import (
    doc_to_text_TumorLesionSize_CoT,
    doc_to_text_TumorLesionSize_CoT_scaledPS,
)

ds = load_dataset(
    "YongchengYAO/MedVision",
    "BraTS24_TumorLesionSize_Task04_Axial_Test",
    split="test",
    trust_remote_code=True,
)

kwargs = {"lmms_eval_specific_kwargs": {"model_name": "vllm_qwen25vl", "model_hf": "Qwen/Qwen2.5-VL-7B-Instruct"}}
doc = ds[0]
S = _get_pixel_size_scale_factor(doc, "uniform")

base_prompt   = doc_to_text_TumorLesionSize_CoT(doc, **kwargs)
scaled_prompt = doc_to_text_TumorLesionSize_CoT_scaledPS(doc, **kwargs)

print("=== BASE PROMPT ===")
print(base_prompt)
print("\n=== SCALED PROMPT ===")
print(scaled_prompt)
print(f"\n--- S = {S:.4f} ---\n")

base_lines   = base_prompt.splitlines()
scaled_lines = scaled_prompt.splitlines()
diffs = [(i + 1, b, s) for i, (b, s) in enumerate(zip(base_lines, scaled_lines)) if b != s]

print("Differing lines:")
for lineno, b, s in diffs:
    print(f"  line {lineno}:")
    print(f"    base:   {b}")
    print(f"    scaled: {s}")

assert len(diffs) == 1, f"Expected exactly 1 differing line, got {len(diffs)}: {diffs}"
assert "pixel" in diffs[0][2].lower(), "The differing line does not mention 'pixel'"
print("\nOK: exactly one line differs (pixel_size)")
