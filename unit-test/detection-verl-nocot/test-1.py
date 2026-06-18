print("=== No-CoT detection verl formatter: tuple-unpack crash ===")
print("Objective : _format_data_DetectionTask_verl (the --without_cot_instruction +")
print("            detection path) must NOT crash. Pre-fix it does:")
print("              prompt, values_dict = _doc_to_text_DetectionTask(example)")
print("            but that formatter returns a SINGLE str -> ValueError on unpack,")
print("            and values_dict is then read to build extra_info.")
print("            Post-fix: prompt = <str>; extra_info is built from the already-")
print("            computed target = _doc_to_target_DetectionTask(example) =")
print("            [coor0_w, coor0_h, coor1_w, coor1_h].")
print("Strategy  : stub the two orthogonal deps -- the text formatter (faithful to its")
print("            real `return question` single-str contract) and the NIfTI image")
print("            processor -- and keep the REAL coordinate transform so the corner")
print("            mapping is actually verified.")

import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

try:
    from medvision_bm.rft.verl import verl_utils
except Exception as e:
    print(f"Skipping: cannot import verl_utils ({type(e).__name__}: {e})")
    sys.exit(0)

# --- Synthetic detection doc: only the fields _doc_to_target_DetectionTask reads ---
# image_size_2d = [H, W]; bounding_boxes in benchmark-planner (h, w) order.
#   min_coords=[[20,10]], max_coords=[[120,60]], H=200, W=100
#   -> target = [coor0_w, coor0_h, coor1_w, coor1_h] = [0.1, 0.4, 0.6, 0.9]
example = {
    "image_size_2d": [200, 100],
    "bounding_boxes": {"min_coords": [[20, 10]], "max_coords": [[120, 60]]},
    # fields below are only touched by the (stubbed) text formatter / image processor
    "dataset_name": "DUMMY",
    "taskID": 1,
    "label": "1",
    "image_file": "dummy.nii.gz",
}

STUB_PROMPT = "Task:\nGiven the input medical image, return the bounding box.\n"

# Faithful stub of the REAL contract (sft_utils.py:1464 -> `return question`, a single str).
verl_utils._doc_to_text_DetectionTask = lambda doc: STUB_PROMPT
# Image embedding is orthogonal to the unpack bug; avoid needing a NIfTI on disk.
verl_utils.img_proccessor_nii2png_save2dataset = lambda doc, shape=None: ["<dummy-image>"]

# Expected corners from the REAL coordinate transform (compute on a copy; the
# formatter mutates the dict it is given).
expected = verl_utils._doc_to_target_DetectionTask(copy.deepcopy(example))
exp_lowerleft = [float(expected[0]), float(expected[1])]
exp_upperright = [float(expected[2]), float(expected[3])]

bar = "-" * 78
print(f"\n{bar}")
print(f"REAL target = _doc_to_target_DetectionTask(example) = {expected}")
print(f"expected extra_info: lowerleft={exp_lowerleft}  upperright={exp_upperright}")
print(bar)

# Pre-fix this raises ValueError at verl_utils.py:520 (RED). Catch it for a clean,
# descriptive failure instead of an uncaught traceback.
try:
    result = verl_utils._format_data_DetectionTask_verl(example, None, None)
except ValueError as e:
    print(f"\nFAIL (bug present): _format_data_DetectionTask_verl raised ValueError:")
    print(f"       {e}")
    print("       (the line-520 tuple-unpack of a single-str return)")
    sys.exit(1)

print(f"\n{bar}\nASSERTIONS\n{bar}")

ei = result["extra_info"]
assert ei["lowerleft_corner_wh"] == exp_lowerleft, (
    f"FAIL: lowerleft {ei['lowerleft_corner_wh']} != expected {exp_lowerleft}"
)
assert ei["upperright_corner_wh"] == exp_upperright, (
    f"FAIL: upperright {ei['upperright_corner_wh']} != expected {exp_upperright}"
)
print(f"  extra_info corners from real target  : "
      f"lowerleft={ei['lowerleft_corner_wh']} upperright={ei['upperright_corner_wh']}  PASS")

# prompt wired from the single-str formatter return
prompt_text = result["prompt"][1]["content"][1]["text"]
assert prompt_text == STUB_PROMPT, "FAIL: user prompt text not wired from formatter return"
assert result["prompt"][1]["content"][0]["type"] == "image", "FAIL: image block missing"
print("  prompt text wired from formatter str : PASS")

# verl row metadata intact
assert result["data_source"] == "medvision-detection"
assert result["ability"] == "medvision-detection"
assert result["reward_model"]["ground_truth"] == result["ground_truth"]
assert result["ground_truth"] == "0.100, 0.400, 0.600, 0.900", (
    f"FAIL: ground_truth string {result['ground_truth']!r}"
)
print(f"  verl metadata + ground_truth string  : {result['ground_truth']!r}  PASS")

print("\nOK")
