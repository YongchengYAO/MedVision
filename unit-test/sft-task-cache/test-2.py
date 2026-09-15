print("=== SFT entry points: launcher-shaped regression after the task-cache change ===")
print("Objective : Drive main() through the REAL argument parser, the same way")
print("            `if __name__ == \"__main__\"` does, in the shapes the shipped")
print("            launchers use, and check nothing the launchers depend on moved:")
print("            (A) every one of the 10 entry points still prints the")
print("                `Prepared dataset saved at '<dir>'` line the 34 launchers sed for,")
print("                writes the expected schema, and on a re-run loads/formats NOTHING,")
print("            (B) on a reference entry point: the true-size dir name, the")
print("                training-side --prepared_ds_dir + --skip_process_dataset path,")
print("                and the _D0 token for an omitted task (tooluse launchers),")
print("            (C) CoT and non-CoT formatters get DIFFERENT cache keys.")
print("            Each entry point's cache must sit BESIDE its prepared dataset.")
print("Data      : stubbed loader + stubbed formatters; no MedVision data or GPU.")

import contextlib
import importlib.util
import io
import json
import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path("src").resolve()))

import datasets
from datasets import Dataset, DatasetDict

datasets.disable_caching()

SFT = pathlib.Path("src/medvision_bm/sft")
REFERENCE = "train__fullFT-CoT__qwen2_5_vl.py"
ENTRY_POINTS = sorted(p.name for p in SFT.glob("train__*.py"))

N = {"tasks_AD": 40, "tasks_detect": 90, "tasks_TL": 60}
VAL = 10

LOADS = []
FORMATS = []


def _rows(n, tag):
    return Dataset.from_dict({
        "image_file": [f"/vol/{tag}/{i % 5}.nii.gz" for i in range(n)],
        "slice_dim": [i % 3 for i in range(n)],
        "slice_idx": list(range(n)),
        "dataset_name": [tag] * n,
    })


def stub_loader(tasks_list_json_path, limit_train_sample, limit_val_sample, **kw):
    tag = pathlib.Path(tasks_list_json_path).stem
    LOADS.append(tag)
    return DatasetDict({"train": _rows(N[tag], tag), "validation": _rows(VAL, tag)})


def make_formatter(name):
    def fmt(example, model_name=None, model_hf=None, process_img=False,
            save_processed_img_to_disk=False, new_shape_hw=None):
        FORMATS.append(name)
        example["messages"] = f"{name}:{example['slice_idx']}"
        example["labels"] = "L"
        return example
    fmt.__name__ = name
    return fmt


def load_entry(filename):
    spec = importlib.util.spec_from_file_location("ep_" + filename.replace(".", "_"), SFT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.setup_env_hf_medvision_ds = lambda **kw: None
    mod.load_split_limit_dataset = stub_loader
    formatters = [n for n in dir(mod) if n.startswith("_format_data_")]
    for name in formatters:
        setattr(mod, name, make_formatter(name))
    return mod, formatters


def run(mod, argv):
    """Mirror the entry point's __main__ block exactly."""
    import medvision_bm.sft.sft_utils as sft_utils
    # The model-name validator imports the vendored lmms_eval, installed only in
    # the SFT training env and unrelated to dataset preparation.
    sft_utils.check_model_supported = lambda *a, **k: None
    sys.argv = ["train.py"] + argv
    args = sft_utils.parse_validate_args_multiTask()
    # The fullFT entry points remap lora_checkpoint_dir -> checkpoint_dir in their
    # __main__ block; the LoRA ones take lora_checkpoint_dir directly. Follow
    # whichever this module's main() actually declares.
    import inspect
    if "checkpoint_dir" in inspect.signature(mod.main).parameters:
        args["checkpoint_dir"] = args.pop("lora_checkpoint_dir")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main(**args)
    return buf.getvalue()


def handoff(out):
    found = re.findall(r"Prepared dataset saved at '([^']*)'", out)
    return found[-1] if found else None


failures = []


def check(name, cond, detail=""):
    print(f"  {name:<52} {'PASS' if cond else 'FAIL'}{(' ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


# Guarded: datasets' save_to_disk builds a worker pool, which needs a __main__
# guard under the spawn start method (production runs main() the same way).
if __name__ == "__main__":
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="medvision-ep-"))
    jsons = {}
    for tag in N:
        p = tmp / f"{tag}.json"
        p.write_text(json.dumps({f"{tag}_cfg": {}}))
        jsons[tag] = str(p)

    def base_argv(data_dir, ckpt):
        return [
            "--run_name", "regress",
            "--model_family_name", "qwen25vl",
            "--base_model_hf", "Qwen/Qwen2.5-VL-7B-Instruct",
            "--data_dir", str(data_dir),
            "--lora_checkpoint_dir", str(ckpt),
            "--process_dataset_only", "true",
            "--skip_process_dataset", "false",
            "--save_processed_img_to_disk", "false",
            "--num_workers_format_dataset", "1",  # in-process, so formatting is countable
            "--num_workers_concat_datasets", "1",
            "--val_sample_limit_task_AD", str(VAL),
            "--val_sample_limit_task_Detection", str(VAL),
            "--val_sample_limit_task_TL", str(VAL),
            "--new_shape_hw", "512", "512",
        ]

    ALL_TASKS = [
        "--tasks_list_json_path_AD", jsons["tasks_AD"],
        "--tasks_list_json_path_detect", jsons["tasks_detect"],
        "--tasks_list_json_path_TL", jsons["tasks_TL"],
    ]

    # --- (A) all 10 entry points: fresh prep, then resume -------------------
    print(f"\n[A] all {len(ENTRY_POINTS)} entry points: fresh prep, then resume")
    for filename in ENTRY_POINTS:
        data_dir = tmp / "data" / filename
        data_dir.mkdir(parents=True)
        argv = base_argv(data_dir, tmp / "ckpt") + ALL_TASKS

        LOADS.clear(); FORMATS.clear()
        mod, formatters = load_entry(filename)
        has_detection = any("Detection" in f for f in formatters)
        expect_tasks = {"tasks_AD", "tasks_TL"} | ({"tasks_detect"} if has_detection else set())
        out = run(mod, argv)
        d = handoff(out)
        ok_fresh = (
            d is not None
            and set(LOADS) == expect_tasks
            and len(FORMATS) == sum(N[t] for t in expect_tasks) + VAL * len(expect_tasks)
        )
        prepared = datasets.load_from_disk(d) if d else None
        ok_schema = prepared is not None and set(prepared["train"].column_names) == {
            "messages", "labels", "image_file", "slice_dim", "slice_idx", "__task_name"
        }
        rows_before = sorted(prepared["train"]["messages"]) if prepared else None

        # The task cache must live beside the prepared dataset it feeds, so the
        # non-CoT entry point caches under SFT_datasets/ and not SFT-CoT_datasets/.
        cache_root = pathlib.Path(d).parent / "_task_cache"
        ok_location = cache_root.is_dir() and any(cache_root.iterdir())
        stray = [
            q / "_task_cache"
            for q in (data_dir / "SFT-CoT_datasets", data_dir / "SFT_datasets")
            if (q / "_task_cache") != cache_root and (q / "_task_cache").exists()
        ]

        shutil.rmtree(d)  # prepared dir gone; only the task caches remain
        LOADS.clear(); FORMATS.clear()
        mod, _ = load_entry(filename)
        out2 = run(mod, argv)
        ok_resume = (
            handoff(out2) == d
            and not LOADS
            and not FORMATS
            and sorted(datasets.load_from_disk(d)["train"]["messages"]) == rows_before
        )
        check(filename, ok_fresh and ok_schema and ok_resume and ok_location and not stray,
              f"({len(expect_tasks)} tasks, schema {'ok' if ok_schema else 'BAD'}, "
              f"resume {'0 loads/0 formats' if ok_resume else 'RECOMPUTED'}, "
              f"cache in {pathlib.Path(d).parent.parent.name}"
              f"{', STRAY ' + str(stray) if stray else ''})")

    # --- (B) reference entry point: naming + launcher paths -----------------
    print(f"\n[B] {REFERENCE}: dir naming and the training-side call")
    data_dir = tmp / "data" / REFERENCE
    argv = base_argv(data_dir, tmp / "ckpt")
    LOADS.clear(); FORMATS.clear()
    out = run(load_entry(REFERENCE)[0], argv + ALL_TASKS)
    d = handoff(out)
    total = N["tasks_AD"] + N["tasks_detect"] + N["tasks_TL"]
    check("dir name = true train sizes",
          pathlib.Path(d).name ==
          f"ds__AD{N['tasks_AD']}_D{N['tasks_detect']}_TL{N['tasks_TL']}_all{total}__resized-wh-512x512",
          f"({pathlib.Path(d).name})")

    cache_root = data_dir / "SFT-CoT_datasets" / "qwen25vl" / "_task_cache"
    before = sorted(p.name for p in cache_root.iterdir())
    skip_argv = list(argv)
    skip_argv[skip_argv.index("--skip_process_dataset") + 1] = "true"
    LOADS.clear(); FORMATS.clear()
    out3 = run(load_entry(REFERENCE)[0], skip_argv + ALL_TASKS + ["--prepared_ds_dir", d])
    after = sorted(p.name for p in cache_root.iterdir())
    check("training-side call loads nothing", not LOADS and not FORMATS,
          f"({len(LOADS)} loads, {len(FORMATS)} formats)")
    check("training-side call builds no cache", before == after)
    check("user dir passed through unchanged", handoff(out3) == d)

    LOADS.clear(); FORMATS.clear()
    out4 = run(load_entry(REFERENCE)[0], argv + [
        "--tasks_list_json_path_AD", jsons["tasks_AD"],
        "--tasks_list_json_path_TL", jsons["tasks_TL"],
    ])
    check("omitted task keeps its _D0 token",
          pathlib.Path(handoff(out4)).name ==
          f"ds__AD{N['tasks_AD']}_D0_TL{N['tasks_TL']}_all{N['tasks_AD'] + N['tasks_TL']}__resized-wh-512x512",
          f"({pathlib.Path(handoff(out4)).name})")

    # --- (C) CoT and non-CoT formatters must not share a cache entry --------
    print("\n[C] CoT and non-CoT now cache in separate folders; keys must differ anyway")
    from medvision_bm.sft.sft_utils import (
        _format_data_DetectionTask,
        _format_data_DetectionTask_CoT,
    )
    from medvision_bm.sft.task_cache import task_cache_inputs, task_cache_key

    def key_for(fn):
        return task_cache_key(task_cache_inputs(
            task_label="Detection", tasks_list_json_path=jsons["tasks_detect"],
            train_limit=-1, val_limit=VAL, tag_ds="BoxSize", mapping_func=fn,
            model_family_name="qwen25vl", base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
            process_img=False, save_processed_img_to_disk=False, new_shape_hw=[512, 512],
            temperature_sampler_task_column="__task_name"))

    k_cot, k_plain = key_for(_format_data_DetectionTask_CoT), key_for(_format_data_DetectionTask)
    check("CoT and non-CoT keys differ", k_cot != k_plain, f"({k_cot[:8]} vs {k_plain[:8]})")

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("OK")
