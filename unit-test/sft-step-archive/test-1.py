print("=== SFT training: archive a checkpoint every N steps ===")
print("Objective : Verify medvision_bm.sft.sft_utils._make_step_archive_callback")
print("            (1) off-cadence steps neither force a save nor archive (even when")
print("                a regular save_steps checkpoint is written),")
print("            (2) an on-cadence step forces a save, and on_save mirrors")
print("                checkpoint-<step> into <output_dir>/archive/checkpoint-<step>,")
print("            (3) the archive survives the source's deletion (save_total_limit")
print("                rotation) and is ignored by get_last_checkpoint,")
print("            (4) an existing archive entry is skipped, not overwritten,")
print("            (5) when hard links fail, files are copied instead,")
print("            (6) --archive_every_n_steps parses and defaults to 0 (off).")
print("Data      : synthetic checkpoint dirs in a temp folder; no model or GPU required.")

import contextlib
import io
import os
import pathlib
import shutil
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path("src").resolve()))

from transformers.trainer_callback import TrainerControl, TrainerState  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from medvision_bm.sft import sft_utils  # noqa: E402
from medvision_bm.sft.sft_utils import _make_step_archive_callback  # noqa: E402

FILES = {"model.safetensors": b"weights", "optimizer.bin": b"optim", "trainer_state.json": b"{}"}

tmp = tempfile.mkdtemp()
args = SimpleNamespace(output_dir=tmp)


def fake_save(step):
    """Stand-in for Trainer._save_checkpoint: write checkpoint-<step>."""
    ckpt = os.path.join(tmp, f"checkpoint-{step}")
    os.makedirs(ckpt)
    for name, data in FILES.items():
        with open(os.path.join(ckpt, name), "wb") as f:
            f.write(data)
    return ckpt


def archived(step):
    return os.path.join(tmp, "archive", f"checkpoint-{step}")


def run(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a)
    return out, buf.getvalue()


cb = _make_step_archive_callback(10)

# (1) step 15 with every_n=10: no forced save; a regular save at 15 is not archived.
state = TrainerState(global_step=15)
control, _ = run(cb.on_step_end, args, state, TrainerControl())
assert not control.should_save
fake_save(15)
run(cb.on_save, args, state, control)
assert not os.path.exists(os.path.join(tmp, "archive"))
print("[PASS] (1) off-cadence step: no forced save, regular checkpoint not archived")

# (2) step 20: save forced, then archived on save.
state = TrainerState(global_step=20)
control, _ = run(cb.on_step_end, args, state, TrainerControl())
assert control.should_save, "on-cadence step must force a save"
src = fake_save(20)
_, log = run(cb.on_save, args, state, control)
assert os.path.isdir(archived(20)), log
for name, data in FILES.items():
    assert pathlib.Path(archived(20), name).read_bytes() == data
assert "[Archive] archived" in log, log
print("[PASS] (2) on-cadence step: save forced, checkpoint-20 mirrored into archive/")

# (3) rotation deletes the source; the archive stays intact and resume ignores it.
shutil.rmtree(src)
for name, data in FILES.items():
    assert pathlib.Path(archived(20), name).read_bytes() == data
fake_save(30)
assert get_last_checkpoint(tmp) == os.path.join(tmp, "checkpoint-30"), get_last_checkpoint(tmp)
print("[PASS] (3) archive survives source deletion; get_last_checkpoint ignores it")

# (4) an archive entry that already exists (e.g. manual copy) is left alone.
os.makedirs(archived(40))
pathlib.Path(archived(40), "manual.txt").write_text("keep")
fake_save(40)
_, log = run(cb.on_save, args, TrainerState(global_step=40), TrainerControl())
assert os.listdir(archived(40)) == ["manual.txt"], os.listdir(archived(40))
assert "already exists" in log, log
print("[PASS] (4) existing archive entry skipped, not overwritten")

# (5) filesystem refuses hard links: fall back to a real copy.
real_link = os.link


def no_link(*_a, **_k):
    raise OSError("links not supported")


os.link = no_link
try:
    fake_save(50)
    run(cb.on_save, args, TrainerState(global_step=50), TrainerControl())
finally:
    os.link = real_link
for name, data in FILES.items():
    a, s = pathlib.Path(archived(50), name), pathlib.Path(tmp, "checkpoint-50", name)
    assert a.read_bytes() == data
    assert a.stat().st_ino != s.stat().st_ino, "fallback must copy, not link"
print("[PASS] (5) hard-link failure falls back to copying")

# (6) CLI flag.
base = ["prog", "--model_family_name", "qwen25vl", "--base_model_hf", "x", "--data_dir", "d"]
sys.argv = base
assert sft_utils.parse_args_multiTask().archive_every_n_steps == 0
sys.argv = base + ["--archive_every_n_steps", "3000"]
assert sft_utils.parse_args_multiTask().archive_every_n_steps == 3000
print("[PASS] (6) --archive_every_n_steps parses; default 0")

shutil.rmtree(tmp)
print("\nALL PASS")
