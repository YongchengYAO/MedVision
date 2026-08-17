"""test-9: the shell entry points must actually RUN, in both MOCK modes.

Why this exists
---------------
`test-sweep.sh` once contained

    OUT="${TD}/judge-out_${T}${SUFFIX}$([ "${MOCK}" = "1" ] && echo .MOCK).jsonl"

which assigns the correct string and is valid syntax, so `bash -n` passed. But
under `set -euo pipefail` a simple command consisting only of an assignment takes
the exit status of its last command substitution, and with MOCK unset the `[`
test fails, so the substitution exits 1 and the SCRIPT DIED THERE -- silently, on
every real sweep, while MOCK=1 sailed through. The only mode testable on a CPU
box was the only mode that worked.

So: drive the real script, in both modes, in a throwaway repo, with a stub
interpreter standing in for vLLM, and assert it reaches the point where it
announces the output file it is about to write.

Run from the repo root:
    python unit-test/llm-parsing/test-9.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
SWEEP = REPO / "script" / "llm-parsing" / "test-sweep.sh"
DRIVER = REPO / "script" / "llm-parsing" / "run_llm_parsing.sh"

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


# A stand-in interpreter: answers the preflight probes, then writes one output row
# so the script's own queue/output reconciliation can run to completion.
#
# judge_config.py is the one call it must NOT fake. That is the judge registry, and
# the shell evals its output to learn which reader it is running -- a stub that
# swallows it and prints nothing leaves every JUDGE_* variable unset, which under
# `set -u` surfaces as "unbound variable" from a line that has nothing to do with
# the cause. It needs only the standard library, so hand it to a real interpreter.
STUB = """#!/usr/bin/env bash
case "${1:-}" in *judge_config.py) exec %(real_python)s "$@" ;; esac
for a in "$@"; do
  case "$a" in
    *device_count*) echo 0; echo "2.8.0 12.8"; exit 0 ;;
    *"import vllm"*) exit 0 ;;
  esac
done
case "${1:-}" in -V) echo "Python 3.11.0 (stub)"; exit 0 ;; esac
out=""
prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"
  prev="$a"
done
[ -n "$out" ] && printf '{"qid":"a","judge_status":"ok"}\\n' > "$out"
exit 0
"""

print("test-9: shell entry points run in both MOCK modes")
check("bash -n test-sweep.sh", subprocess.run(["bash", "-n", str(SWEEP)]).returncode == 0)
check("bash -n run_llm_parsing.sh", subprocess.run(["bash", "-n", str(DRIVER)]).returncode == 0)

for mock in ("0", "1"):
    with tempfile.TemporaryDirectory(prefix=f"t9_mock{mock}_") as td:
        root = pathlib.Path(td)
        (root / "script" / "llm-parsing").mkdir(parents=True)
        (root / "Results" / "MedVision-TL-v2-CoT").mkdir(parents=True)
        # Stage EVERY shell file, not just the entry point. test-sweep.sh sources
        # judge_env.sh for the judge-model resolution it shares with the driver,
        # and copying only the named script reproduced a bare "No such file or
        # directory" that reading test-sweep.sh alone does not explain. The glob
        # also means the next shared file cannot silently break this test.
        for sh in sorted(SWEEP.parent.glob("*.sh")):
            shutil.copy(sh, root / "script" / "llm-parsing" / sh.name)
        # judge_env.sh queries the judge registry, so the throwaway repo needs it.
        shutil.copy(SWEEP.parent / "judge_config.py",
                    root / "script" / "llm-parsing" / "judge_config.py")

        stub = root / "stubpy"
        stub.write_text(STUB % {"real_python": sys.executable})
        stub.chmod(0o755)

        queue = root / "Results" / "MedVision-TL-v2-CoT" / "judge-queue_TL.jsonl"
        queue.write_text(json.dumps({
            "qid": "a", "task_type": "TL", "step_key": "TL", "response": "x",
            "cache_key": "c", "prompt_fp": "f", "model": "m",
            "file": "f.jsonl", "doc_id": 1,
        }) + "\n")

        # TP=1 is pinned, not inherited from the registry. This test is about the
        # output-path assignment executing under `set -euo pipefail`, not about GPU
        # capacity -- and the registry's tensor_parallel is a capacity floor that
        # legitimately changes per reader. Left unpinned, a reader with TP>1 trips
        # the preflight's `NUM_SHARDS x TP > devices` refusal against the stub's
        # single fake device, and this test fails for a reason it does not test.
        # The topology guard itself is covered by test-11.
        env = dict(os.environ,
                   MOCK=mock, SKIP_GPU_CHECK="1", TASKS="TL",
                   NUM_SHARDS="1", TP="1", PYTHON=str(stub))
        p = subprocess.run(["bash", str(root / "script" / "llm-parsing" / "test-sweep.sh")],
                           env=env, capture_output=True, text=True, timeout=180)
        out = p.stdout + p.stderr

        # The banner prints the resolved output path, so reaching it proves the
        # assignment executed AND shows which filename the mode selected.
        reached = "queued rows ->" in out
        check(f"MOCK={mock}: reaches the per-task banner", reached,
              f"rc={p.returncode} tail={out[-400:]!r}")

        if reached:
            banner = next(l for l in out.splitlines() if "queued rows ->" in l)
            has_infix = ".MOCK.jsonl" in banner
            check(f"MOCK={mock}: output path {'has' if mock == '1' else 'has no'} .MOCK infix",
                  has_infix == (mock == "1"), banner.strip())

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-9: all checks passed")
