# Requirements Catalogue (per-model pin sets)

The repository ships one frozen requirements file per evaluated model family (21 eval files) and per SFT base model (4 files) under `requirements/`. Values below were extracted with `grep` from the files at the commit this skill was generated from; **nine of the 25 files pin `flash-attn` as a direct wheel URL** — five eval sets (`healthgpt`, `huatuogpt_vision`, `lingshu` on cp311; `llavamed` cp310; `meddr` cp39) and all four SFT sets (cp311); the SFT no-requirements path installs the same 2.7.3 cp311 wheel programmatically, see `cli-reference.md`. `datasets==3.6.0` is pinned in every file (`trust_remote_code` was removed in datasets 4.x). `-` means the package is not listed.

Use `../scripts/check_env_pins.py --requirements <file>` (or `--model <key>`; the script embeds this snapshot as a fallback when the checkout is unavailable) to compare an environment against any row.

**Locally-built packages are excluded by design.** `pip freeze` originally captured `medvision_bm`,
`medvision_lmms_eval`, `medvision_ds` and `llava_med` as machine-local `file://` / `-e` pins, which broke
`pip install -r` on any other host. Those lines now carry a comment naming the installer instead
(`install_vendored_lmms_eval`, `install_medvision_ds`, the launcher's source build, and
`eval__llava_med`'s `--dir_third_party` clone). Re-freezing these files will reintroduce them — strip them again.

**Python version.** Every row below assumes **Python 3.11** except three families whose launchers and
Dockerfiles pin another interpreter: MedDr (`meddr`) **3.9**, LLaVA-Med (`llavamed`) **3.10**, and
MiniMax-M3 / MiniMax-M3-INT4 (`minimax-m3-int4`; the non-INT4 launcher pins no requirements file) **3.12**. Their flash-attn wheels are cp39/cp310-specific (or the stack is
py3.12-only), so installing them under 3.11 fails with "not a supported wheel on this platform".

## Eval requirements (`requirements/requirements_eval_<key>.txt`)

| File | torch | torchvision | vllm | transformers | accelerate | huggingface_hub | numpy | protobuf | other notable pins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `requirements_eval_claude.txt` | - | - | - | 4.57.1 | 1.14.0 | 0.36.0 | 2.4.6 | 6.33.6 | anthropic 0.109.1, openai 2.41.1, peft 0.19.1, bitsandbytes 0.49.2, deepspeed 0.19.1 |
| `requirements_eval_gemini.txt` | - | - | - | 4.57.1 | 1.14.0 | 0.36.0 | 2.4.6 | 6.33.6 | google-genai 2.8.0, openai 2.41.1, peft 0.19.1 |
| `requirements_eval_gpt.txt` | - | - | - | 4.57.1 | 1.14.0 | 0.36.0 | 2.4.6 | 6.33.6 | openai 2.41.1, peft 0.19.1 |
| `requirements_eval_kimi.txt` | - | - | - | 4.57.1 | 1.14.0 | 0.36.0 | 2.4.6 | 6.33.6 | openai 2.41.1, peft 0.19.1 |
| `requirements_eval_gemma3.txt` | 2.8.0 | 0.23.0 | 0.10.2 | 4.57.1 | 1.10.1 | 0.35.3 | 2.2.6 | 6.33.0 | torchaudio 2.8.0, xformers 0.0.32.post1, deepspeed 0.18.0 |
| `requirements_eval_gemma4.txt` | 2.10.0 | 0.25.0 | 0.19.0 | 5.10.2 | 1.13.0 | 1.18.0 | 2.2.6 | 6.33.6 | torchaudio 2.10.0, anthropic 0.107.1, deepspeed 0.19.1 |
| `requirements_eval_glm4v.txt` | 2.10.0 | 0.25.0 | 0.19.1 | 5.12.1 | 1.14.0 | 1.20.1 | 2.2.6 | 6.33.6 | torchaudio 2.10.0, deepspeed 0.19.2 |
| `requirements_eval_healthgpt.txt` | 2.6.0 | 0.21.0 | - | - (model repo pins its own) | 0.27.0 | 0.35.3 | 1.26.4 | 3.20.0 | torchaudio 2.6.0, bitsandbytes 0.41.0, deepspeed 0.9.5, peft 0.4.0 |
| `requirements_eval_huatuogpt_vision.txt` | 2.6.0 | 0.21.0 | - | 4.40.0 | 1.10.1 | 0.35.3 | 1.26.4 | 3.20.0 | torchaudio 2.6.0, deepspeed 0.18.0 |
| `requirements_eval_internvl3.txt` | 2.7.1 | 0.22.1 | 0.10.0 | 4.57.1 | 1.10.1 | 0.35.3 | 2.2.6 | 6.33.0 | xformers 0.0.31, openai 1.90.0 |
| `requirements_eval_lingshu.txt` | 2.6.0 | 0.21.0 | - | 4.52.1 | 1.10.1 | 0.35.3 | 1.26.4 | 3.20.0 | torchaudio 2.6.0 |
| `requirements_eval_llama3_vision.txt` | 2.8.0 | 0.23.0 | 0.10.2 | 4.57.1 | 1.10.1 | 0.35.3 | 2.2.6 | 6.33.0 | xformers 0.0.32.post1 |
| `requirements_eval_llava_onevision.txt` | 2.7.1 | 0.22.1 | 0.10.0 | 4.57.1 | 1.10.1 | 0.35.3 | 2.2.6 | 6.33.0 | xformers 0.0.31, openai 1.90.0 |
| `requirements_eval_llavamed.txt` | 2.6.0 | 0.21.0 | - | 4.37.2 | 1.10.1 | 0.35.3 | 1.26.4 | 3.20.0 | torchaudio 2.6.0 |
| `requirements_eval_meddr.txt` | 2.6.0 | 0.21.0 | - | 4.37.2 | 0.34.2 | 0.35.3 | 1.26.4 | 3.20.0 | bitsandbytes 0.45.2, peft 0.10.0 |
| `requirements_eval_medgemma.txt` | 2.9.0 | 0.24.0 | - | 4.57.1 | 1.10.1 | 0.35.3 | 2.2.6 | 6.33.0 | no torchaudio; HF transformers inference |
| `requirements_eval_medvision-v0.txt` | 2.7.1 | 0.22.1 | 0.10.0 | 4.54.1 | 1.9.0 | 0.35.3 | 2.2.6 | 6.33.0 | xformers 0.0.31, openai 1.90.0 (identical stack to qwen25vl) |
| `requirements_eval_minimax-m3-int4.txt` | 2.11.0 | 0.26.0 | - (vLLM fork built from source) | 5.12.1 | 1.14.0 | 1.20.1 | 2.3.5 | 6.33.6 | setuptools 80.10.2, torchaudio 2.11.0, anthropic 0.111.0 |
| `requirements_eval_qwen25vl.txt` | 2.7.1 | 0.22.1 | 0.10.0 | 4.54.1 | 1.9.0 | 0.35.3 | 2.2.6 | 6.33.0 | xformers 0.0.31, openai 1.90.0 |
| `requirements_eval_qwen25vl_update1.txt` | 2.9.1 | 0.24.1 | 0.14.0 | 5.0.0rc2 | 1.9.0 | 1.3.3 | 2.2.6 | 6.33.4 | experimental transformers-5 variant; matches the commented alternative in `eval__qwen2_5_vl` |
| `requirements_eval_qwen3vl.txt` | 2.8.0 | 0.23.0 | 0.11.0 | 4.57.0 | 1.13.0 | 0.36.0 | 2.2.6 | 6.33.6 | xformers 0.0.32.post1 |

Reading the table:

- **transformers 4.x rows** (hub `0.35.3`/`0.36.0`) and **transformers 5.x rows** (hub `>=1.3`) are incompatible with each other's `huggingface_hub`: transformers 4.x needs `huggingface_hub<1.0`, transformers 5.x needs `>=1.5` (glm4v/gemma4/minimax rows). Never mix rows in one environment.
- **torch ↔ vllm** pairs are ABI-bound: vllm 0.10.0 ↔ torch 2.7.1, vllm 0.10.2 ↔ torch 2.8.0, vllm 0.11.0 ↔ torch 2.8.0, vllm 0.14.0 ↔ torch 2.9.1, vllm 0.19.x ↔ torch 2.10.0. Installing `medvision_bm` **with** dependencies afterwards pulls `torch==2.6.0` and breaks these pairs (see `troubleshooting.md`).
- API-model rows (claude/gemini/gpt/kimi) carry **no torch**; the corresponding Docker images uninstall CUDA torch and install `torch==2.6.0+cpu`.

## SFT requirements (`requirements/requirements_sft_<key>.txt`)

| File | torch | torchvision | transformers | accelerate | huggingface_hub | numpy | protobuf | other notable pins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `requirements_sft_qwen25vl.txt` | 2.6.0 | 0.21.0 | 4.54.0 | 1.11.0 | 0.35.3 | 2.2.6 | 6.33.0 | trl 0.19.1, peft 0.17.1, bitsandbytes 0.48.1, torchaudio 2.6.0 |
| `requirements_sft_medgemma.txt` | 2.6.0 | 0.21.0 | 4.54.0 | 1.9.0 | 0.36.0 | 2.4.6 | 6.33.0 | trl 0.19.1, peft 0.19.1, deepspeed 0.19.2, torchaudio 2.6.0+cu124 |
| `requirements_sft_gemma4.txt` | 2.6.0+cu124 | 0.21.0+cu124 | 5.5.0 | 1.14.0 | 1.22.0 | 2.4.6 | 6.33.0 | trl 0.19.1, peft 0.19.1, deepspeed 0.19.2 |
| `requirements_sft_qwen3.6vl.txt` | 2.6.0+cu124 | 0.21.0+cu124 | 5.5.0 | 1.14.0 | 1.22.0 | 2.4.6 | 6.33.0 | trl 0.19.1, peft 0.19.1, deepspeed 0.19.2 |

No SFT file pins `vllm`. SFT-specific launch recipes live in `../../sft/SKILL.md`; the RFT parquet builders reuse `requirements_sft_qwen25vl.txt` through `medvision_bm.sft.env_setup` (`../../rft/SKILL.md`).

## Model key → wrapper → extra → automatic install parameters

What each `python -m medvision_bm.benchmark.eval__<module>` wrapper installs when run **without** `--skip_env_setup` (values read from each wrapper's `main()`; `hub` = `ensure_hf_hub_installed(...)` version, only applied when no `huggingface_hub` is importable). The lmms_eval extra is what to pass as `--lmms_eval_opt_deps`.

| Requirements key | Wrapper module | lmms_eval extra | hub | torch step | vLLM | transformers step (after vLLM) | Docker tag / conda env |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `claude` | `eval__claude` | `claude` | 0.36.0 | - | - | - | `eval_claude` / `eval-claude` |
| `gemini` | `eval__gemini` | `gemini` | 0.36.0 | - | - | - | `eval_gemini` / `eval-gemini` |
| `gpt` | `eval__openai` | `openai` | 0.36.0 | - | - | - | `eval_gpt` / `eval-openai` |
| `kimi` | `eval__kimi` | `kimi` | 0.36.0 | - | - | - | `eval_kimi` / `eval-kimi` |
| `gemma3` | `eval__gemma3` | (none) | 0.35.3 | `install_torch_cu124` | 0.10.2 | - | `eval_gemma3` / `eval-gemma3` |
| `gemma4` | `eval__gemma4` | (none) | 0.35.3 | `install_torch_cu124` | 0.19.0 | `transformers>=5.5.0` | `eval_gemma4` / `eval-gemma4` |
| `glm4v` | `eval__glm4v` | `glm4v` | 0.35.3 | `install_torch_cu124` | 0.19.1 | `transformers==5.12.1` | `eval_glm4v` / `eval-glm4v` |
| `healthgpt` | `eval__healthgpt` | (none) | 0.35.3 | `install_flash_attention_torch_and_deps_py311_v2` | - | - | `eval_healthgpt` / `eval-healthgpt` |
| `huatuogpt_vision` | `eval__huatuogpt_vision` | `huatuogpt_vision` | 0.35.3 | `install_flash_attention_torch_and_deps_py311_v2` | - | - | `eval_huatuogptvision` |
| `internvl3` | `eval__intern_vl3` | (none) | 0.35.3 | `install_torch_cu124` | 0.10.0 | - | `eval_internvl3` |
| `lingshu` | `eval__lingshu` | `lingshu` | 0.35.3 | `install_flash_attention_torch_and_deps_py311_v2` | - | - | `eval_lingshu` |
| `llama3_vision` | `eval__llama3_2_vision` | (none) | 0.35.3 | `install_torch_cu124` | 0.10.2 | - | `eval_llama3vision` |
| `llava_onevision` | `eval__llava_onevision` | (none) | 0.35.3 | `install_torch_cu124` | 0.10.0 | - | `eval_llavaonevision` |
| `llavamed` | `eval__llava_med` | `llava_med` | 0.35.3 | `install_flash_attention_torch_and_deps_py310_v2` | - | - | `eval_llavamed` |
| `meddr` | `eval__meddr` | `meddr` | 0.35.3 | `install_flash_attention_torch_and_deps_py39_v2` | - | - | `eval_meddr` |
| `medgemma` | `eval__medgemma` | (none) | 0.36.0 | - | - | - | `eval_medgemma` |
| `medvision-v0` | `eval__medvision-model-rft` | `qwen2_5_vl` (the `medvision_v0` extra is identical) | 0.35.3 | `install_torch_cu124` | 0.10.0 | `transformers==4.54.1`, `accelerate==1.9.0` | `eval_medvision-v0` / `eval-medvision-v0` |
| `minimax-m3-int4` | `eval__minimax_m3` | `minimax_m3` | 0.35.3 | `install_torch_cu124` | `--vllm_version` (wrapper default 0.11.0); the Docker image passes 0.23.0 and then replaces vLLM with a source-built fork via `VLLM_USE_PRECOMPILED=1` | `transformers==4.57.1` | `eval_minimax-m3-int4` |
| `qwen25vl` | `eval__qwen2_5_vl` | `qwen2_5_vl` | 0.35.3 | `install_torch_cu124` | 0.10.0 | `transformers==4.54.1`, `accelerate==1.9.0` | `eval_qwen25vl` / `eval-qwen25vl` |
| `qwen25vl_update1` | `eval__qwen2_5_vl` (commented alternative) | `qwen2_5_vl` | - | - | 0.14.0 | `transformers==5.0.0rc2` | `eval_qwen25vl_update1` |
| `qwen3vl` | `eval__qwen3_vl` | `qwen3_vl` | 0.35.3 | `install_torch_cu124` | 0.11.0 | `transformers==4.57.0` | `eval_qwen3vl` |
| (tool-use SFT models) | `eval__qwen25vl_tooluse` | (none) | - | - | 0.10.0 | `transformers==4.54.1` | reuse `eval-qwen25vl` |

Notes:

- The wrapper's `install_torch_cu124` step installs **torch 2.6.0+cu124**, which the later `pip install vllm==<ver>` then replaces with the torch that vLLM version requires (e.g. 2.7.1 for 0.10.0). This is why the automatic path ("Method 2") is slower and more fragile than installing the frozen requirements file ("Method 1", see `installation.md`).
- `hub` is applied by `ensure_hf_hub_installed`, which is a **no-op** when any `huggingface_hub` is already importable — the effective hub version comes from the requirements file or from what `pip install vllm` / the transformers step resolve.
- Conda environment names in the launchers and Dockerfiles **mostly** follow `eval-<key>` / `sft-<key>`, with five exceptions: `gpt` → `eval-openai`, `huatuogpt_vision` → `eval-huatuogpt-vision`, `llama3_vision` → `eval-llama3-vision`, `llava_onevision` → `eval-llava-onevision`, `qwen25vl_update1` → `eval-qwen25vl`; Docker image tags follow `eval_<key>` / `sft_<key>` (`vincentycyao/medvision:<tag>`). Three Dockerfile tags spell the key differently from the requirements file: `eval_huatuogptvision`, `eval_llama3vision` and `eval_llavaonevision`.
- Hardware per model (VRAM, GPU count, tensor-parallel notes) is in `../../../references/model-roster.md`.

## Vendored `lmms_eval` extras (`--lmms_eval_opt_deps`)

From the vendored `lmms_eval` `pyproject.toml` `[project.optional-dependencies]` (extras with model-relevant pins; `audio`, `metrics`, `reka`, `qwen`, `mmsearch`, `all` also exist):

| Extra | Pins |
| --- | --- |
| `meddr` | `bitsandbytes==0.45.2` |
| `lingshu` | `transformers==4.52.1`, `qwen-vl-utils==0.0.11` |
| `huatuogpt_vision` | `transformers==4.40.0` |
| `llava_med` | `transformers==4.37.2`, `numpy==1.26.4`, `protobuf>=3.20` |
| `qwen2_5_vl` | `transformers==4.54.1`, `accelerate==1.9.0`, `decord`, `qwen_vl_utils`, `qwen-vl-utils[decord]>=0.0.8` |
| `medvision_v0` | identical to `qwen2_5_vl` |
| `qwen3_vl` | `transformers==4.57.0`, `qwen-vl-utils==0.0.14` |
| `minimax_m3` | `transformers==4.57.1` |
| `glm4v` | `transformers==5.12.1` (needs `Glm46VImageProcessor`, transformers>=5.2; vLLM 0.19.x needs >=5.6) |
| `gemini` | `google-genai>=2.8.0`, `openai`, `transformers==4.57.1` |
| `claude` | `anthropic`, `openai`, `transformers==4.57.1` |
| `openai` | `openai`, `transformers==4.57.1` |
| `kimi` | `openai`, `transformers==4.57.1` |

The base dependencies of the vendored package pin `datasets==3.6.0`, `torch>=2.1.0,<2.8` and `torchvision>=0.16.0,<0.23` ("cap below 2.8 for vllm 0.10.0 ABI compatibility") and use `opencv-python-headless`; note that `medvision_ds` requires the non-headless `opencv-python` (see `troubleshooting.md` for the libGL consequence). The API-model extras pin `transformers==4.57.1` explicitly because the base `transformers` dependency is unpinned and transformers 5.x imports `is_offline_mode` from `huggingface_hub`, which 0.36.0 does not provide.
