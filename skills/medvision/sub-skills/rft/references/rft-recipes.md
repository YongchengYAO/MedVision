# RFT recipes: rewards, task mixing, curriculum, and the MedVision-V0 configuration

**Reference only.** GRPO training does not run from `medvision_bm`; it runs in the external verl fork
(`https://github.com/YongchengYAO/verl`, branch **`medvision-rl`**, rebased on upstream verl; the paper cites verl
v0.7.0). Facts below come from the repository README's RFT section, the MedVision-V0 model card, the paper-facts
summary shipped with the repository, and the fork's `README.md`, `REWARDS.md`, `CURRICULUM_FILTERING.md` and
`examples/grpo_trainer/train__rft-*.sh` headers. Nothing here is bundled; do not treat the values as runnable on a
CPU host. Training requires 4x H200-class GPUs (paper) and a verl environment.

## 1. Two-stage MedVision-V0 recipe (paper)

| Stage | What | Data | Where |
| --- | --- | --- | --- |
| 1. SFT (CoT) | full fine-tuning of `Qwen/Qwen2.5-VL-7B-Instruct`, 3 epochs, effective batch 256, bf16 FSDP, 512x512, weighted sampler | 121K CoT samples: 110K detection / 5.5K T/L / 5.5K A/D, **axial slices only for detection and T/L** (coronal/sagittal held out for the plane-OOD split); the A/D pool is mostly sagittal/coronal and has no plane-OOD split | `medvision_bm.sft` -> `../../sft/SKILL.md` |
| 2. RFT (GRPO) | full-parameter GRPO on the SFT checkpoint, **sequentially A/D -> T/L -> detection**, as three single-task parquets built from the same training pools — A/D 5.5K, T/L 5.5K, detection 1M — with the CoT answers removed (prompts keep the CoT instruction); the 121K mixture is used by the multi-task ablation, not by this stage | parquet built by this sub-skill's builders at 512x512 for `qwen25vl` | verl fork recipes `train__rft-sequential__{1-AD,2-TL,3-detection}.sh` |

The released `YongchengYAO/MedVision-V0-7B` is stage 3's `global_step_250` (recipe header of
`train__rft-sequential__3-detection.sh`); stage 3 trains on the **1M detection** parquet
(`ds__AD0_D1000000_TL0_all1000000__resized-hw-512x512`, built with the checkpointed builder), stages 1-2 on the 5.5K
single-task parquets.

## 2. Rewards (`verl/utils/reward_score/medvision_rewards/`, fork `REWARDS.md`)

```
r = r_format + r_process * r_answer     composition=multiplicative  (default; paper Algorithm 2)
r = r_format + r_process + r_answer     composition=additive        (reward-design ablation)
r = r_format + r_answer                 detection (no process reward)
```

Every component is in [0, 1]; accuracy components map an error `e` through `rho(e) = exp(-e)` (`reward_mapping_func=exp_decay`,
k = 1; alternatives `scaled_sigmoid`, `gaussian_proxy`). Unparseable value -> 0.

| Component | A/D and T/L | Detection |
| --- | --- | --- |
| `r_format` | `soft` (default): `0.8 x reasoning-structure score + 0.2 x binary <answer> check`; `binary`: answer check only | binary answer check |
| `r_process` | mean over CoT steps of `rho(step error)`: localization steps = worst-point displacement / sqrt(2) (two-endpoint steps take the better ordering); measurement steps = relative error. T/L 4 steps (major endpoints, minor endpoints, major length, minor length); A/D 3 steps (landmark/line 1, landmark/line 2, value) | -- |
| `r_answer` | `rho(mean relative error)` of the `<answer>` values (1 number A/D, 2 numbers T/L) | `rho((1 - CIoU) / 2)`, `CIoU = IoU - d^2/c^2 - alpha*v` -- graded even for non-overlapping boxes |

Binary answer check: `<answer>` holds exactly the task's count of non-negative decimals separated by commas (2 T/L,
1 A/D, 4 detection), optionally in `( )` / `[ ]`. Ground-truth landmarks come from the parquet `extra_info`
(`landmark_P{1..4}_wh`, `landmark_{1,2}_wh`, `line_{1,2}_point_{1,2}_wh`); dispatch uses `ability` (injected into
`extra_info` by the fork's reward managers) and `metric_type`. Each call returns `score`, `format_reward`,
`process_reward`, `answer_reward`, `answer_error`, `localization_error`, `measurement_error` (NaN where N/A).

Hydra wiring used by every recipe:

```
custom_reward_function.path=<verl>/verl/utils/reward_score/medvision_rewards/medvision_general.py
custom_reward_function.name=compute_score
+custom_reward_function.reward_kwargs.format_reward=soft          # soft | binary
+custom_reward_function.reward_kwargs.composition=multiplicative  # multiplicative | additive
+reward_model.use_reward_loop=True                                # async reward loop; required for the curriculum's per-sample answer_error
```

## 3. Temperature-scaled task mixing (`verl/utils/dataset/temperature_sampler.py`)

Task `t` with `N_t` rows is drawn with probability `P(t) ~ N_t^(1/T)`; one epoch draws `len(dataset)` rows with
replacement (seeded by `data.seed`). `T=1` = natural proportions (90.9 / 4.5 / 4.5 % on the 121K mix); **`T=8`** (the
multi-task RFT runs) gives Detection 42.1 % / A/D 29.0 % / T/L 29.0 %. Angle and distance are merged into one A/D task.
Falls back to the stock sampler with fewer than two distinct tasks (so the sequential single-task stages do not set it).

```
+data.temperature_sampler.enable=True +data.temperature_sampler.T=8 +data.temperature_sampler.task_key=ability
"+data.temperature_sampler.task_group_map='medvision-angle:AD,medvision-distance:AD'"
```

## 4. Epoch-level curriculum (`verl/utils/dataset/curriculum.py`, fork `CURRICULUM_FILTERING.md`)

Per-task hard / easy pools. At each epoch end, the drawn hard samples are ranked by EMA reward and the top fraction
whose EMA answer error clears the task gate is promoted to "easy" (dropped from training); a retention mix-in of easy
samples ramps up with the solved fraction; a rotating audit slice re-validates the easy pool with hysteresis-guarded
demotion; a per-task floor keeps every task represented; temperature weights are recomputed over the active set every
epoch (epochs shrink). Pool snapshots: `<default_local_dir>/curriculum_pools/epoch_NNNN.json` (row positions, not case ids).

| Option (`+data.curriculum.*`) | Default (recipe) | Paper symbol / meaning |
| --- | --- | --- |
| `enable` | `False` (recipe: `True`) | master switch (off = stock training) |
| `easy_top_frac` | 0.20 | `f`, promotion fraction per epoch |
| `mre_gate` | 0.10 | `g_t` for A/D, T/L (MRE < 0.1 = the benchmark's success bar) |
| `detection_gate` | 0.25 | `g_t` for detection on `(1 - CIoU)/2` (<=> CIoU > 0.5) |
| `threshold_frac` | 0.50 | `p*`, solved fraction at full mix-in |
| `mixin_easy_frac` | 0.30 | `m`, max easy share |
| `mixin_ramp` | True | ramp vs binary phase |
| `task_floor_frac` | 0.10 | `phi`, anti-extinction floor |
| `demote_easy` | True | re-demote regressing easy samples |
| `ema_alpha` | 0.4 | EMA weight |
| `promote_patience` / `demote_patience` | 1 / 1 | consecutive passing / failing epochs |
| `demote_margin` | 1.5 | `lambda`, hysteresis (demote at MRE >= 0.15) |
| `audit_frac` | 0.05 | `a`, rotating audit slots |
| `task_key` / `task_group_map` | `ability` / `'medvision-angle:AD,medvision-distance:AD'` | pooling column and merges |

Known limitation (fork docs): the LR-schedule horizon is computed from the first epoch, so a decaying schedule does not
fully decay as epochs shrink (a constant LR is fine).

## 5. Recipes (`examples/grpo_trainer/` in the fork)

| Script | Experiment | Base model | `DATASET_ROOT` variant | Reward |
| --- | --- | --- | --- | --- |
| `train__rft-sequential__1-AD.sh` | MedVision-V0 stage 1 | full-SFT CoT checkpoint | `ds__AD5500_D0_TL0_all5500__resized-hw-512x512` | soft, multiplicative |
| `train__rft-sequential__2-TL.sh` | stage 2 | stage 1 `global_step_N/actor/merged_hf_model` | `ds__AD0_D0_TL5500_all5500__resized-hw-512x512` | soft, multiplicative |
| `train__rft-sequential__3-detection.sh` | stage 3 -> **MedVision-V0** (`global_step_250`) | stage 2 merged model | `ds__AD0_D1000000_TL0_all1000000__resized-hw-512x512` (accepts `shards/train_shard_*.parquet`) | soft, multiplicative |
| `train__rft-multitask.sh` | multi-task ablation: one stage, T=8 mixing, curriculum | full-SFT CoT checkpoint (`BASE_MODEL_HF` default is a **private** repo -- override) | `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512` | soft, multiplicative |
| `train__rft-multitask__additive-reward.sh` | reward-design ablation | same | same | soft, **additive** |

Variables (all scripts fail fast when missing):

| Variable | Meaning |
| --- | --- |
| `DATASET_ROOT` (required) | prepared parquet directory; reads `train_verl.parquet` + `validation_verl.parquet` (stage 3 / multitask prefer `shards/train_shard_*.parquet` when present) |
| `BASE_MODEL_PATH` **xor** `BASE_MODEL_HF` | local checkpoint dir, or a Hub id that the recipe **downloads first** into `models/<name>` (a Hub id must never reach verl directly: ~15 Ray/vLLM workers would fetch concurrently and one failure leaves a vLLM server without a processor -> CUDA device-side assert) |
| `EXP_NAME` | checkpoints / logs under `checkpoints|log/<EXP_NAME>`; reuse to resume |
| `ENGINE` (`vllm`), `ENV_NAME` (`verl`), `DRY_RUN=1` | rollout engine, conda env, print-only preview |
| trailing args | Hydra overrides, e.g. `+data.curriculum.enable=False`, `+custom_reward_function.reward_kwargs.format_reward=binary`, `trainer.total_epochs=5` |

Shared GRPO configuration (4x H200, full-parameter, vLLM rollouts; verified from the recipe bodies):
`algorithm.adv_estimator=grpo`, `data.train_batch_size=256`, `ppo_mini_batch_size=128`,
`ppo_micro_batch_size_per_gpu=4`, `actor.optim.lr=3e-6`, `use_kl_loss=True`, `kl_loss_coef=0.01`
(`low_var_kl`), `entropy_coeff=0`, `rollout.n=8`, `max_prompt_length=4096`, `max_response_length=4096`,
`filter_overlong_prompts=False`, `truncation='error'`, `data.image_key=images`,
`rollout.tensor_model_parallel_size=2`, `gpu_memory_utilization=0.55` (multitask: 0.45), gradient checkpointing on,
no param/optimizer offload, `n_gpus_per_node=4`, `test_freq=10`, `data.custom_cls=MedVisionDataset`, `wandb` +
console logging. Stage-specific: `total_epochs=100`, `save_freq=50` (stages 1-2); `total_epochs=10`, `save_freq=10`,
`RAY_memory_usage_threshold=0.98` (stage 3, multitask); multitask adds `data.seed=1024`, the sampler (§3) and the
curriculum (§4) options.

After training each recipe merges the latest checkpoint:

```
python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/global_step_N/actor --target_dir <ckpt>/global_step_N/actor/merged_hf_model
```

The fork patches the merged `config.json` to `dtype=bfloat16` and pins `transformers<5` because transformers 5.x
writes nested Qwen2.5-VL configs that the pinned MedVision eval stack (`transformers==4.54.1`) cannot read.

## 6. Environment of the fork (not bundled)

`setup_conda_verl.sh` creates the `verl` conda env (Python 3.12) with pinned torch / vLLM / SGLang / flash-attn wheels;
`requirements_medvision.txt` is a frozen `pip freeze` (257 packages). Unit tests in the fork:
`tests/utils/reward_score/test_medvision_reward_on_cpu.py`, `test_medvision_ciou_on_cpu.py`,
`tests/utils/dataset/test_curriculum_on_cpu.py` (CPU). Environment questions for `medvision_bm` itself go to
`../../environment-setup/SKILL.md`.
