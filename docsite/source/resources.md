# Resources

Citation, canonical links, licensing, and credits for the MedVision benchmark and the `medvision_bm` codebase.

## Citation

If MedVision, the `medvision_bm` code, or the MedVision-V0 models are useful in your research, please cite the preprint:

```bibtex
@misc{yao2026medvisionbenchmarkingquantitativemedical,
      title={MedVision: Benchmarking Quantitative Medical Image Analysis},
      author={Yongcheng Yao and Yongshuo Zong and Raman Dutt and Yongxin Yang and Sotirios A Tsaftaris and Timothy Hospedales},
      year={2026},
      eprint={2511.18676},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18676},
}
```

## Links

Everything the project publishes lives at one of these entry points:

- **Paper (arXiv)** — <https://arxiv.org/abs/2511.18676>
- **Project page & open leaderboard** — <https://medvision-vlm.github.io> (per-task score tables, an interactive case viewer, and a frontier API-model pilot study)
- **Dataset (Hugging Face)** — <https://huggingface.co/datasets/YongchengYAO/MedVision>
- **MedVision-V0 model collection** — <https://huggingface.co/collections/YongchengYAO/medvision-v0>
- **Interactive demo (HF Space)** — <https://huggingface.co/spaces/YongchengYAO/MedVision-V0-demo>
- **Source code (GitHub)** — <https://github.com/YongchengYAO/MedVision>
- **Docker images** — <https://hub.docker.com/r/vincentycyao/medvision/tags>
- **verl fork for RFT** — <https://github.com/YongchengYAO/verl/tree/medvision-rl>

:::{tip}
For a reproducible environment, pull the published Docker image rather than resolving pinned dependencies by hand. See [Installation](getting-started/installation.md) for the setup path.
:::

## License

The `medvision_bm` package is distributed under **Creative Commons Attribution 4.0 (CC-BY 4.0)** — see the license metadata in `pyproject.toml` and <https://creativecommons.org/licenses/by/4.0/>. The MedVision annotations themselves are released under the same **CC-BY 4.0** license, which permits reuse and adaptation for academic and commercial work provided you give appropriate credit.

:::{warning}
MedVision is a **meta-dataset**: it layers new annotations on top of many independently published source datasets. The CC-BY 4.0 grant covers only MedVision's own annotations — it does **not** relicense the underlying imaging data. Any use of a given case must also honour the original license and usage terms of the dataset that case was drawn from. Confirming compliance for every constituent source is the user's responsibility.
:::

## Acknowledgements

This work was supported by UK Research and Innovation (grant **EP/S02431X/1**), through the UKRI Centre for Doctoral Training in Biomedical AI at the University of Edinburgh, School of Informatics.

MedVision builds directly on several open-source projects:

- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** — the VLM evaluation framework underpinning the benchmark harness.
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** — LLM evaluation framework.
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM/VLM inference backend.
- **[verl](https://github.com/volcengine/verl)** — reinforcement learning for LLMs, used for RFT (via the [medvision-rl fork](https://github.com/YongchengYAO/verl/tree/medvision-rl)).
- **[TRL](https://github.com/huggingface/trl)** — supervised and preference post-training utilities.
