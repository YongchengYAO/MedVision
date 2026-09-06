# Resources

Licensing and credits for the MedVision benchmark and the `medvision_bm` codebase. See the [front page](index.md) for the citation and the canonical project links.

:::{tip}
For a reproducible environment, pull the published Docker image rather than resolving pinned dependencies by hand. See [Installation](getting-started/installation.md) for the setup path.
:::

## License

The `medvision_bm` package is distributed under **Creative Commons Attribution 4.0 (CC-BY 4.0)** — see the license metadata in `pyproject.toml` and <https://creativecommons.org/licenses/by/4.0/>. The MedVision annotations themselves are released under the same **CC-BY 4.0** license, which permits reuse and adaptation for academic and commercial work provided you give appropriate credit.

:::{warning}
MedVision is a **meta-dataset**: it layers new annotations on top of many independently published source datasets. The CC-BY 4.0 grant covers only MedVision's own annotations — it does **not** relicense the underlying imaging data. Any use of a given case must also honour the original license and usage terms of the dataset that case was drawn from. Confirming compliance for every constituent source is the user's responsibility.
:::

## Acknowledgements

This work was supported by UK Research and Innovation (grant **EP/S02431X/1**), through the UKRI Centre for Doctoral Training in Biomedical AI at the University of Edinburgh, School of Informatics. It was also supported by the Edinburgh International Data Facility (EIDF) and the Data-Driven Innovation Programme at the University of Edinburgh.

MedVision builds directly on several open-source projects:

- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** — the VLM evaluation framework underpinning the benchmark harness.
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** — LLM evaluation framework.
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM/VLM inference backend.
- **[verl](https://github.com/volcengine/verl)** — reinforcement learning for LLMs, used for RFT (via the [medvision-rl fork](https://github.com/YongchengYAO/verl/tree/medvision-rl)).
- **[TRL](https://github.com/huggingface/trl)** — supervised and preference post-training utilities.
