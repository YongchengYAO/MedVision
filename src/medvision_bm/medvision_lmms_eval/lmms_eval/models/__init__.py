import importlib
import os
import sys

from loguru import logger

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

# Commented out models are not fully tested
# ---
# NOTE:
# In the MedVision benchmark (https://medvision-vlm.github.io),
# we use the vllm inference engine for some models (e.g., Gemma3, InternVL3, Llama-3.2-Vision, Llava-OneVision, Qwen2.5-VL).
# Commented out models are those not included in MedVision Benchmark.
# ---
AVAILABLE_MODELS = {
    # Claude (Anthropic API or OpenRouter)
    "claude": "Claude",
    # Gemini (Google Gemini API or OpenRouter; 2.5 + 3 series)
    "gemini": "Gemini",
    # Gemma3
    "vllm_gemma3": "VLLM_Gemma3",
    # Gemma4
    "vllm_gemma4": "VLLM_Gemma4",
    # HealthGPT
    "healthgpt": "HealthGPT",
    # HuatuoGPT-Vision
    "huatuogpt_vision": "HuatuoGPT_Vision",
    # InternVL3
    "vllm_internvl3": "VLLM_InternVL3",
    # "internvl3": "InternVL3",
    # Lingshu
    "lingshu": "Lingshu",
    # Llama
    "vllm_llama_3_2_vision": "VLLM_Llama_3_2_Vision",
    # "llama4": "Llama4",
    # LLaVA-Med
    "llava_med": "LLaVA_Med",
    # LLaVA-OneVision
    # "llava_onevision": "Llava_OneVision",
    "vllm_llava_onevision": "VLLM_Llava_OneVision",
    # MedDr
    "meddr": "MedDr",
    # MedGemma
    "medgemma": "MedGemma",
    # MiniMax-M3 (vision-language; 428B sparse MoE, ~23B active)
    "vllm_minimax_m3": "VLLM_MiniMaxM3",
    # OpenAI (official OpenAI API or OpenRouter)
    "openai": "OpenAI_GPT",
    # Qwen2.5-VL
    # "qwen2_5_vl": "Qwen2_5_VL",
    "vllm_qwen25vl": "VLLM_Qwen25VL",
    "vllm_qwen25vl_tooluse": "VLLM_Qwen25VL_ToolUse",
    # Qwen3-VL
    "vllm_qwen3vl": "VLLM_Qwen3VL",
    # BiomedGPT
    # "biomedgpt": "BiomedGPT",
}


def get_available_model_names():
    """Return available model identifiers as a plain list."""
    return list(AVAILABLE_MODELS.keys())


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
