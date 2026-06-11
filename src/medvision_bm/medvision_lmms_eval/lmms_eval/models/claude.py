import base64
import io
import json
import math
import os
from typing import List, Optional, Tuple, Union

import backoff
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Per-model image-resolution caps (long_edge_cap_px, max_image_tokens), taken from
# the official Anthropic vision docs ("Evaluate image size"):
#   https://platform.claude.com/docs/en/build-with-claude/vision
#
# WHY THIS IS AN EXPLICIT, ENUMERATED LIST (and not a generic default):
#   MedVision's quantitative tasks (Tumor/Lesion size, Angle/Distance) put the image
#   size and pixel size into the prompt, and the model must do the pixel->mm arithmetic
#   itself. Those numbers MUST match the resolution the model's vision encoder actually
#   perceives after its internal resize. Models in the SAME family can differ here:
#   high-resolution vision (2576 px / 4784 image tokens) exists on Claude Fable 5,
#   Opus 4.8 and Opus 4.7, but NOT on Opus 4.6 / 4.5 or the Sonnet/Haiku tiers
#   (1568 px / 1568 tokens). A silent fallback would emit a wrong pixel size for an
#   unverified model and corrupt every measurement -- so an unrecognized model RAISES.
#   To add a model: confirm its caps in the docs above and add an entry here. This is the
#   single source of truth -- the task layer (medvision_utils.get_resized_img_shape) imports
#   anthropic_resized_hw() from here, so there is no second table to update.
#
# Keys are matched as PREFIXES of the normalized model code (see _normalize_model_code):
# the leading "anthropic/" is stripped and "." -> "-", so "claude-opus-4-8" (Anthropic
# direct) and "anthropic/claude-opus-4.8" (OpenRouter), plus "-fast"/date suffixes, all
# resolve to the same entry. The LONGEST matching prefix wins.
# ---------------------------------------------------------------------------
_HIGH_RES = (2576, 4784)
_STANDARD_RES = (1568, 1568)
SUPPORTED_MODEL_CAPS = {
    # High-resolution vision (4784 image tokens / 2576 px long edge)
    "claude-fable-5": _HIGH_RES,
    "claude-opus-4-8": _HIGH_RES,
    "claude-opus-4-7": _HIGH_RES,
    # Standard vision (1568 image tokens / 1568 px long edge)
    "claude-opus-4-6": _STANDARD_RES,
    "claude-opus-4-5": _STANDARD_RES,
    "claude-sonnet-4-6": _STANDARD_RES,
    "claude-sonnet-4-5": _STANDARD_RES,
    "claude-haiku-4-5": _STANDARD_RES,
}


def _normalize_model_code(model_code: str) -> str:
    # OpenRouter model IDs look like "anthropic/claude-opus-4.8"; normalize to the
    # bare Anthropic form ("claude-opus-4-8") for capability matching.
    return model_code.split("/")[-1].replace(".", "-")


def anthropic_image_caps(model_code: str) -> Tuple[int, int]:
    """Return (long_edge_cap_px, max_image_tokens) for a Claude model code.

    Raises ValueError for any model not in SUPPORTED_MODEL_CAPS: its image-resolution
    caps are unverified, so the image/pixel size stated in MedVision TL/AD prompts could
    be wrong. This is intentional -- a hard error beats silently corrupting measurements.
    """
    normalized = _normalize_model_code(model_code)
    matched = [prefix for prefix in SUPPORTED_MODEL_CAPS if normalized.startswith(prefix)]
    if not matched:
        raise ValueError(
            f"[claude] Unsupported model code {model_code!r} (normalized {normalized!r}). "
            f"Its image-resolution caps are not verified, so the image size / pixel size "
            f"stated in MedVision TL/AD prompts could be wrong. Look up the per-model limits "
            f"at https://platform.claude.com/docs/en/build-with-claude/vision and add an entry "
            f"to SUPPORTED_MODEL_CAPS in lmms_eval/models/claude.py (the single source of truth)."
        )
    return SUPPORTED_MODEL_CAPS[max(matched, key=len)]  # longest-prefix match wins


def _floor_to_multiple_28(x: float) -> int:
    """Largest multiple of 28 that is <= x (min 28)."""
    return max(28, (int(x) // 28) * 28)


def anthropic_resized_hw(img_h: int, img_w: int, model_code: str) -> Tuple[int, int]:
    """
    Compute the on-grid image shape to send so it is a FIXED POINT of Claude's vision
    pipeline -- i.e. the canvas the model perceives equals the image we send and the size
    we state in the prompt.

    Why this matters: MedVision asks the model for RELATIVE coordinates in [0, 1]
    (coordinate / canvas dimension). The model normalizes by the canvas it actually
    perceives. Claude's pipeline (https://platform.claude.com/docs/en/build-with-claude/vision):
        - An image uses approximately width * height / 750 tokens.
        - The maximal native resolution is model-dependent (see SUPPORTED_MODEL_CAPS):
            - Claude Fable 5 / Opus 4.8 / Opus 4.7: 4784 tokens AND at most 2576 px long edge.
            - Opus 4.6 / 4.5, Sonnet 4.6 / 4.5, Haiku 4.5: 1568 tokens AND at most 1568 px long edge.
        - It resizes down (aspect-preserving) to fit both caps, THEN pads the bottom/right
          to a multiple of 28 px. That padding ENLARGES the perceived canvas, so it WOULD
          change the denominator of the model's relative coordinates (and the lower-left
          origin sits on the padded edge) -- corrupting every coordinate.

    To avoid the padding entirely we round each side DOWN to a multiple of 28. A 28-grid
    image makes Claude's resize AND pad steps no-ops, so perceived == sent == stated, the
    content fills the whole canvas, and relative coordinates align with the prompt. This is
    the same grid-alignment strategy used for Qwen2.5-VL. Flooring never upscales and stays
    within both caps. Per-axis sizes (independent of aspect ratio) are fine: the prompt's
    pixel size is adjusted per axis to conserve physical extent.

    Raises ValueError for unsupported models (via anthropic_image_caps).

    NOTE: this is the SINGLE SOURCE OF TRUTH for the Claude resize rule + caps. The task
    layer (lmms_eval/tasks/medvision/medvision_utils.py::get_resized_img_shape) imports and
    calls this function to set the image-size / pixel-size stated in TL/AD prompts, so the
    prompt and the actually-sent image can never diverge.
    """
    long_edge_cap, max_img_tokens = anthropic_image_caps(model_code)
    scale = min(
        1.0,
        long_edge_cap / max(img_h, img_w),
        math.sqrt(max_img_tokens * 750.0 / (img_h * img_w)),
    )
    return _floor_to_multiple_28(img_h * scale), _floor_to_multiple_28(img_w * scale)


def _giveup_on_bad_request(e: Exception) -> bool:
    # 400s are deterministic (invalid request); retrying them just wastes time.
    # Both the anthropic and openai SDKs expose status_code on their API errors.
    return getattr(e, "status_code", None) == 400


@register_model("claude")
class Claude(lmms):
    """
    Claude models via the Anthropic API or OpenRouter.

    Image processing pipeline
    -------------------------
    1. Input: doc_to_visual yields exactly one PIL image (optionally already reshaped
       to reshape_image_hw on the task side).
    2. Client-side pre-resize to a 28-px grid (see anthropic_resized_hw() above): the
       image is downscaled to fit the model's caps AND rounded down so both sides are
       multiples of 28. This makes Claude's own resize and bottom/right padding no-ops, so
       the canvas the model perceives equals the image we send (no padding):
           image tokens ~= width * height / 750
           caps: per SUPPORTED_MODEL_CAPS (high-res 4784 tokens / 2576 px for Fable 5,
                 Opus 4.8, Opus 4.7; standard 1568 tokens / 1568 px otherwise)
           scale = min(1.0, long_edge_cap / max(h, w), sqrt(max_tokens * 750 / (h * w)))
           new_h, new_w = floor_28(h * scale), floor_28(w * scale)
       Never upscales. Caps are looked up per model from the enumerated SUPPORTED_MODEL_CAPS
       table; an unsupported model raises (caps must be verified against the docs, since
       same-family models can differ). Ref: https://platform.claude.com/docs/en/build-with-claude/vision
    3. Encoding: PNG (lossless) -> base64, sent as the image content block before the
       text block in a single user message.
    4. Server-side: because the sent image is already <= native resolution and a multiple
       of 28 on each side, Claude neither resizes nor pads it. The content fills the whole
       perceived canvas, so the model's RELATIVE coordinates ([0, 1], normalized by the
       canvas) align with the image size stated in the prompt. (Were the image not on the
       28-grid, Claude would pad the bottom/right, enlarging the canvas and skewing every
       relative coordinate -- which is exactly what this grid alignment prevents.)
    5. Prompt consistency: get_resized_img_shape() in
       lmms_eval/tasks/medvision/medvision_utils.py imports and calls anthropic_resized_hw()
       (this same function) so the image size / pixel size stated in TL/AD task prompts matches
       the image the model actually sees -- one source of truth, no separate copy.

    Args:
        model:
            - provider="anthropic": Anthropic model ID, e.g. "claude-fable-5".
              Ref: https://platform.claude.com/docs/en/about-claude/models/overview
            - provider="openrouter": OpenRouter model ID, e.g. "anthropic/claude-opus-4.8".
              Ref: https://openrouter.ai/models
        provider: "anthropic" (direct Anthropic API, key from ANTHROPIC_API_KEY) or
            "openrouter" (OpenAI-compatible endpoint at https://openrouter.ai/api/v1,
            key from OPENROUTER_API_KEY).
        thinking: [bool] = True
            - If True, enable adaptive thinking (thinking={"type": "adaptive"}); via
              OpenRouter this maps to the unified reasoning parameter
              (https://openrouter.ai/docs/use-cases/reasoning-tokens).
            - If False, the thinking parameter is omitted entirely. Do NOT send an
              explicit {"type": "disabled"}: Claude Fable 5 rejects it with a 400.
            - Sampling parameters (temperature/top_p/top_k) are never sent; they are
              removed on Fable 5 / Opus 4.8 / Opus 4.7 (400 if sent).
        max_tokens: default max output tokens per request; a per-task max_new_tokens
            from the task YAML (gen_kwargs) takes precedence.
        stop_strings: optional stop sequences.
    """

    def __init__(
        self,
        model: str = "claude-fable-5",
        provider: str = "anthropic",
        thinking: Optional[bool] = True,
        max_tokens: Optional[int] = 16000,
        stop_strings: Optional[Union[List[str], str]] = None,
        **kwargs,  # absorbs model_hf / reshape_image_hw, which are consumed task-side via the evaluator
    ) -> None:
        super().__init__()
        self.model_code = model
        # Fail fast (at construction, before any task runs) if the model's image-resolution
        # caps are not verified -- otherwise TL/AD prompts could state a wrong pixel size.
        anthropic_image_caps(model)
        if provider not in ["anthropic", "openrouter"]:
            raise ValueError(f"Unsupported provider: {provider}. Use 'anthropic' or 'openrouter'.")
        self.provider = provider
        self.thinking = thinking
        self.max_tokens = int(max_tokens)
        self.stop_strings: List[str] = json.loads(stop_strings) if isinstance(stop_strings, str) else (stop_strings or [])
        self.prepare_model()

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def prepare_model(self):
        # Lazy-import per provider so only the required SDK must be installed.
        # NOTE: keys are stripped because pod/k8s-injected env secrets can carry a
        # trailing newline, which is an illegal HTTP header value.
        if self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
        else:
            import openai

            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            )

    def _encode_image(self, visual: Image.Image) -> str:
        """Pre-resize to the 28-px grid (so the stated image size matches what the model
        actually perceives, with no server-side padding), then base64-encode as PNG."""
        img_w, img_h = visual.size
        new_h, new_w = anthropic_resized_hw(img_h, img_w, self.model_code)
        if (new_h, new_w) != (img_h, img_w):
            visual = visual.resize((new_w, new_h), Image.LANCZOS)
        # Guard against formula regressions: the sent image must be on the 28-px grid (so
        # Claude adds no padding) and within both caps (so Claude does not resize it again).
        long_edge_cap, max_img_tokens = anthropic_image_caps(self.model_code)
        assert new_h % 28 == 0 and new_w % 28 == 0
        assert max(new_h, new_w) <= long_edge_cap and (new_h * new_w) / 750.0 <= max_img_tokens + 1
        buffer = io.BytesIO()
        visual.convert("RGB").save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    @backoff.on_exception(backoff.expo, Exception, max_tries=10, base=1.0, jitter=backoff.random_jitter, giveup=_giveup_on_bad_request)
    def _generate_content_with_retry(self, image_b64: str, contexts: str, max_tokens: int) -> str:
        if self.provider == "anthropic":
            request_kwargs = dict(
                model=self.model_code,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                            {"type": "text", "text": contexts},
                        ],
                    }
                ],
            )
            if self.thinking:
                request_kwargs["thinking"] = {"type": "adaptive"}
            # else: omit the thinking parameter entirely (Fable 5 rejects an explicit "disabled")
            if self.stop_strings:
                request_kwargs["stop_sequences"] = self.stop_strings
            response = self.client.messages.create(**request_kwargs)
            # Thinking blocks carry empty text by default (display defaults to "omitted")
            return "".join(block.text for block in response.content if block.type == "text")
        else:
            request_kwargs = dict(
                model=self.model_code,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            {"type": "text", "text": contexts},
                        ],
                    }
                ],
            )
            if self.thinking:
                # OpenRouter's unified reasoning parameter, mapped to Claude thinking
                request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
            if self.stop_strings:
                request_kwargs["stop"] = self.stop_strings
            response = self.client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Image inputs
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if len(visuals) == 1 and isinstance(visuals[0], Image.Image):
                visual = visuals[0]
            else:
                raise ValueError("We only support 1 image input for now and it should be of Image.Image type.")
            image_b64 = self._encode_image(visual)

            # Per-task max_new_tokens (from task YAML) takes precedence over the model default
            max_tokens = int(gen_kwargs.get("max_new_tokens", self.max_tokens)) if gen_kwargs else self.max_tokens

            # Get model response with retry mechanism
            # NOTE: no extra answer-format suffix is appended -- MedVision task prompts
            # already require the final values inside <answer></answer>, which parse_outputs parses.
            resp = self._generate_content_with_retry(image_b64, contexts, max_tokens)
            res.append(resp)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Claude")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Claude")
