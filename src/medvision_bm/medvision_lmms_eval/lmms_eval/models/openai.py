import base64
import io
import json
import math
import os
import re
from typing import List, Optional, Tuple, Union

import backoff
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Per-model image-processing rules, taken from the official OpenAI vision docs
# ("Calculating costs", verified verbatim 2026-06-12):
#   https://developers.openai.com/api/docs/guides/images-vision
#
# Unlike Anthropic (one rule for all Claude models), OpenAI has TWO rule families:
#
#   "patch" (gpt-5.5 / gpt-5.4 flagships, all mini/nano tiers, o4-mini):
#       image is covered by 32x32-px patches; tokens = ceil(w/32) * ceil(h/32).
#       If the patch count exceeds the model's budget, the image is downscaled by
#       shrink = sqrt(32^2 * budget / (w*h)) (plus a secondary adjustment that
#       32-aligns one side). An image WITHIN the budget is sent at native
#       resolution unchanged. Budgets at detail "high" (the detail we always send):
#         - gpt-5.5 / gpt-5.4: 2500 patches AND 2048 px max dimension
#         - mini / nano tiers and o4-mini: 1536 patches
#       ("original" detail -- 10000 patches / 6000 px -- exists on the flagships
#       only; the cost multipliers 1.62/2.46/1.72 affect billing, not resolution.)
#
#   "tile" (gpt-4.1, gpt-4o, o3, ...):
#       scale to fit in a 2048x2048 square (aspect-preserving), then scale so the
#       shortest side is 768 px, then count 512-px tiles.
#       The docs use neutral "scale" wording for the 768 step, but the live probe
#       (unit-test/openai-image-resize/check_openai_count_tokens.py, 2026-06-12 via
#       OpenRouter) settled it: a raw 512x512 image at detail "high" on gpt-4o cost
#       exactly 255 tokens (85 base + 170 x 1 tile), i.e. NOT upscaled to 768x768
#       (which would cost 765) -- so the min(1, ...) fixed point below holds.
#
#   Live-probe note (2026-06-12): gpt-5.5 billed image tokens at ~1.2x the patch
#   count (constant across square and non-square images => pure BILLING multiplier,
#   no geometry effect; the docs suggest 1.0 for the flagship). Cost-only -- the
#   perceived patch grid matches the prediction exactly, which is what matters here.
#
# WHY THIS IS AN EXPLICIT, ENUMERATED LIST (and not a generic default):
#   MedVision's quantitative tasks (Tumor/Lesion size, Angle/Distance) put the image
#   size and pixel size into the prompt, and the model must do the pixel->mm arithmetic
#   itself. Those numbers MUST match the resolution the model's vision encoder actually
#   perceives after its internal resize. Models in the SAME family differ here (flagship
#   2500-patch budget vs mini/nano 1536; patch vs tile rule). A silent fallback would
#   emit a wrong pixel size for an unverified model and corrupt every measurement -- so
#   an unrecognized model RAISES. To add a model: confirm its rule family + caps in the
#   docs above and add an entry here. This is the single source of truth -- the task
#   layer (medvision_utils.get_resized_img_shape) imports openai_resized_hw() from here,
#   so there is no second table to update.
#
# Keys are matched EXACTLY after normalization (see _normalize_model_code: the OpenRouter
# "openai/" prefix and a trailing "-YYYY-MM-DD" snapshot suffix are stripped; dots are
# kept -- they are part of OpenAI ids). Exact matching is deliberate: with prefix
# matching, an unverified sibling like "gpt-5.5-codex" or "gpt-4o-audio" would silently
# inherit "gpt-5.5"/"gpt-4o" caps (which may be wrong). NOTE: "gpt-5" base is intentionally
# NOT listed -- two doc-validation passes disagreed on whether it is tile- or patch-based,
# so rather than risk a wrong scale it is left out and any "gpt-5" request RAISES.
# ---------------------------------------------------------------------------
_PATCH = "patch"  # (family, patch_budget, max_dimension_px or None) at detail "high"
_TILE = "tile"  # (family, long_edge_cap_px, short_edge_cap_px) at detail "high"
SUPPORTED_MODEL_CAPS = {
    # Patch-based flagships (detail "high": 2500 patches / 2048 px max dimension)
    "gpt-5.5": (_PATCH, 2500, 2048),
    # gpt-5.5-pro: same vision family/caps as gpt-5.5 -- live-probed 2026-06-13 via
    # OpenRouter Chat Completions (a 512x512 image billed 308 image tokens, identical to
    # gpt-5.5 = 256 patches x the ~1.2 multiplier), and reachable via Chat Completions
    # (not Responses-API-locked).
    "gpt-5.5-pro": (_PATCH, 2500, 2048),
    "gpt-5.4": (_PATCH, 2500, 2048),
    # Patch-based mini/nano tiers (1536-patch budget; no documented px cap)
    "gpt-5.4-mini": (_PATCH, 1536, None),
    "gpt-5.4-nano": (_PATCH, 1536, None),
    "gpt-5-mini": (_PATCH, 1536, None),
    "gpt-5-nano": (_PATCH, 1536, None),
    "o4-mini": (_PATCH, 1536, None),
    # Tile-based (fit 2048x2048, shortest side 768, 512-px tiles).
    # NOTE: "gpt-5" base is intentionally omitted -- its family (tile vs patch) is
    # unconfirmed (conflicting doc reads), so it RAISES rather than risk a wrong scale.
    "gpt-4.1": (_TILE, 2048, 768),
    "gpt-4o": (_TILE, 2048, 768),
    "o3": (_TILE, 2048, 768),
}

# OpenAI snapshot ids append the release date, e.g. "gpt-5-2025-08-07".
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _normalize_model_code(model_code: str) -> str:
    # OpenRouter model IDs look like "openai/gpt-5.5"; strip the provider prefix and any
    # trailing date snapshot suffix. Dots are NOT replaced -- they are part of OpenAI ids.
    return _DATE_SUFFIX_RE.sub("", model_code.split("/")[-1])


def openai_image_caps(model_code: str) -> Tuple[str, int, Optional[int]]:
    """Return (rule_family, cap_a, cap_b) for an OpenAI model code.

    For "patch" models: ('patch', patch_budget, max_dimension_px or None).
    For "tile" models: ('tile', long_edge_cap_px, short_edge_cap_px).

    Raises ValueError for any model not in SUPPORTED_MODEL_CAPS: its image-processing
    rule is unverified, so the image/pixel size stated in MedVision TL/AD prompts could
    be wrong. This is intentional -- a hard error beats silently corrupting measurements.
    """
    normalized = _normalize_model_code(model_code)
    if normalized not in SUPPORTED_MODEL_CAPS:
        raise ValueError(
            f"[openai] Unsupported model code {model_code!r} (normalized {normalized!r}). "
            f"Its image-processing rule (patch vs tile family, budget/caps) is not verified, "
            f"so the image size / pixel size stated in MedVision TL/AD prompts could be wrong. "
            f"Look up the per-model rule at https://developers.openai.com/api/docs/guides/images-vision "
            f"and add an entry to SUPPORTED_MODEL_CAPS in lmms_eval/models/openai.py "
            f"(the single source of truth)."
        )
    return SUPPORTED_MODEL_CAPS[normalized]


def _floor_to_multiple_32(x: float) -> int:
    """Largest multiple of 32 that is <= x (min 32)."""
    return max(32, (int(x) // 32) * 32)


def openai_resized_hw(img_h: int, img_w: int, model_code: str) -> Tuple[int, int]:
    """
    Compute the image shape to send so it is a FIXED POINT of OpenAI's vision pipeline
    at detail "high" -- i.e. the canvas the model perceives equals the image we send and
    the size we state in the prompt.

    Why this matters: MedVision asks the model for RELATIVE coordinates in [0, 1]
    (coordinate / canvas dimension), and TL/AD prompts state the image + pixel size for
    the pixel->mm arithmetic. The model normalizes by the canvas it actually perceives.
    OpenAI's pipeline (https://developers.openai.com/api/docs/guides/images-vision):

    Patch family (gpt-5.5/5.4, mini/nano tiers, o4-mini):
        - tokens = ceil(w/32) * ceil(h/32); over-budget images are downscaled by
          sqrt(32^2 * budget / (w*h)); within-budget images are sent unchanged.
        - "A patch may extend beyond the image boundary": a non-32-aligned image is
          covered by overhanging edge patches, i.e. the patch grid the encoder sees is
          LARGER than the image -- which would skew relative coordinates exactly like
          Anthropic's bottom/right padding.
        We downscale with scale = min(1, sqrt(budget*32^2/(h*w)), max_dim/long_edge)
        and then floor each side to a multiple of 32. A 32-aligned, within-budget image
        is re-sent unchanged by the server and has no patch overhang, so
        perceived == sent == stated. Same grid-alignment strategy as Claude's floor-28.

    Tile family (gpt-4.1, gpt-4o, o3):
        - fit into 2048x2048, then shortest side to 768 px, then 512-px tiles.
        We apply scale = min(1, 2048/long_edge, 768/short_edge) so both server resize
        steps are no-ops. No grid floor: the docs document no padding to tile
        boundaries. Live-verified 2026-06-12 (check_openai_count_tokens.py): the 768
        step does NOT upscale (raw 512x512 on gpt-4o cost exactly 1 tile, not 4), so
        this fixed point holds.

    Flooring never upscales and stays within all caps. Per-axis sizes (independent of
    aspect ratio) are fine: the prompt's pixel size is adjusted per axis to conserve
    physical extent.

    Raises ValueError for unsupported models (via openai_image_caps).

    NOTE: this is the SINGLE SOURCE OF TRUTH for the OpenAI resize rule + caps. The task
    layer (lmms_eval/tasks/medvision/medvision_utils.py::get_resized_img_shape) imports
    and calls this function to set the image-size / pixel-size stated in TL/AD prompts,
    so the prompt and the actually-sent image can never diverge.
    """
    family, cap_a, cap_b = openai_image_caps(model_code)
    if family == _PATCH:
        patch_budget, max_dim = cap_a, cap_b
        scale = min(1.0, math.sqrt(patch_budget * 32.0 * 32.0 / (img_h * img_w)))
        if max_dim is not None:
            scale = min(scale, max_dim / max(img_h, img_w))
        return _floor_to_multiple_32(img_h * scale), _floor_to_multiple_32(img_w * scale)
    else:
        long_edge_cap, short_edge_cap = cap_a, cap_b
        scale = min(1.0, long_edge_cap / max(img_h, img_w), short_edge_cap / min(img_h, img_w))
        return max(1, int(img_h * scale)), max(1, int(img_w * scale))


def _giveup_on_bad_request(e: Exception) -> bool:
    # 400s are deterministic (invalid request); retrying them just wastes time.
    # The openai SDK exposes status_code on its API errors.
    return getattr(e, "status_code", None) == 400


@register_model("openai")
class OpenAI_GPT(lmms):
    """
    OpenAI GPT models via the official OpenAI API or OpenRouter.

    Image processing pipeline
    -------------------------
    1. Input: doc_to_visual yields exactly one PIL image (optionally already reshaped
       to reshape_image_hw on the task side).
    2. Client-side pre-resize (see openai_resized_hw() above) makes the sent image a
       fixed point of OpenAI's pipeline at detail "high":
         - patch-family models: downscale to the patch budget (and px cap where
           applicable), then floor each side to a 32-px multiple, so the server resends
           it unchanged and no edge patch extends beyond the image boundary;
         - tile-family models: scale = min(1, 2048/long_edge, 768/short_edge), so both
           server resize steps are no-ops (see the tile-family caveat in
           openai_resized_hw).
       Never upscales. Caps are looked up per model from the enumerated
       SUPPORTED_MODEL_CAPS table; an unsupported model raises (rules must be verified
       against the docs, since same-family models differ).
       Ref: https://developers.openai.com/api/docs/guides/images-vision
    3. Encoding: PNG (lossless) -> base64 data URL, sent as the image content block
       before the text block in a single user message, with detail "high" explicit.
       detail is pinned to "high" because the cap table encodes the high-detail
       budgets AND the server default differs per model ("auto" resolves to "original"
       on gpt-5.5 but to "high" on gpt-5.4) -- an implicit default would silently
       change the perceived resolution.
    4. Server-side: because the sent image already satisfies the model's caps, OpenAI
       does not resize it again, so the model's RELATIVE coordinates ([0, 1], normalized
       by the canvas) align with the image size stated in the prompt.
    5. Prompt consistency: get_resized_img_shape() in
       lmms_eval/tasks/medvision/medvision_utils.py imports and calls openai_resized_hw()
       (this same function) so the image size / pixel size stated in TL/AD task prompts
       matches the image the model actually sees -- one source of truth, no separate copy.

    Args:
        model:
            - provider="openai": OpenAI model ID, e.g. "gpt-5.5-pro".
              Ref: https://developers.openai.com/api/docs/models
            - provider="openrouter": OpenRouter model ID, e.g. "openai/gpt-5.5-pro".
              Ref: https://openrouter.ai/models
        provider: "openai" (official OpenAI API, key from OPENAI_API_KEY) or
            "openrouter" (OpenAI-compatible endpoint at https://openrouter.ai/api/v1,
            key from OPENROUTER_API_KEY). Both use the Chat Completions request format.
        reasoning_effort: [Optional[str]] = None
            - If set (e.g. "low" / "medium" / "high"), sent as reasoning_effort
              (official API) or extra_body={"reasoning": {"effort": ...}} (OpenRouter,
              https://openrouter.ai/docs/use-cases/reasoning-tokens).
            - If None, the parameter is omitted entirely (provider default applies).
        detail: image detail level; only "high" is supported -- SUPPORTED_MODEL_CAPS
            encodes the high-detail budgets, so any other level would desync the
            stated image/pixel size from the perceived canvas.
        max_tokens: default max output tokens per request; a per-task max_new_tokens
            from the task YAML (gen_kwargs) takes precedence. Sent as
            max_completion_tokens on the official API (gpt-5.x / o-series reject
            max_tokens) and as max_tokens via OpenRouter.
        stop_strings: optional stop sequences. NOTE: OpenAI reasoning models may reject
            the stop parameter (400); leave unset for gpt-5.x (no benchmark script
            uses it).
    """

    def __init__(
        self,
        model: str = "gpt-5.5-pro",
        provider: str = "openai",
        reasoning_effort: Optional[str] = None,
        detail: str = "high",
        max_tokens: Optional[int] = 16000,
        stop_strings: Optional[Union[List[str], str]] = None,
        **kwargs,  # absorbs model_hf / reshape_image_hw, which are consumed task-side via the evaluator
    ) -> None:
        super().__init__()
        self.model_code = model
        # Fail fast (at construction, before any task runs) if the model's image-processing
        # rule is not verified -- otherwise TL/AD prompts could state a wrong pixel size.
        openai_image_caps(model)
        if provider not in ["openai", "openrouter"]:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'openrouter'.")
        if detail != "high":
            raise ValueError(
                f"Unsupported detail level: {detail!r}. Only 'high' is supported: "
                f"SUPPORTED_MODEL_CAPS encodes the high-detail budgets, so other levels would "
                f"desync the image/pixel size stated in TL/AD prompts from the perceived canvas."
            )
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.detail = detail
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
        # Lazy import so the SDK is only required at eval time.
        # NOTE: keys are stripped because pod/k8s-injected env secrets can carry a
        # trailing newline, which is an illegal HTTP header value.
        import openai

        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        else:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            )

    def _encode_image(self, visual: Image.Image) -> str:
        """Pre-resize to a fixed point of the model's vision pipeline (so the stated image
        size matches what the model actually perceives), then base64-encode as PNG."""
        img_w, img_h = visual.size
        new_h, new_w = openai_resized_hw(img_h, img_w, self.model_code)
        if (new_h, new_w) != (img_h, img_w):
            visual = visual.resize((new_w, new_h), Image.LANCZOS)
        # Guard against formula regressions: the sent image must satisfy the model's caps
        # (so OpenAI does not resize it again) and, for patch models, sit on the 32-px grid
        # (so no edge patch extends beyond the image boundary).
        family, cap_a, cap_b = openai_image_caps(self.model_code)
        if family == _PATCH:
            assert new_h % 32 == 0 and new_w % 32 == 0
            assert (new_h // 32) * (new_w // 32) <= cap_a
            assert cap_b is None or max(new_h, new_w) <= cap_b
        else:
            assert max(new_h, new_w) <= cap_a and min(new_h, new_w) <= cap_b
        buffer = io.BytesIO()
        visual.convert("RGB").save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    @backoff.on_exception(backoff.expo, Exception, max_tries=10, base=1.0, jitter=backoff.random_jitter, giveup=_giveup_on_bad_request)
    def _generate_content_with_retry(self, image_b64: str, contexts: str, max_tokens: int) -> str:
        # One Chat Completions request path for both providers (OpenRouter is
        # OpenAI-compatible); only the token-limit and reasoning parameters differ.
        request_kwargs = dict(
            model=self.model_code,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": self.detail}},
                        {"type": "text", "text": contexts},
                    ],
                }
            ],
        )
        if self.provider == "openai":
            # gpt-5.x / o-series reject max_tokens on the official API
            request_kwargs["max_completion_tokens"] = max_tokens
            if self.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            request_kwargs["max_tokens"] = max_tokens
            if self.reasoning_effort is not None:
                # OpenRouter's unified reasoning parameter
                request_kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
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
        raise NotImplementedError("Loglikelihood is not implemented for OpenAI models")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for OpenAI models")
