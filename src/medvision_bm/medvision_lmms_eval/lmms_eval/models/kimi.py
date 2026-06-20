import base64
import io
import json
import math
import os
from typing import List, NamedTuple, Optional, Tuple, Union

import backoff
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Per-model image-processing rules for Moonshot's Kimi K2.x multimodal models.
#
# Source of the resize geometry (read verbatim from the open weights, NOT the API docs,
# which publish no internal resize math): MoonViT's `navit_resize_image` in
#   https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/media_utils.py
# parameterized by `media_proc_cfg` in
#   https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/preprocessor_config.json
# (in_patch_limit=16384, patch_size=14, merge_kernel_size=2, patch_limit_on_one_side=512),
# corroborated by the Kimi-VL technical report (arXiv:2504.07491): MoonViT is a NaViT-style
# *native-resolution* encoder (no fixed canvas, no sub-image tiling) that "authentically
# encodes up to 3.2 million pixels" (= in_patch_limit * patch_size^2 = 16384 * 196).
#
# WHY THIS IS AN EXPLICIT, ENUMERATED LIST (and not a generic default):
#   MedVision's quantitative tasks (Tumor/Lesion size, Angle/Distance) put the image size
#   and pixel size into the prompt, and the model must do the pixel->mm arithmetic itself.
#   Those numbers MUST match the resolution the model's vision encoder actually perceives
#   after its internal resize. MoonViT's budget is model-specific (Kimi-VL-A3B used a 4x
#   smaller in_token_limit=4096; K2.6 is 16384), so a silent fallback would emit a wrong
#   pixel size for an unverified model and corrupt every measurement -- an unrecognized
#   model RAISES. To add a model: confirm its `media_proc_cfg` in that model's
#   preprocessor_config.json and add an entry here. This is the single source of truth --
#   the task layer (medvision_utils.get_resized_img_shape) imports kimi_resized_hw() from
#   here, so there is no second table to update.
#
# Keys are matched EXACTLY after normalization (see _normalize_model_code: the OpenRouter
# "moonshotai/" prefix is stripped; dots are KEPT -- they are part of Moonshot ids, e.g.
# "kimi-k2.6"). Exact matching is deliberate: a sibling like "kimi-k2.5" or "kimi-k2.7-code"
# may carry a different MoonViT budget, so it must be verified and added explicitly rather
# than silently inheriting k2.6's caps.
#
# CAVEAT (assumption guarded empirically -- same posture as gemini.py's Gemini-3 geometry):
#   The geometry above is the OPEN-WEIGHTS local processor. The hosted endpoint
#   (api.moonshot.ai / OpenRouter) is ASSUMED to run the same MoonViT pipeline; the Moonshot
#   API docs publish no server-side resize math (only a soft "image resolution should not
#   exceed 4k" recommendation). This pass-through-fixed-point assumption is checked by
#   unit-test/kimi-image-resize/ (offline formula tests + an optional live coordinate-frame
#   probe). MedVision's largest slice (1935x2400) is far below the 7168 px / 16384-patch
#   caps, so in practice the dominant effect is the floor-to-28 grid alignment.
# ---------------------------------------------------------------------------


class MoonViTCaps(NamedTuple):
    """MoonViT media_proc_cfg parameters that drive navit_resize_image."""

    patch_size: int  # 14x14 px ViT patch
    merge_kernel_size: int  # 2 -> 2x2 pixel-shuffle in the patchmerger projector
    in_patch_limit: int  # max PRE-merge patches per image (the area budget)
    patch_limit_on_one_side: int  # max patches per side -> side px cap = this * patch_size


# Kimi K2.6 (model_type "kimi_k25", vision encoder MoonViT). Values verbatim from the model's
# preprocessor_config.json media_proc_cfg.
_K2_6 = MoonViTCaps(patch_size=14, merge_kernel_size=2, in_patch_limit=16384, patch_limit_on_one_side=512)
SUPPORTED_MODEL_CAPS = {
    "kimi-k2.6": _K2_6,
    # Add other Kimi multimodal ids (kimi-k2.5, kimi-k2.7-code, ...) only after confirming
    # their media_proc_cfg budget in that model's preprocessor_config.json -- unknown -> raise.
}


def _normalize_model_code(model_code: str) -> str:
    # OpenRouter model IDs look like "moonshotai/kimi-k2.6"; strip the provider prefix.
    # Dots are NOT replaced -- they are part of Moonshot ids ("kimi-k2.6").
    return model_code.split("/")[-1]


def kimi_image_caps(model_code: str) -> MoonViTCaps:
    """Return the MoonViTCaps for a Kimi model code.

    Raises ValueError for any model not in SUPPORTED_MODEL_CAPS: its MoonViT budget is
    unverified, so the image/pixel size stated in MedVision TL/AD prompts could be wrong.
    This is intentional -- a hard error beats silently corrupting measurements.
    """
    normalized = _normalize_model_code(model_code)
    if normalized not in SUPPORTED_MODEL_CAPS:
        raise ValueError(
            f"[kimi] Unsupported model code {model_code!r} (normalized {normalized!r}). "
            f"Its MoonViT image budget (in_patch_limit, patch_limit_on_one_side, patch_size, "
            f"merge_kernel_size) is not verified, so the image size / pixel size stated in "
            f"MedVision TL/AD prompts could be wrong. Read the model's preprocessor_config.json "
            f"media_proc_cfg (e.g. https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/preprocessor_config.json) "
            f"and add an entry to SUPPORTED_MODEL_CAPS in lmms_eval/models/kimi.py (the single source of truth)."
        )
    return SUPPORTED_MODEL_CAPS[normalized]


def _moonvit_navit_resize(img_h: int, img_w: int, caps: MoonViTCaps) -> Tuple[int, int, int, int, int]:
    """Faithful re-implementation of MoonViT's navit_resize_image (media_utils.py).

    Returns (new_h, new_w, pad_h, pad_w, num_tokens): the server downscales (only) so the
    pre-merge patch area fits in_patch_limit and each side fits patch_limit_on_one_side, then
    PADS each side UP to a multiple of factor = patch_size * merge_kernel_size (= 28). Used to
    GUARD that the image we pre-resize and send is a true fixed point (scale==1, pad==0).
    """
    P, M = caps.patch_size, caps.merge_kernel_size
    factor = P * M
    side_px_cap = caps.patch_limit_on_one_side * P
    # area scale to fit the pre-merge patch budget (integer floor div, as in the source)
    s1 = math.sqrt(caps.in_patch_limit / (max(1.0, img_w // P) * max(1.0, img_h // P)))
    scale = min(1.0, s1, side_px_cap / img_w, side_px_cap / img_h)  # downscale-only
    new_w = min(max(1, int(img_w * scale)), side_px_cap)  # int() = floor toward zero
    new_h = min(max(1, int(img_h * scale)), side_px_cap)
    pad_w = (factor - new_w % factor) % factor
    pad_h = (factor - new_h % factor) % factor
    num_tokens = ((new_h + pad_h) // factor) * ((new_w + pad_w) // factor)
    return new_h, new_w, pad_h, pad_w, num_tokens


def kimi_resized_hw(img_h: int, img_w: int, model_code: str) -> Tuple[int, int]:
    """
    Compute the image shape to send so it is a FIXED POINT of MoonViT's vision pipeline --
    i.e. the canvas the model perceives equals the image we send and the size we state in the
    prompt.

    Why this matters: MedVision asks the model for RELATIVE coordinates in [0, 1]
    (coordinate / canvas dimension), and TL/AD prompts state the image + pixel size for the
    pixel->mm arithmetic. The model normalizes by the canvas it actually perceives. MoonViT's
    navit_resize_image (https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/media_utils.py):
        - scale = min(1.0, sqrt(in_patch_limit / ((w//14)*(h//14))), 7168/w, 7168/h)  [downscale only]
        - new = floor(side * scale), clamped to <= 7168 px
        - THEN pads each side UP to a multiple of factor = patch_size*merge_kernel_size = 28.
      That bottom/right padding ENLARGES the perceived canvas, so it WOULD change the
      denominator of the model's relative coordinates -- corrupting every coordinate (exactly
      like Anthropic's bottom/right padding).

    To avoid the padding entirely we round each side DOWN to a multiple of 28 (same grid-
    alignment strategy as claude.py / Qwen2.5-VL). A 28-grid image that is within the patch
    budget makes MoonViT's resize AND pad steps both no-ops, so perceived == sent == stated.

    Subtlety: the budget uses integer floor division (w//14), so flooring to 28 can still
    leave a non-square image a few patches over in_patch_limit -> the server would re-downscale
    (breaking the fixed point). We trim the longer side by one 28-step until within budget.

    Never upscales (no min_pixels lift -- MoonViT has none). Per-axis sizes (independent of
    aspect ratio) are fine: the prompt's pixel size is adjusted per axis to conserve physical
    extent. Raises ValueError for unsupported models (via kimi_image_caps).

    NOTE: this is the SINGLE SOURCE OF TRUTH for the Kimi resize rule + caps. The task layer
    (lmms_eval/tasks/medvision/medvision_utils.py::get_resized_img_shape) imports and calls
    this function to set the image-size / pixel-size stated in TL/AD prompts, so the prompt
    and the actually-sent image can never diverge.
    """
    caps = kimi_image_caps(model_code)
    P, M = caps.patch_size, caps.merge_kernel_size
    factor = P * M  # 28
    side_px_cap = caps.patch_limit_on_one_side * P  # 7168
    scale = min(
        1.0,
        math.sqrt(caps.in_patch_limit / (max(1.0, img_w // P) * max(1.0, img_h // P))),
        side_px_cap / img_w,
        side_px_cap / img_h,
    )
    new_h = min(int(img_h * scale), side_px_cap)
    new_w = min(int(img_w * scale), side_px_cap)
    out_h = max(factor, (new_h // factor) * factor)  # floor DOWN to the 28-grid (kills server pad)
    out_w = max(factor, (new_w // factor) * factor)
    while (out_w // P) * (out_h // P) > caps.in_patch_limit and max(out_w, out_h) > factor:
        if out_w >= out_h:
            out_w -= factor
        else:
            out_h -= factor
    return out_h, out_w


def _giveup_on_bad_request(e: Exception) -> bool:
    # 400s are deterministic (invalid request); retrying them just wastes time.
    # The openai SDK exposes status_code on its API errors.
    return getattr(e, "status_code", None) == 400


@register_model("kimi")
class Kimi(lmms):
    """
    Moonshot Kimi K2.6 (multimodal) via the Moonshot Open Platform or OpenRouter.

    Both providers speak the OpenAI-compatible Chat Completions format, so a single request
    path serves both; only the base_url and API-key env var differ.

    Image processing pipeline
    -------------------------
    1. Input: doc_to_visual yields exactly one PIL image (optionally already reshaped to
       reshape_image_hw on the task side).
    2. Client-side pre-resize (see kimi_resized_hw() above) makes the sent image a fixed point
       of MoonViT's pipeline: downscaled to the in_patch_limit patch budget (and the 7168-px
       per-side cap), then floored to the 28-px grid so the server's resize AND pad-up steps
       are both no-ops and no edge padding extends the perceived canvas. Never upscales. Caps
       are looked up per model from the enumerated SUPPORTED_MODEL_CAPS table; an unsupported
       model raises (the MoonViT budget must be verified against the model's
       preprocessor_config.json, since different Kimi models differ).
    3. Encoding: PNG (lossless) -> base64 data URL (Moonshot's vision API accepts base64 data
       URLs / file ids only -- NOT remote http image URLs), sent as the image content block
       before the text block in a single user message.
    4. Server-side: because the sent image already satisfies MoonViT's caps and sits on the
       28-px grid, the server neither downscales nor pads it, so the model's RELATIVE
       coordinates ([0, 1], normalized by the canvas) align with the image size stated in the
       prompt.
    5. Prompt consistency: get_resized_img_shape() in
       lmms_eval/tasks/medvision/medvision_utils.py imports and calls kimi_resized_hw() (this
       same function) so the image size / pixel size stated in TL/AD task prompts matches the
       image the model actually sees -- one source of truth, no separate copy.

    Args:
        model:
            - provider="moonshot": Moonshot model ID, e.g. "kimi-k2.6".
              Ref: https://platform.moonshot.ai/docs
            - provider="openrouter": OpenRouter model ID, e.g. "moonshotai/kimi-k2.6".
              Ref: https://openrouter.ai/models
        provider: "moonshot" (Moonshot Open Platform, key from MOONSHOT_API_KEY, base_url
            https://api.moonshot.ai/v1 -- override with MOONSHOT_BASE_URL, e.g. the China
            endpoint https://api.moonshot.cn/v1) or "openrouter" (https://openrouter.ai/api/v1,
            key from OPENROUTER_API_KEY). Both use the Chat Completions request format.
        max_tokens: default max output tokens per request; a per-task max_new_tokens from the
            task YAML (gen_kwargs) takes precedence.
        stop_strings: optional stop sequences.
    """

    def __init__(
        self,
        model: str = "kimi-k2.6",
        provider: str = "moonshot",
        max_tokens: Optional[int] = 16000,
        stop_strings: Optional[Union[List[str], str]] = None,
        **kwargs,  # absorbs model_hf / reshape_image_hw, which are consumed task-side via the evaluator
    ) -> None:
        super().__init__()
        self.model_code = model
        # Fail fast (at construction, before any task runs) if the model's MoonViT budget is
        # not verified -- otherwise TL/AD prompts could state a wrong pixel size.
        kimi_image_caps(model)
        if provider not in ["moonshot", "openrouter"]:
            raise ValueError(f"Unsupported provider: {provider}. Use 'moonshot' or 'openrouter'.")
        self.provider = provider
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
        # NOTE: keys are stripped because pod/k8s-injected env secrets can carry a trailing
        # newline, which is an illegal HTTP header value.
        import openai

        if self.provider == "moonshot":
            base_url = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").strip()
            self.client = openai.OpenAI(base_url=base_url, api_key=os.environ["MOONSHOT_API_KEY"].strip())
        else:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            )

    def _encode_image(self, visual: Image.Image) -> str:
        """Pre-resize to a fixed point of MoonViT's vision pipeline (so the stated image size
        matches what the model perceives, with no server-side padding), then base64-encode as PNG."""
        img_w, img_h = visual.size
        new_h, new_w = kimi_resized_hw(img_h, img_w, self.model_code)
        if (new_h, new_w) != (img_h, img_w):
            visual = visual.resize((new_w, new_h), Image.LANCZOS)
        # Guard against formula regressions: the sent image must be a true fixed point of
        # MoonViT's navit_resize_image -- on the 28-px grid AND within the patch/side caps, so
        # the server applies scale==1.0 and pad==0 (perceived == sent == stated).
        caps = kimi_image_caps(self.model_code)
        factor = caps.patch_size * caps.merge_kernel_size
        sh, sw, pad_h, pad_w, _ = _moonvit_navit_resize(new_h, new_w, caps)
        assert new_h % factor == 0 and new_w % factor == 0
        assert (sh, sw, pad_h, pad_w) == (new_h, new_w, 0, 0)
        buffer = io.BytesIO()
        visual.convert("RGB").save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    @backoff.on_exception(backoff.expo, Exception, max_tries=10, base=1.0, jitter=backoff.random_jitter, giveup=_giveup_on_bad_request)
    def _generate_content_with_retry(self, image_b64: str, contexts: str, max_tokens: int) -> str:
        # One Chat Completions request path for both providers (both OpenAI-compatible).
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
        if self.stop_strings:
            request_kwargs["stop"] = self.stop_strings
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # resume: skip already-finished samples (greedy decoding only)
            _greedy = not (gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0)) if gen_kwargs else True
            _key = self._resp_cache_key(doc_id, task, split, contexts)
            if _greedy:
                _cached = self.resp_cache_get(task, _key)
                if _cached is not None:
                    res.append(_cached)
                    pbar.update(1)
                    continue

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
            # NOTE: no extra answer-format suffix is appended -- MedVision task prompts already
            # require the final values inside <answer></answer>, which parse_outputs parses.
            resp = self._generate_content_with_retry(image_b64, contexts, max_tokens)
            if _greedy:
                self.resp_cache_put(task, _key, resp)
            res.append(resp)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Kimi")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Kimi")
