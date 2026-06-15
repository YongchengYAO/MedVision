import base64
import io
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import backoff
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Per-model image-processing capabilities, from the official Google docs:
#   - image understanding (tiling rule, pre-Gemini-3): https://ai.google.dev/gemini-api/docs/image-understanding
#   - media resolution (Gemini 3 token budgets):       https://ai.google.dev/gemini-api/docs/media-resolution
#   - input scaling limits (3072x3072 downscale):      https://firebase.google.com/docs/ai-logic/input-file-requirements
#   - model codes:                                     https://ai.google.dev/gemini-api/docs/models
#
# WHY THIS IS AN EXPLICIT, ENUMERATED LIST (and not a generic default):
#   MedVision's quantitative tasks (Tumor/Lesion size, Angle/Distance) put the image
#   size and pixel size into the prompt, and the model must do the pixel->mm arithmetic
#   itself. Those numbers MUST match the canvas the model actually perceives. The two
#   Gemini series process images differently:
#     - "2.5" series: pan-&-scan tiling (crop unit clamp(floor(min(w,h)/1.5), 256, 768),
#       each tile resampled to 768x768, 258 tokens/tile; both dims <= 384 -> single unit)
#       plus a global low-res view. Crop-based detail sampling preserves the global frame.
#     - "3" series: fixed per-image token budgets selected by media_resolution
#       (LOW 280 / MEDIUM 560 / HIGH 1120 = default; the pixel geometry is undocumented,
#       so we pin media_resolution="high" for reproducibility and guard the pass-through
#       assumption empirically -- see unit-test/gemini-image-resize/).
#   A silent fallback would emit a wrong pixel size for an unverified model and corrupt
#   every measurement -- so an unrecognized model RAISES. To add a model: confirm its
#   series + image handling in the docs above and add an entry here. This is the single
#   source of truth -- the task layer (medvision_utils.get_resized_img_shape) imports
#   gemini_resized_hw() from here, so there is no second table to update.
#
# NOTE: "gemini-3-pro-preview" was shut down on 2026-03-09 (replaced by
# "gemini-3.1-pro-preview"); do not add it back. There was never a stable "gemini-3-pro".
#
# Keys are matched as PREFIXES of the normalized model code (see _normalize_model_code):
# the leading "google/" (OpenRouter) is stripped, so "gemini-2.5-pro" (Google direct) and
# "google/gemini-2.5-pro" (OpenRouter), plus dated suffixes like
# "gemini-2.5-flash-preview-09-2025", all resolve to the same entry. The LONGEST matching
# prefix wins (so "gemini-2.5-flash-lite" is not shadowed by "gemini-2.5-flash").
# ---------------------------------------------------------------------------
# Long-edge cap: images larger than 3072x3072 are scaled down (and padded) server-side;
# the client-side pre-resize in gemini_resized_hw() keeps that path unreachable.
_LONG_EDGE_CAP = 3072
SUPPORTED_MODEL_CAPS = {
    # Gemini 2.5 series (stable): pan-&-scan tiling; media_resolution must stay UNSET
    "gemini-2.5-pro": {"series": "2.5", "long_edge_cap": _LONG_EDGE_CAP},
    "gemini-2.5-flash": {"series": "2.5", "long_edge_cap": _LONG_EDGE_CAP},
    "gemini-2.5-flash-lite": {"series": "2.5", "long_edge_cap": _LONG_EDGE_CAP},
    # Gemini 3.x series: media_resolution token budgets; pinned to "high" by default
    "gemini-3.1-pro-preview": {"series": "3", "long_edge_cap": _LONG_EDGE_CAP},
    "gemini-3-flash-preview": {"series": "3", "long_edge_cap": _LONG_EDGE_CAP},
    "gemini-3.1-flash-lite": {"series": "3", "long_edge_cap": _LONG_EDGE_CAP},
    "gemini-3.5-flash": {"series": "3", "long_edge_cap": _LONG_EDGE_CAP},
}

# media_resolution levels supported via the global generationConfig (v1beta).
# MEDIA_RESOLUTION_ULTRA_HIGH (2240 tokens) is per-part/v1alpha-only -- deliberately excluded.
_MEDIA_RESOLUTION_LEVELS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}

_THINKING_LEVELS = ["minimal", "low", "medium", "high"]


def _normalize_model_code(model_code: str) -> str:
    # OpenRouter model IDs look like "google/gemini-2.5-pro"; normalize to the bare
    # Google form ("gemini-2.5-pro") for capability matching. Google codes natively
    # contain dots ("2.5", "3.1"), so unlike Claude no "." -> "-" mapping is applied.
    return model_code.split("/")[-1]


def gemini_image_caps(model_code: str) -> Dict:
    """Return the capability entry ({"series", "long_edge_cap"}) for a Gemini model code.

    Raises ValueError for any model not in SUPPORTED_MODEL_CAPS: its image handling is
    unverified, so the image/pixel size stated in MedVision TL/AD prompts could be wrong.
    This is intentional -- a hard error beats silently corrupting measurements.
    """
    normalized = _normalize_model_code(model_code)
    matched = [prefix for prefix in SUPPORTED_MODEL_CAPS if normalized.startswith(prefix)]
    if not matched:
        raise ValueError(
            f"[gemini] Unsupported model code {model_code!r} (normalized {normalized!r}). "
            f"Its image handling is not verified, so the image size / pixel size stated in "
            f"MedVision TL/AD prompts could be wrong. Check the per-model image processing at "
            f"https://ai.google.dev/gemini-api/docs/image-understanding and "
            f"https://ai.google.dev/gemini-api/docs/media-resolution, then add an entry to "
            f"SUPPORTED_MODEL_CAPS in lmms_eval/models/gemini.py (the single source of truth). "
            f"Note: 'gemini-3-pro-preview' was retired 2026-03-09 -- use 'gemini-3.1-pro-preview'."
        )
    return SUPPORTED_MODEL_CAPS[max(matched, key=len)]  # longest-prefix match wins


def gemini_resized_hw(img_h: int, img_w: int, model_code: str) -> Tuple[int, int]:
    """
    Compute the image shape to send to Gemini so the stated size equals the canvas the
    model perceives. For Gemini this is PASS-THROUGH (identity) with a single guard:
    only images whose long edge exceeds 3072 px are downscaled (aspect-preserving).

    Why pass-through is correct for Gemini (unlike Claude's floor-28 fixed point):
        - Claude's server PADS the bottom/right of images below its caps, ENLARGING the
          canvas the model normalizes relative coordinates against -- hence Claude needs
          a client pre-resize that makes the pad a no-op.
        - Gemini does NOT pad the canvas for images <= 3072 px. Its pipeline is
          crop-based detail sampling: ~768x768 tiles + a global low-res view ("2.5"
          series) or a fixed token-budget resample ("3" series). The documented spatial
          contract (box_2d in [0,1000] normalized to the INPUT image dimensions) anchors
          all coordinates to the sent canvas, and the google-genai SDK never resizes
          client-side. So sent size == perceived canvas == stated size, and the pixel
          sizes in TL/AD prompts stay valid at resize ratio 1.0.
        - The only documented destructive server op -- "larger images are scaled down
          and padded to fit a maximum resolution of 3072 x 3072"
          (https://firebase.google.com/docs/ai-logic/input-file-requirements) -- applies
          to >3072 px images only; the guard here pre-empts it, so the ambiguous server
          "pad" can never fire. No MedVision slice (max 1935x2400) triggers the guard.
        - Gemini "3" series caveat: the pixel geometry behind the media_resolution token
          budgets is undocumented. The coordinate contract is unchanged in the Gemini 3
          docs, and unit-test/gemini-image-resize/check_gemini_coordinate_frame.py guards
          the pass-through assumption empirically (markers at known relative positions on
          non-square images).

    Never upscales. Raises ValueError for unsupported models (via gemini_image_caps).

    NOTE: this is the SINGLE SOURCE OF TRUTH for the Gemini resize rule + supported-model
    table. The task layer (lmms_eval/tasks/medvision/medvision_utils.py::get_resized_img_shape)
    imports and calls this function to set the image-size / pixel-size stated in TL/AD
    prompts, so the prompt and the actually-sent image can never diverge.
    """
    long_edge_cap = gemini_image_caps(model_code)["long_edge_cap"]
    long_edge = max(img_h, img_w)
    if long_edge <= long_edge_cap:
        return int(img_h), int(img_w)
    scale = long_edge_cap / long_edge
    return max(1, int(img_h * scale)), max(1, int(img_w * scale))


def _giveup_on_bad_request(e: Exception) -> bool:
    # 400s are deterministic (invalid request); retrying them just wastes time.
    # The openai SDK exposes status_code on its API errors; the google-genai SDK
    # exposes code on APIError.
    return getattr(e, "status_code", None) == 400 or getattr(e, "code", None) == 400


@register_model("gemini")
class Gemini(lmms):
    """
    Gemini models (2.5 and 3 series) via the Google Gemini API or OpenRouter.

    Replaces the former gemini__2_5 / gemini__2_5_woTool Detection-only wrappers (removed).

    Image processing pipeline
    -------------------------
    1. Input: doc_to_visual yields exactly one PIL image (optionally already reshaped
       to reshape_image_hw on the task side).
    2. Client-side pre-resize (see gemini_resized_hw() above): PASS-THROUGH for all
       MedVision inputs; only >3072-px long edges are downscaled (aspect-preserving) so
       the server's own ">3072 downscale + pad" path can never fire. Never upscales.
       Caps/series are looked up per model from the enumerated SUPPORTED_MODEL_CAPS
       table; an unsupported model raises (image handling must be verified first).
    3. Encoding: provider "google" sends the (resized) PIL image to the SDK (which
       re-encodes losslessly without changing dimensions); provider "openrouter" sends
       a base64 PNG data URL before the text block in a single user message.
    4. Server-side: no global rescale or canvas padding below 3072 px. "2.5" series
       tiles the image (~768x768 crops + global view); "3" series resamples to a fixed
       media_resolution token budget. Both are frame-preserving: spatial outputs are
       documented as normalized to the INPUT image dims, so the model's relative
       coordinates and pixel-size arithmetic align with the size stated in the prompt.
    5. Prompt consistency: get_resized_img_shape() in
       lmms_eval/tasks/medvision/medvision_utils.py imports and calls gemini_resized_hw()
       (this same function) so the image size / pixel size stated in TL/AD task prompts
       matches the image the model actually sees -- one source of truth, no separate copy.

    Args:
        model:
            - provider="google": Gemini model code, e.g. "gemini-3.1-pro-preview".
              Ref: https://ai.google.dev/gemini-api/docs/models
            - provider="openrouter": OpenRouter model ID, e.g. "google/gemini-3.1-pro-preview".
              Ref: https://openrouter.ai/models
        provider: "google" (direct Gemini API, key from GEMINI_API_KEY or GOOGLE_API_KEY)
            or "openrouter" (OpenAI-compatible endpoint at https://openrouter.ai/api/v1,
            key from OPENROUTER_API_KEY).
        use_tool: [bool] = False
            - Code execution (https://ai.google.dev/gemini-api/docs/code-execution).
            - provider="google" only; raises with "openrouter".
        json_output: [bool] = False
            - Structured output (https://ai.google.dev/gemini-api/docs/structured-output).
            - provider="google" only; raises with "openrouter". Not combinable with
              use_tool on the 2.5 series. Default False = plain text: MedVision task
              prompts already require the final values inside <answer></answer>, which
              parse_outputs parses (no extra answer-format suffix is appended).
        json_fields: List[str], str, Dict[str, str] -- fields for the structured JSON
            output (must contain "Answer"); only used when json_output=True.
        ignore_thoughts: bool=False -- when json_output=True, return only the "Answer"
            field instead of all fields.
        thinking_level: Optional[str] = None
            - Gemini 3 series only ("minimal" | "low" | "medium" | "high").
            - None: the parameter is omitted (model default; "high" on 3.1 Pro).
              Thinking cannot be disabled on Gemini 3.1 Pro.
            - Never sent together with a thinking budget (the API returns 400).
            - Ref: https://ai.google.dev/gemini-api/docs/thinking
        thinkingBudget: Optional[int] = None
            - Gemini 2.5 series only. None -> -1 (dynamic thinking, the prior default).
              0 disables thinking (2.5 Flash / Flash-Lite only; 2.5 Pro cannot disable).
            - Raises if set for a Gemini 3 code (use thinking_level instead).
        media_resolution: Optional[str] = None
            - provider="google" only ("low" | "medium" | "high").
            - None: pinned to "high" for BOTH series. This is load-bearing -- verified
              against the live API (2026-06-12): with the google-genai SDK default UNSET,
              Gemini 2.5 returns a single ~258-token global thumbnail for every size (no
              pan-&-scan), crushing small-structure detail; only HIGH triggers full
              tiling. For Gemini 3 the budget is fixed (~1120) and default already equals
              HIGH; pinning keeps results reproducible. LOW/MEDIUM are an explicit opt-in
              that collapses resolution. Raises with provider="openrouter" (not
              controllable there; the Google-side default applies).
            - Ref: https://ai.google.dev/gemini-api/docs/media-resolution
        max_tokens: default max output tokens per request; a per-task max_new_tokens
            from the task YAML (gen_kwargs) takes precedence.
        stop_strings: optional stop sequences.
    """

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        provider: str = "google",
        use_tool: Optional[bool] = False,
        json_output: Optional[bool] = False,
        json_fields: Optional[Union[List[str], str, Dict[str, str]]] = ["Thought", "Answer"],
        ignore_thoughts: Optional[bool] = False,
        ignore_code: Optional[bool] = True,
        thinking_level: Optional[str] = None,
        thinkingBudget: Optional[int] = None,
        media_resolution: Optional[str] = None,
        max_tokens: Optional[int] = 16000,
        stop_strings: Optional[Union[List[str], str]] = None,
        **kwargs,  # absorbs model_hf / reshape_image_hw, which are consumed task-side via the evaluator
    ) -> None:
        super().__init__()
        self.model_code = model
        # Fail fast (at construction, before any task runs) if the model's image handling
        # is not verified -- otherwise TL/AD prompts could state a wrong pixel size.
        self.series = gemini_image_caps(model)["series"]

        if provider not in ["google", "openrouter"]:
            raise ValueError(f"Unsupported provider: {provider}. Use 'google' or 'openrouter'.")
        self.provider = provider

        # Feature toggles: code execution and structured output exist only on the Google SDK.
        if provider == "openrouter" and use_tool:
            raise ValueError("[gemini] use_tool (code execution) is only available with provider='google'.")
        if provider == "openrouter" and json_output:
            raise ValueError("[gemini] json_output (structured output) is only available with provider='google'.")
        self.use_tool = use_tool
        self.json_output = json_output
        self.ignore_code = ignore_code

        # Thinking: the two series use mutually exclusive parameters (mixing them -> 400).
        if thinking_level is not None and self.series != "3":
            raise ValueError(f"[gemini] thinking_level is a Gemini 3 parameter; use thinkingBudget for {model!r} (2.5 series).")
        if thinking_level is not None and thinking_level not in _THINKING_LEVELS:
            raise ValueError(f"[gemini] Invalid thinking_level {thinking_level!r}. Use one of {_THINKING_LEVELS}.")
        if thinkingBudget is not None and self.series != "2.5":
            raise ValueError(f"[gemini] thinkingBudget is a Gemini 2.5 parameter; use thinking_level for {model!r} (3 series).")
        self.thinking_level = thinking_level
        self.thinkingBudget = (-1 if thinkingBudget is None else int(thinkingBudget)) if self.series == "2.5" else None

        # media_resolution: pinned "high" by default for BOTH series on the google provider.
        # This is load-bearing and was verified against the live API (2026-06-12): with the
        # google-genai SDK (>=2.8.0) default UNSET, Gemini 2.5 returns a single ~258-token
        # global thumbnail for every input size (NO pan-&-scan), which would crush all
        # small-structure detail; only HIGH triggers full tiling (e.g. 1935x2400 -> 3354
        # tokens = 12 tiles + global view). For Gemini 3 the budget is fixed (~1120) and
        # default already equals HIGH; pinning keeps results reproducible if Google changes
        # defaults. LOW/MEDIUM are allowed as an explicit opt-in but collapse resolution.
        # Not controllable via OpenRouter (the Google-side default applies there).
        if media_resolution is not None:
            if provider == "openrouter":
                raise ValueError("[gemini] media_resolution cannot be set via OpenRouter (Google-side default applies).")
            if media_resolution not in _MEDIA_RESOLUTION_LEVELS:
                raise ValueError(f"[gemini] Invalid media_resolution {media_resolution!r}. Use one of {list(_MEDIA_RESOLUTION_LEVELS)}.")
        self.media_resolution = (media_resolution or "high") if provider == "google" else None

        # JSON fields (structured output only): must contain "Answer" (used when formatting the reply)
        if self.json_output:
            if isinstance(json_fields, str):
                import ast

                try:
                    json_fields = ast.literal_eval(json_fields)
                except (ValueError, SyntaxError):
                    raise ValueError("Failed to parse string for argument 'json_fields'")
            if isinstance(json_fields, list):
                json_fields = {field: str for field in json_fields}
            if "Answer" not in json_fields:
                raise ValueError("The 'Answer' key must be present in json_fields")
            self.json_fields = json_fields
        self.ignore_thoughts = ignore_thoughts

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
        # Lazy-import per provider so only the required SDK must be installed (and so
        # gemini_resized_hw stays importable without any SDK, e.g. from medvision_utils).
        # NOTE: keys are stripped because pod/k8s-injected env secrets can carry a
        # trailing newline, which is an illegal HTTP header value.
        if self.provider == "google":
            from google import genai

            api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
            if not api_key:
                raise EnvironmentError("[gemini] Set GEMINI_API_KEY (or GOOGLE_API_KEY) for provider='google'.")
            self.client = genai.Client(api_key=api_key)
            if self.json_output:
                from pydantic import Field, create_model

                field_definitions = {name: (ftype, Field(...)) for name, ftype in self.json_fields.items()}
                self._json_schema = create_model("JSON_format", **field_definitions)
        else:
            import openai

            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            )

    def _resize_image(self, visual: Image.Image) -> Image.Image:
        """Apply the pass-through + 3072-cap rule (so the stated image size matches the
        canvas the model perceives); a no-op for all MedVision inputs."""
        img_w, img_h = visual.size
        new_h, new_w = gemini_resized_hw(img_h, img_w, self.model_code)
        if (new_h, new_w) != (img_h, img_w):
            visual = visual.resize((new_w, new_h), Image.LANCZOS)
        # Guard against formula regressions: the sent image must be within the server's
        # downscale threshold, so Gemini never rescales or pads the canvas.
        assert max(new_h, new_w) <= gemini_image_caps(self.model_code)["long_edge_cap"]
        return visual

    def _encode_image_b64(self, visual: Image.Image) -> str:
        buffer = io.BytesIO()
        visual.convert("RGB").save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _build_google_config(self, max_tokens: int):
        from google.genai import types

        if self.series == "3":
            # Gemini 3: thinking_level only (mixing with thinking_budget -> 400);
            # omit when unset (model default; cannot be disabled on 3.1 Pro).
            thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level is not None else None
        else:
            thinking_config = types.ThinkingConfig(thinking_budget=self.thinkingBudget)
        config_kwargs = dict(
            thinking_config=thinking_config,
            response_mime_type="application/json" if self.json_output else "text/plain",
            response_schema=self._json_schema if self.json_output else None,
            stop_sequences=self.stop_strings if self.stop_strings else None,
            max_output_tokens=max_tokens,
        )
        if self.use_tool:
            config_kwargs["tools"] = [types.Tool(code_execution=types.ToolCodeExecution)]
        if self.media_resolution is not None:
            config_kwargs["media_resolution"] = _MEDIA_RESOLUTION_LEVELS[self.media_resolution]
        return types.GenerateContentConfig(**config_kwargs)

    @backoff.on_exception(backoff.expo, Exception, max_tries=10, base=1.0, jitter=backoff.random_jitter, giveup=_giveup_on_bad_request)
    def _generate_content_with_retry(self, visual: Image.Image, contexts: str, max_tokens: int) -> str:
        if self.provider == "google":
            response = self.client.models.generate_content(
                model=self.model_code,
                contents=[visual, contexts],
                config=self._build_google_config(max_tokens),
            )
            if self.json_output:
                resp_json = response.parsed
                if self.ignore_thoughts:
                    return getattr(resp_json, "Answer")
                final_answer = ""
                for key in [k for k in self.json_fields.keys() if k != "Answer"]:
                    final_answer += f"{key}\n: {getattr(resp_json, key, '<not provided>')}\n"
                final_answer += f"Answer\n: {getattr(resp_json, 'Answer', '<not provided>')}\n"
                return final_answer
            elif self.use_tool:
                # ref: https://ai.google.dev/gemini-api/docs/code-execution
                final_answer = ""
                part_counter_text = 0
                part_counter_code = 0
                part_counter_code_result = 0
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        part_counter_text += 1
                        final_answer += f"Text Block {part_counter_text}:\n{part.text}\n"
                    if part.executable_code is not None:
                        part_counter_code += 1
                        if not self.ignore_code:
                            final_answer += f"Code Block {part_counter_code}:\n{part.executable_code.code}\n"
                        else:
                            final_answer += f"Code Block {part_counter_code}:\n<code block ignored>\n"
                    if part.code_execution_result is not None:
                        part_counter_code_result += 1
                        final_answer += f"Code Result Block {part_counter_code_result}:\n{part.code_execution_result.output}\n"
                return final_answer
            else:
                return response.text if response.text is not None else ""
        else:
            image_b64 = self._encode_image_b64(visual)
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
            # OpenRouter's unified reasoning parameter (https://openrouter.ai/docs/use-cases/reasoning-tokens):
            # Gemini 3 -> reasoning.effort maps 1:1 to thinkingLevel; Gemini 2.5 -> reasoning.max_tokens
            # passes through as thinkingBudget (-1 dynamic -> just "enabled"; 0 -> omitted = off).
            if self.series == "3":
                if self.thinking_level is not None:
                    request_kwargs["extra_body"] = {"reasoning": {"effort": self.thinking_level}}
            else:
                if self.thinkingBudget == -1:
                    request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
                elif self.thinkingBudget > 0:
                    request_kwargs["extra_body"] = {"reasoning": {"max_tokens": self.thinkingBudget}}
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
            visual = self._resize_image(visual)

            # Per-task max_new_tokens (from task YAML) takes precedence over the model default
            max_tokens = int(gen_kwargs.get("max_new_tokens", self.max_tokens)) if gen_kwargs else self.max_tokens

            # Get model response with retry mechanism
            # NOTE: no extra answer-format suffix is appended -- MedVision task prompts
            # already require the final values inside <answer></answer>, which parse_outputs parses.
            resp = self._generate_content_with_retry(visual, contexts, max_tokens)
            if _greedy:
                self.resp_cache_put(task, _key, resp)
            res.append(resp)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Gemini")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Gemini")
