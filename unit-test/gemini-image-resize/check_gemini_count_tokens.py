"""Empirical check: verify the Gemini image-tokenization model against the live API.

The pass-through rule (see gemini_resized_hw() in lmms_eval/models/gemini.py) rests on the
documented server-side image pipeline; this script measures each image's real token cost and
compares it to the documented prediction. Image tokens are isolated by subtracting a
text-only baseline.

This pins media_resolution="high" for the google path, exactly as lmms_eval/models/gemini.py
does -- WITHOUT it the SDK default returns a single ~258-token thumbnail for any 2.5 input
(verified 2026-06-12), which is why the model file pins "high".

Expectations (at media_resolution="high"):
    - Gemini 2.5 series: tokens ~= 258 * tiles + 258 (a global low-res view is added when
      the image is tiled), where both dims <= 384 -> a single 258-token unit (no global
      view), else crop_unit = clamp(floor(min(w, h) / 1.5), 256, 768);
      tiles = ceil(w/cu) * ceil(h/cu). E.g. 1935x2400 -> 12 tiles + global = 3354.
      Ref: https://ai.google.dev/gemini-api/docs/image-understanding
    - Gemini 3 series: a fixed budget independent of input size (~1120 tokens at "high").
      Ref: https://ai.google.dev/gemini-api/docs/media-resolution
    A token count matching a CONSTANT 16-tile grid for every size would instead indicate a
    pad-to-3072-canvas behavior (refuting the pass-through assumption) -- not expected.

Usage:
    python check_gemini_count_tokens.py                                  # google (GEMINI_API_KEY / GOOGLE_API_KEY)
    python check_gemini_count_tokens.py --model gemini-3.1-pro-preview
    python check_gemini_count_tokens.py --provider openrouter --model google/gemini-2.5-pro  # OPENROUTER_API_KEY

Provider notes:
    - google: uses the free models.count_tokens endpoint (no generation).
    - openrouter: has no count_tokens endpoint; makes a 1-output-token chat completion per
      image and reads usage.prompt_tokens (negligible cost).
"""

import argparse
import base64
import io
import math
import os
import sys

from PIL import Image

# Make the vendored lmms_eval importable as a top-level package, then import the shipped
# functions directly (run in an env with the eval deps, e.g. the `eval-gemini` env).
# __file__ is unit-test/gemini-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))
from lmms_eval.models.gemini import gemini_image_caps, gemini_resized_hw  # noqa: E402

PROMPT = "Describe this image in one word."
# (h, w): MedVision-typical sizes + tiling edge cases
TEST_SIZES = [(256, 256), (384, 384), (512, 512), (540, 960), (1024, 1024), (1935, 2400)]


def predicted_image_tokens(h, w, model_code):
    series = gemini_image_caps(model_code)["series"]
    if series == "3":
        return 1120  # fixed budget at "high" media resolution
    if h <= 384 and w <= 384:
        return 258  # single unit, no separate global view
    crop_unit = min(768, max(256, int(min(w, h) / 1.5)))
    tiles = math.ceil(w / crop_unit) * math.ceil(h / crop_unit)
    return 258 * tiles + 258  # tiles + one global low-res view


def make_image(h, w, model_code):
    img = Image.effect_noise((w, h), 64).convert("RGB")  # noise: incompressible, realistic
    nh, nw = gemini_resized_hw(h, w, model_code)
    if (nh, nw) != (h, w):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def count_google(model_code):
    from google import genai
    from google.genai import types

    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    client = genai.Client(api_key=api_key)

    # count_tokens ignores generation config, so it cannot reflect media_resolution=high.
    # Use a 1-thinking-token generation and read usage_metadata.prompt_token_count, with
    # media_resolution="high" pinned exactly as lmms_eval/models/gemini.py does.
    series = gemini_image_caps(model_code)["series"]
    base_cfg = dict(max_output_tokens=64, media_resolution="MEDIA_RESOLUTION_HIGH")
    if series == "3":
        base_cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    def prompt_tokens(contents):
        resp = client.models.generate_content(model=model_code, contents=contents, config=types.GenerateContentConfig(**base_cfg))
        return resp.usage_metadata.prompt_token_count

    baseline = prompt_tokens([PROMPT])
    print(f"text-only baseline: {baseline} tokens (media_resolution=high)\n")
    for h, w in TEST_SIZES:
        img = make_image(h, w, model_code)
        report(h, w, prompt_tokens([img, PROMPT]) - baseline, model_code)


def count_openrouter(model_code):
    import openai

    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"].strip())

    def prompt_tokens(content):
        resp = client.chat.completions.create(model=model_code, max_tokens=1, messages=[{"role": "user", "content": content}])
        return resp.usage.prompt_tokens

    baseline = prompt_tokens([{"type": "text", "text": PROMPT}])
    print(f"text-only baseline: {baseline} tokens\n")
    for h, w in TEST_SIZES:
        img = make_image(h, w, model_code)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        total = prompt_tokens(
            [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]
        )
        report(h, w, total - baseline, model_code)


def report(h, w, measured, model_code):
    predicted = predicted_image_tokens(h, w, model_code)
    ratio = measured / predicted if predicted else float("nan")
    flag = "OK" if 0.85 <= ratio <= 1.15 else "MISMATCH"
    print(f"{h:>5}x{w:<5}  measured={measured:>6}  predicted={predicted:>6}  ratio={ratio:5.2f}  {flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="google", choices=["google", "openrouter"])
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model code (OpenRouter form for --provider openrouter).")
    args = parser.parse_args()

    print(f"provider={args.provider}  model={args.model}  series={gemini_image_caps(args.model)['series']}\n")
    if args.provider == "google":
        count_google(args.model)
    else:
        count_openrouter(args.model)


if __name__ == "__main__":
    main()
