"""Empirical check: verify the OpenAI image-resize rules against the live API.

The resize rule (see openai_resized_hw() in lmms_eval/models/openai.py) predicts the image
size the model actually sees; this script measures the image's real token cost and compares
it to the prediction. Image tokens are isolated by subtracting a text-only baseline.
OpenAI has no count_tokens endpoint, so each probe is a 1-output-token chat completion and
reads usage.prompt_tokens (negligible cost).

Two checks (the doc gaps flagged in lmms_eval/models/openai.py):

1. PATCH family (gpt-5.5): the measured image tokens must be PROPORTIONAL to the
   predicted patch count (h/32)*(w/32) with the SAME ratio across differently-shaped
   images -- that constant ratio is the billing multiplier (measured ~1.2 live on
   gpt-5.5, 2026-06-12, vs the 1.0 the docs suggest; cost-only, no geometry effect).
   Geometry is what matters for MedVision: any server-side pad/resize/transpose would
   change the per-image ratio differently for square vs non-square inputs (ceil effects),
   so a constant ratio confirms the server perceives exactly the patch grid we predict.

2. TILE family (gpt-4o): the docs do not explicitly say the shortest-side-768 step never
   UPSCALES, and partial-tile perception is undocumented. A raw 512x512 image at
   detail "high" is decisive:
       ~ 85 + 170*1 = 255 image tokens -> image unchanged (downscale-only; our
         min(1, 2048/long, 768/short) fixed point is correct)
       ~ 85 + 170*4 = 765 image tokens -> the server upscaled to 768x768; the tile
         formula in openai_resized_hw() must be changed to PIN the short side to 768.

Usage:
    python unit-test/openai-image-resize/check_openai_count_tokens.py                        # official API (OPENAI_API_KEY)
    python unit-test/openai-image-resize/check_openai_count_tokens.py --provider openrouter  # OpenRouter (OPENROUTER_API_KEY)
"""

import argparse
import base64
import io
import os
import sys

from PIL import Image

# Make the vendored lmms_eval importable as a top-level package, then import the shipped
# resize function directly (run in an env with the eval deps, e.g. the `eval-openai` env).
# __file__ is unit-test/openai-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))
from lmms_eval.models.openai import openai_resized_hw  # noqa: E402


def make_png_b64(h, w, model_code=None):
    """Noise PNG (incompressible, realistic); pre-resized via the shipped rule unless model_code is None."""
    img = Image.effect_noise((w, h), 64).convert("RGB")
    nh, nw = (h, w)
    if model_code is not None:
        nh, nw = openai_resized_hw(h, w, model_code)
        if (nh, nw) != (h, w):
            img = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode(), nh, nw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    args = parser.parse_args()

    import openai

    # .strip(): pod-injected env secrets can carry a trailing newline (illegal in HTTP headers)
    if args.provider == "openai":
        prefix = ""
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    else:
        prefix = "openai/"
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"].strip(),
        )

    def count_input_tokens(model, content):
        kwargs = dict(model=model, messages=[{"role": "user", "content": content}])
        if args.provider == "openai":
            kwargs["max_completion_tokens"] = 16  # gpt-5.x rejects max_tokens
        else:
            kwargs["max_tokens"] = 16
            kwargs["extra_body"] = {"usage": {"include": True}}
        return client.chat.completions.create(**kwargs).usage.prompt_tokens

    def image_block(b64):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}

    # ---- Check 1: PATCH family (gpt-5.5) -- geometry via constant measured/predicted ratio ----
    model = f"{prefix}gpt-5.5"
    baseline = count_input_tokens(model, [{"type": "text", "text": "hi"}])
    print(f"[patch] model={model} | text-only baseline: {baseline} tokens")
    ratios = []
    for h, w in [(512, 512), (1800, 2400)]:  # one square, one non-square (downscaled by the budget)
        b64, nh, nw = make_png_b64(h, w, model)
        total = count_input_tokens(model, [image_block(b64), {"type": "text", "text": "hi"}])
        measured = total - baseline
        predicted = (nh // 32) * (nw // 32)
        ratios.append(measured / predicted)
        print(f"[patch] orig {h}x{w} -> pre-resized {nh}x{nw} | measured image tokens: {measured} | "
              f"predicted ({nh // 32}x{nw // 32} patches): {predicted} | measured/predicted = {ratios[-1]:.3f}")
    # Same ratio for a square and a non-square image => the server perceives exactly the
    # predicted patch grid (any pad/resize would shift the two ratios differently); the
    # common ratio is the billing multiplier (cost-only). Live-measured ~1.2 on gpt-5.5.
    assert abs(ratios[0] - ratios[1]) <= 0.05, \
        f"FAIL: measured/predicted ratios differ across shapes ({ratios[0]:.3f} vs {ratios[1]:.3f}) -- geometry mismatch"
    assert 0.85 <= ratios[0] <= 1.5, f"FAIL: implausible image-token multiplier {ratios[0]:.3f}"
    print(f"[patch] geometry OK (constant ratio); billing multiplier ~= {sum(ratios) / 2:.2f}")

    # ---- Check 2: TILE family (gpt-4o) -- decisive upscale probe, RAW image ----
    model = f"{prefix}gpt-4o"
    baseline = count_input_tokens(model, [{"type": "text", "text": "hi"}])
    print(f"[tile] model={model} | text-only baseline: {baseline} tokens")
    b64, _, _ = make_png_b64(512, 512, model_code=None)  # raw 512x512, no pre-resize
    total = count_input_tokens(model, [image_block(b64), {"type": "text", "text": "hi"}])
    measured = total - baseline
    unchanged, upscaled = 85 + 170 * 1, 85 + 170 * 4
    print(f"[tile] raw 512x512 @ detail=high | measured image tokens: {measured} | "
          f"hypotheses: unchanged(1 tile)={unchanged}, upscaled-to-768(4 tiles)={upscaled}")
    if abs(measured - unchanged) <= abs(measured - upscaled):
        print("[tile] VERDICT: image NOT upscaled -- the min(1, 2048/long, 768/short) fixed point holds.")
    else:
        raise AssertionError(
            "[tile] VERDICT: the server UPSCALED the image (shortest-side-768 is not downscale-only). "
            "Change the tile branch of openai_resized_hw() to pin the short side to 768."
        )

    print("Empirical check PASSED: resize rules match live API behavior.")


if __name__ == "__main__":
    main()
