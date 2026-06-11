"""Empirical check: verify the Anthropic image-resize constants against the live API.

The resize rule (see anthropic_resized_hw() in lmms_eval/models/claude.py) predicts the
image size the model actually sees; this script measures the image's real token cost and
compares it to the prediction. Image tokens are isolated by subtracting a text-only baseline.
Expectation: measured image tokens ~= ceil28(w) * ceil28(h) / 750 (padded dims), within ~15%.
A much LOWER count would mean the server resized further than our formula predicts.

Usage:
    python tests/check_claude_count_tokens.py                        # direct Anthropic (ANTHROPIC_API_KEY)
    python tests/check_claude_count_tokens.py --provider openrouter  # OpenRouter (OPENROUTER_API_KEY)

Provider notes:
    - anthropic: uses the free messages.count_tokens endpoint (no generation).
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
# resize function directly (run in an env with the eval deps, e.g. the `eval-claude` env).
# __file__ is unit-test/claude-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))
from lmms_eval.models.claude import anthropic_resized_hw  # noqa: E402


def make_png_b64(h, w, model_code):
    img = Image.effect_noise((w, h), 64).convert("RGB")  # noise: incompressible, realistic
    nh, nw = anthropic_resized_hw(h, w, model_code)
    if (nh, nw) != (h, w):
        img = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode(), nh, nw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter"])
    args = parser.parse_args()

    # .strip(): pod-injected env secrets can carry a trailing newline (illegal in HTTP headers)
    if args.provider == "anthropic":
        import anthropic

        model = "claude-fable-5"
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())

        def count_input_tokens(content):
            return client.messages.count_tokens(
                model=model, messages=[{"role": "user", "content": content}]
            ).input_tokens

        def image_block(b64):
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}

    else:
        import openai

        model = "anthropic/claude-fable-5"
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"].strip(),
        )

        def count_input_tokens(content):
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1,
                messages=[{"role": "user", "content": content}],
                extra_body={"usage": {"include": True}},
            )
            return resp.usage.prompt_tokens

        def image_block(b64):
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

    baseline = count_input_tokens([{"type": "text", "text": "hi"}])
    print(f"provider={args.provider} model={model} | text-only baseline: {baseline} tokens")

    ceil28 = lambda x: math.ceil(x / 28) * 28
    for h, w in [(512, 512), (4000, 3000)]:
        b64, nh, nw = make_png_b64(h, w, model)
        total = count_input_tokens([image_block(b64), {"type": "text", "text": "hi"}])
        measured = total - baseline
        expected_padded = ceil28(nw) * ceil28(nh) / 750.0
        expected_unpadded = nw * nh / 750.0
        ratio = measured / expected_padded
        print(f"orig {h}x{w} -> pre-resized {nh}x{nw} | measured image tokens: {measured} | "
              f"expected (padded {ceil28(nh)}x{ceil28(nw)}): {expected_padded:.0f} | "
              f"expected (unpadded): {expected_unpadded:.0f} | measured/padded = {ratio:.3f}")
        assert 0.85 <= ratio <= 1.15, f"FAIL: measured {measured} deviates >15% from padded estimate {expected_padded:.0f}"

    print("Empirical check PASSED: resize constants match live API behavior.")


if __name__ == "__main__":
    main()
