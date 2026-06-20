"""
LIVE-API guard for the Kimi / MoonViT pass-through assumption.

test_kimi_resize.py proves (offline) that kimi_resized_hw() is a fixed point of the
OPEN-WEIGHTS navit_resize_image. What it cannot prove is that the HOSTED endpoint
(api.moonshot.ai / OpenRouter) actually runs that same MoonViT geometry on the image we
send. This script settles that empirically -- the same posture as
unit-test/gemini-image-resize/check_gemini_coordinate_frame.py.

How: build a NON-SQUARE canvas, pre-resize it with the production rule (kimi_resized_hw),
draw distinct colored dots at known RELATIVE positions, and ask the model for each dot's
(x, y) as fractions of width/height in [0, 1]. If the model normalizes by the canvas we
sent (no hidden re-tiling / re-padding / thumbnailing), the returned fractions match the
known positions. A wrong/padded canvas shows up as a systematic offset, especially on the
non-square axis.

This costs one API call. It does NOT block the integration -- run it when you have a key.

Run:
  MOONSHOT_API_KEY=... python unit-test/kimi-image-resize/check_kimi_coordinate_frame.py
  OPENROUTER_API_KEY=... python unit-test/kimi-image-resize/check_kimi_coordinate_frame.py \
        --provider openrouter --model moonshotai/kimi-k2.6
(run in an env with the eval deps: openai SDK + the vendored lmms_eval importable)
"""

import argparse
import base64
import io
import json
import os
import re
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))
from lmms_eval.models.kimi import kimi_resized_hw  # noqa: E402

# Known marker positions as (name, color, rel_x, rel_y) -- spread across the canvas so a
# squashed/padded frame on either axis is exposed.
_MARKERS = [
    ("red", (220, 30, 30), 0.25, 0.20),
    ("green", (30, 180, 60), 0.80, 0.35),
    ("blue", (40, 70, 230), 0.50, 0.75),
]


def _build_image(model_code, raw_h=560, raw_w=896):
    """A deliberately NON-SQUARE canvas, then pre-resized exactly as production does."""
    new_h, new_w = kimi_resized_hw(raw_h, raw_w, model_code)
    img = Image.new("RGB", (new_w, new_h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    r = max(8, min(new_w, new_h) // 30)
    for _, color, rx, ry in _MARKERS:
        cx, cy = int(rx * new_w), int(ry * new_h)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img, new_h, new_w


def _encode(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _client(provider):
    import openai

    if provider == "moonshot":
        base_url = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").strip()
        return openai.OpenAI(base_url=base_url, api_key=os.environ["MOONSHOT_API_KEY"].strip())
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="moonshot", choices=["moonshot", "openrouter"])
    ap.add_argument("--model", default=None, help="default kimi-k2.6 (moonshot) / moonshotai/kimi-k2.6 (openrouter)")
    ap.add_argument("--tol", type=float, default=0.06, help="max |error| per axis to PASS")
    args = ap.parse_args()
    model = args.model or ("kimi-k2.6" if args.provider == "moonshot" else "moonshotai/kimi-k2.6")

    img, h, w = _build_image(model)
    names = ", ".join(n for n, *_ in _MARKERS)
    prompt = (
        f"The image is {w} px wide and {h} px tall and contains these colored dots: {names}. "
        f"For each dot, give its center as fractions of the image width (x) and height (y), each in [0,1]. "
        f'Reply ONLY with JSON: {{"red": [x, y], "green": [x, y], "blue": [x, y]}}.'
    )
    resp = _client(args.provider).chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode(img)}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        print(f"[FAIL] could not parse JSON from model reply:\n{text}")
        sys.exit(1)
    got = json.loads(m.group(0))

    print(f"model={model} provider={args.provider} sent canvas={w}x{h} (WxH)")
    ok = True
    for name, _, rx, ry in _MARKERS:
        gx, gy = got.get(name, [None, None])
        if gx is None or gy is None:
            print(f"  {name:5s} MISSING in reply"); ok = False; continue
        ex, ey = abs(gx - rx), abs(gy - ry)
        flag = "ok" if (ex <= args.tol and ey <= args.tol) else "OFF"
        if flag == "OFF":
            ok = False
        print(f"  {name:5s} expected=({rx:.2f},{ry:.2f}) got=({gx:.2f},{gy:.2f}) err=({ex:.2f},{ey:.2f}) [{flag}]")
    print("\nRESULT:", "PASS -- model normalizes by the sent canvas (pass-through holds)" if ok
          else "FAIL -- coordinate frame diverges from the sent canvas; the hosted resize may differ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
