"""Empirical fixed-point check: is Gemini's coordinate frame the SENT image canvas?

WHY: MedVision TL/AD prompts state the sent image size and ask for relative coordinates
normalized by that canvas. For Claude this required a client pre-resize (the server pads
the canvas); for Gemini the docs say spatial outputs are normalized to the INPUT image,
so we send images unchanged (see gemini_resized_hw() in lmms_eval/models/gemini.py).
The Gemini 2.5 tiling pipeline is documented; the Gemini 3 media_resolution resample
geometry is NOT -- this script is the decisive guard for the pass-through assumption.

METHOD: synthetic non-square images with bright square markers at known relative positions
(0.25/0.75 of each axis). The model is asked for each marker's center as relative [0,1]
coordinates. If responses match the ground truth, the model's frame is the sent canvas.
If x/y come back systematically compressed toward a SQUARE frame (e.g. a 512x2048 image
whose marker at x=0.25 is reported near 0.0625-ish after letterbox math), the provider
letterboxes and pass-through would be INVALID for that model -- report and investigate.

Usage:
    python check_gemini_coordinate_frame.py                                   # google API key
    python check_gemini_coordinate_frame.py --model gemini-3.1-pro-preview
    python check_gemini_coordinate_frame.py --provider openrouter --model google/gemini-2.5-pro
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
from lmms_eval.models.gemini import gemini_resized_hw  # noqa: E402

# Markers at fixed RELATIVE positions (x, y) in [0,1]; tolerance generous enough for model
# noise but far below any letterbox-induced systematic shift on the test aspect ratios.
MARKERS = [(0.25, 0.25), (0.75, 0.75)]
TOLERANCE = 0.08
N_TRIALS = 3  # best-of-N per size: tolerate model output noise, still catch a deterministic letterbox
# (h, w): extreme + MedVision-realistic aspect ratios, plus a square control
TEST_SIZES = [(512, 512), (512, 2048), (2048, 512), (1935, 2400)]

PROMPT = (
    "The image shows two bright white square markers on a black background. "
    "Return the center of each marker as relative coordinates in [0, 1], where x is the "
    "horizontal position normalized by the image width and y is the vertical position "
    "normalized by the image height, with the origin at the top-left corner. "
    'Answer with ONLY a JSON list of two [x, y] pairs sorted by x, e.g. [[0.1, 0.2], [0.8, 0.9]].'
)


def make_marker_image(h, w):
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    r = max(4, min(h, w) // 32)  # marker half-size: visible at any aspect ratio
    for rx, ry in MARKERS:
        cx, cy = rx * w, ry * h
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill="white")
    nh, nw = gemini_resized_hw(h, w, "gemini-2.5-pro")  # pass-through for all test sizes
    assert (nh, nw) == (h, w)
    return img


def _to_unit(v):
    # Models sometimes emit the [0,1000] box convention instead of [0,1] for a given axis;
    # normalize per-coordinate so this check tests the coordinate FRAME, not the output scale.
    v = float(v)
    return v / 1000.0 if v > 1.5 else v


def parse_pairs(text):
    match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON pair list found in response: {text!r}")
    pairs = json.loads(match.group(0))
    return sorted([(_to_unit(x), _to_unit(y)) for x, y in pairs])


def ask_google(client, model_code, img):
    resp = client.models.generate_content(model=model_code, contents=[img, PROMPT])
    return resp.text or ""


def ask_openrouter(client, model_code, img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    resp = client.chat.completions.create(
        model=model_code,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="google", choices=["google", "openrouter"])
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Model code (OpenRouter form for --provider openrouter).")
    args = parser.parse_args()

    if args.provider == "google":
        from google import genai

        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        client = genai.Client(api_key=api_key)
        ask = ask_google
    else:
        import openai

        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"].strip())
        ask = ask_openrouter

    print(f"provider={args.provider}  model={args.model}  tolerance={TOLERANCE}  best_of={N_TRIALS}\n")
    expected = sorted(MARKERS)
    n_fail = 0
    for h, w in TEST_SIZES:
        img = make_marker_image(h, w)
        # Best-of-N: relative-coordinate output is noisy on extreme aspect ratios, but a real
        # letterbox-to-square would fail EVERY trial deterministically. Take the lowest-error
        # trial so a single bad generation doesn't masquerade as a frame transform.
        best_err, best_got = float("inf"), None
        for _ in range(N_TRIALS):
            try:
                got = parse_pairs(ask(client, args.model, img))
            except Exception:
                continue
            err = max(max(abs(gx - ex), abs(gy - ey)) for (gx, gy), (ex, ey) in zip(got, expected))
            if err < best_err:
                best_err, best_got = err, got
            if err <= TOLERANCE:
                break
        if best_got is None:
            print(f"{h:>5}x{w:<5}  ERROR: no parseable response in {N_TRIALS} trials")
            n_fail += 1
            continue
        ok = best_err <= TOLERANCE
        n_fail += 0 if ok else 1
        print(f"{h:>5}x{w:<5}  expected={expected}  got={[(round(x, 3), round(y, 3)) for x, y in best_got]}  "
              f"max_err={best_err:.3f}  {'OK' if ok else 'MISMATCH'}")

    if n_fail:
        print(f"\n{n_fail}/{len(TEST_SIZES)} sizes FAILED. A systematic shift on non-square sizes (square OK) "
              f"means the provider letterboxes to a square frame -> the pass-through rule in "
              f"lmms_eval/models/gemini.py is INVALID for this model; do not run TL/AD with it until resolved.")
        sys.exit(1)
    print("\nAll sizes OK: the model's coordinate frame is the sent canvas (pass-through holds).")


if __name__ == "__main__":
    main()
