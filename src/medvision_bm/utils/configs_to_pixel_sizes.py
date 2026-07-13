"""Summarize per-task raw pixel-size (H x W, mm) distributions from a MedVision config list (CSV).

Streams each config from the HF ``YongchengYAO/MedVision`` dataset and buckets the per-sample
``pixel_size`` column. The HF loader applies all per-family and version-conditional filtering
(only Tumor-Lesion differs by version) and selects the dataset version from the
``MedVision_PLANNER_VERSION`` env var, so counts are HF-exact by construction. Sibling of
``configs_to_image_sizes.py`` (same driver; mm spacing vs px size). Shared loop:
``size_dist_utils.run``.
"""

from medvision_bm.utils.size_dist_utils import build_parser, run


def _ps_key(ps):
    # `pixel_size` is the slice's in-plane [height, width] spacing in mm (float16 in the dataset).
    # Round to 3 decimals to absorb float16 noise, so one physical spacing maps to one bucket.
    h, w = float(ps[0]), float(ps[1])
    return f"{h:.3f}x{w:.3f}"


def _iso_summary(dist):
    # Isotropic = square pixels (height == width). Keys are "{h:.3f}x{w:.3f}",
    # so a string compare of the two sides is exact at the bucket precision.
    iso = aniso = 0
    for key, n in dist.items():
        h, w = key.split("x")
        if h == w:
            iso += n
        else:
            aniso += n
    return {"isotropic": iso, "anisotropic": aniso}


def main():
    args = build_parser(
        "Summarize per-task raw pixel-size (HxW mm) distributions by streaming a MedVision "
        "config list (CSV) from the HF dataset."
    ).parse_args()
    run(args, "pixel_size", _ps_key, _iso_summary, "pixel sizes")


if __name__ == "__main__":
    main()
