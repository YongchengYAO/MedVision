"""Summarize per-task 2D image-size (H x W, pixels) distributions from a MedVision config list (CSV).

Streams each config from the HF ``YongchengYAO/MedVision`` dataset and buckets the per-sample
``image_size_2d`` column. The HF loader applies all per-family and version-conditional filtering
(only Tumor-Lesion differs by version) and selects the dataset version from the
``MedVision_PLANNER_VERSION`` env var, so counts are HF-exact by construction. Sibling of
``configs_to_pixel_sizes.py`` (same driver; px size vs mm spacing). Shared loop:
``size_dist_utils.run``.
"""

from medvision_bm.utils.size_dist_utils import build_parser, run


def _is_key(size):
    # `image_size_2d` is the slice's 2D [height, width] in pixels (uint16 in the dataset).
    h, w = int(size[0]), int(size[1])
    return f"{h}x{w}"


def _wstats(pairs):
    """min/max/(weighted)median of (value, count) pairs."""
    if not pairs:
        return {}
    pairs = sorted(pairs)
    total = sum(n for _, n in pairs)
    cum, median = 0, pairs[0][0]
    for value, n in pairs:
        cum += n
        if cum >= total / 2:
            median = value
            break
    return {"min": pairs[0][0], "max": pairs[-1][0], "median": median}


def _size_summary(dist):
    """Square/non-square sample counts + H/W min/max/median for one task's distribution."""
    square = nonsquare = 0
    hs, ws = [], []
    for key, n in dist.items():
        h, w = (int(x) for x in key.split("x"))
        if h == w:
            square += n
        else:
            nonsquare += n
        hs.append((h, n))
        ws.append((w, n))
    return {
        "square": square,
        "nonsquare": nonsquare,
        "n_distinct_sizes": len(dist),
        "height": _wstats(hs),
        "width": _wstats(ws),
    }


def main():
    args = build_parser(
        "Summarize per-task 2D image-size (HxW pixels) distributions by streaming a MedVision "
        "config list (CSV) from the HF dataset."
    ).parse_args()
    run(args, "image_size_2d", _is_key, _size_summary, "image sizes")


if __name__ == "__main__":
    main()
