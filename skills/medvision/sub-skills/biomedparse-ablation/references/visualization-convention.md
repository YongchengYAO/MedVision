# Per-sample Figure Convention (shared rotation + coordinate transform)

Distilled from the ablation's `docs/visualization.md` and the plotting functions in `eval_detect.py`
(`plot_bbox_on_image`) and `eval_tl.py` (`plot_ellipse_on_image`, `_get_appropriate_scale`). The same convention is
expected when adding per-sample visualizations to the MedVision benchmark itself (see
`../../../references/visualization-catalog.md` for the benchmark's figure entry points).

## Shared: 90-degree CCW rotation

Both tasks rotate the 2-D slice 90 degrees counter-clockwise (`np.rot90`) before display so the anatomical long
axis is horizontal.

```
Array space (origin upper-left):     After np.rot90 -> display space:
  img_2d[dim0, dim1]                    x-axis = dim0            (pixel_sizes[0] per step)
                                        y-axis = dim1, flipped   (pixel_sizes[1] per step)
```

Transform of any array-space point `(dim0, dim1)` to display `(x, y)`:

```
x = dim0
y = W_orig - 1 - dim1          # W_orig = img_2d.shape[1] (the ORIGINAL width, before rotation)
```

After rotation the array shape becomes `(W_orig, H_orig)`.

## Aspect ratio for `imshow`

Two formulas appear, both correct for their respective display call:

| Task | Call | Aspect |
|---|---|---|
| T/L | `ax.imshow(img_rot, cmap="gray", aspect=pixel_sizes[1] / pixel_sizes[0])` | y-physical / x-physical |
| Detection | `ax.imshow(img_2d_rotated, cmap="gray", aspect=pixel_size[0] / pixel_size[1])` | the detect script draws `patches.Rectangle(xy=(x_min, y_min), ...)` with x = display column and y = display row; the ratio differs by convention |

Verify against your own display call before copying either formula.

## Detection - `plot_bbox_on_image`

Inputs: `img_2d (H, W)`, GT box `[dim0_min, dim1_min, dim0_max, dim1_max]`, model box in the same format,
`pixel_size`.

1. `h_orig, w_orig = img_2d.shape[:2]`; `img_2d = np.rot90(img_2d)` (shape `(W, H)`).
2. Rotate both boxes with `rotate_coords(coords, w_orig)`:

   ```python
   # input : [ymin, xmin, ymax, xmax]   (array space, y = dim0, x = dim1)
   # output: [w-1-xmax, ymin, w-1-xmin, ymax]   (display space after 90-degree CCW)
   new_ymin = w_dim - 1 - xmax
   new_ymax = w_dim - 1 - xmin
   new_xmin = ymin
   new_xmax = ymax
   ```

3. `aspect_ratio = pixel_size[0] / pixel_size[1]`.
4. Draw GT as a green `Rectangle((x_min, y_min), x_max - x_min, y_max - y_min)` and the model box in red;
   legend, `axis("off")`, title `'<label>'\n(P: .., R: .., F1: .., IoU: ..)`; save `<fig_dir>/<base>.png`,
   `dpi=100`, `bbox_inches="tight"`.

## T/L - `plot_ellipse_on_image`

Inputs: `img_2d (H, W)`, predicted `mask_2d (H, W)`, optional `gt_mask_2d`, `valid_ellipses_info["landmarks_coords"]`
(list of 4-tuples `(dim0, dim1)` per cluster, P1-P4), `pixel_sizes`, GT/pred axis lengths, MAE, MRE.

1. `img_rot = np.rot90(img_2d)`, `mask_rot = np.rot90(mask_2d)` (and the GT mask when present).
2. `ax.imshow(img_rot, cmap="gray", aspect=pixel_sizes[1] / pixel_sizes[0])`.
3. Predicted contour: `ax.contour(mask_rot > 0, levels=[0.5], colors=["#97D540"], linewidths=2)`; GT contour in
   cyan when the GT mask has any foreground.
4. Landmarks: `_to_disp(pt) = (pt[0], W_orig - 1 - pt[1])`; draw P1->P2 (major axis, `#F37020`), P3->P4 (minor
   axis, `#FBBC05`), white dots with black edges at all four points; legend entries carry the predicted lengths in
   mm.
5. L-shaped scale bar in the lower-left corner (below); title with GT/pred axes, MAE, MRE.
6. Save into an MRE bucket: `bucket = min(int(mre / 0.1) + 1, 9)` -> `<fig_dir>/MRE0<bucket>/<base>.png`.

## Scale bar helper

```python
def _get_appropriate_scale(pixel_size, img_size, init_scale=10):
    """Return (scale_mm, scale_pixels) for a bar that spans 5-25 % of the image dimension."""
    scales = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]   # mm
    scale_pixels_num = int(init_scale / pixel_size)
    min_pixels, max_pixels = img_size * 0.05, img_size * 0.25
    if scale_pixels_num < min_pixels:      # try the next larger scale
        for scale in scales:
            if scale > init_scale:
                return _get_appropriate_scale(pixel_size, img_size, scale)
    elif scale_pixels_num > max_pixels:    # try the next smaller scale
        for scale in reversed(scales):
            if scale < init_scale:
                return _get_appropriate_scale(pixel_size, img_size, scale)
    return init_scale, scale_pixels_num
```

Two-direction L-bar with the same physical length along both axes (pick the scale from the shorter image
dimension, then express that many mm in pixels for the other dimension):

```python
min_idx = np.argmin(img_2d.shape[:2])
scale_mm, scale_px_min = _get_appropriate_scale(pixel_sizes[min_idx], img_2d.shape[min_idx], 10)
scale_px_other = int(scale_mm / pixel_sizes[1 - min_idx])
scale_px_dim0 = scale_px_min if min_idx == 0 else scale_px_other
scale_px_dim1 = scale_px_other if min_idx == 0 else scale_px_min

# rotated display: x = dim0 (0 .. H_orig-1), y = dim1 flipped (0 = top, W_orig-1 = bottom)
sb_x = int(H_orig * 0.05)
sb_y = int(W_orig * 0.88)
ax.plot([sb_x, sb_x + scale_px_dim0], [sb_y, sb_y], color="white", linewidth=3)           # horizontal arm (dim0)
ax.plot([sb_x, sb_x], [sb_y, sb_y - scale_px_dim1], color="white", linewidth=3)           # vertical arm, upward
ax.text(sb_x + scale_px_dim0 + int(H_orig * 0.01), sb_y, f"{scale_mm} mm", color="white", ha="left", va="center")
```

## Checklist when re-using the convention

- Compute `W_orig` from the **un-rotated** image; every landmark/box transform uses it.
- Rotate masks with the same `np.rot90` call as the image (no transpose + `origin="lower"` mix).
- Keep the physical aspect ratio from `pixel_sizes`, never from array shape.
- Use the L-bar so anisotropic pixels are visible; label to the right of the horizontal arm.
- Bucket T/L figures by MRE so quality tiers can be browsed; delete stale figures when re-scoring a dataset (the
  MRE, hence the bucket, may change).
