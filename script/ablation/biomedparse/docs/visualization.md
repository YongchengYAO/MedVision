# Per-sample Figure Convention

All per-sample figures produced by `src/eval_detect.py` and `src/eval_tl.py` follow a shared convention. The same pattern should be used when adding visualizations to the MedVision benchmark.

### Shared: 90° CCW image rotation

Both tasks rotate the 2D image 90° CCW before display so the anatomical long axis is horizontal.

```
Array space (origin upper-left):     After np.rot90 → display space:
  (dim0, dim1)                         x-axis = dim0  (pixel_sizes[0] per step)
                                        y-axis = dim1 flipped  (pixel_sizes[1] per step)
```

Transform for any array-space point `(dim0, dim1)` to display `(x, y)`:
```
x = dim0
y = W_orig - 1 - dim1        # W_orig = img_2d.shape[1]
```

Aspect ratio for `imshow` after rotation:
```python
aspect_ratio = pixel_sizes[1] / pixel_sizes[0]   # y-physical / x-physical
```

---

### Detection task — `plot_bbox_on_image`

**File:** `src/eval_detect.py`

**Inputs:** `img_2d (H,W)`, GT coords `[dim0_min, dim1_min, dim0_max, dim1_max]`, model coords same format.

**Steps:**
1. Rotate image: `img_2d = np.rot90(img_2d)` — shape becomes `(W, H)`.
2. Rotate bounding box corners with `rotate_coords(coords, w_orig)`:
   ```python
   # Input:  [ymin, xmin, ymax, xmax]  (array-space, y=dim0, x=dim1)
   # Output: [W-1-xmax, ymin, W-1-xmin, ymax]  (display-space after 90° CCW)
   ```
3. Aspect ratio: `pixel_size[0] / pixel_size[1]` — note: after rotation x=original-dim0, so height/width ratio flips vs. the array convention.
4. Draw GT box (green `Rectangle`) and model box (red `Rectangle`) using `(x_min, y_min), width, height`.

> **Note on detect aspect ratio:** detect uses `pixel_size[0] / pixel_size[1]` while TL uses `pixel_sizes[1] / pixel_sizes[0]`. Both are correct: the detect script uses `patches.Rectangle(xy=(x_min, y_min), ...)` where x=display-col and y=display-row, so the aspect formula differs by convention. Verify with your display call.

---

### TL (Tumor/Lesion) task — `plot_ellipse_on_image`

**File:** `src/eval_tl.py`

**Inputs:** `img_2d (H,W)`, `mask_2d (H,W)` (predicted segmentation), `valid_ellipses_info["landmarks_coords"]` (list of 4-tuples `(dim0, dim1)` per cluster).

**Steps:**
1. Rotate image and mask: `img_rot = np.rot90(img_2d)`, `mask_rot = np.rot90(mask_2d)`.
2. Display: `ax.imshow(img_rot, cmap="gray", aspect=pixel_sizes[1]/pixel_sizes[0])`.
3. Predicted mask contour (green): `ax.contour(mask_rot > 0, levels=[0.5])`.
4. Convert landmarks with `_to_disp(pt) = (pt[0], W_orig - 1 - pt[1])`, then draw:
   - Orange line P1→P2: major ellipse axis.
   - Yellow line P3→P4: minor ellipse axis.
   - White dots at all four landmarks.
5. L-shaped scale bar (lower-left corner): same physical `scale_mm` expressed in pixels along both axes using `_get_appropriate_scale`. Horizontal arm along x (dim0), vertical arm going upward (y decreasing). Label to the right of the horizontal arm.

---

### Scale bar helper

```python
def _get_appropriate_scale(pixel_size, img_size, init_scale=10):
    """Returns (scale_mm, scale_pixels) for a scale bar that is 5–25% of the image dimension."""
```

For a 2-direction L-bar matching both axes:
```python
min_idx = np.argmin(img_2d.shape[:2])
scale_mm, scale_px_min = _get_appropriate_scale(pixel_sizes[min_idx], img_2d.shape[min_idx], 10)
scale_px_other = int(scale_mm / pixel_sizes[1 - min_idx])
scale_px_dim0 = scale_px_min if min_idx == 0 else scale_px_other
scale_px_dim1 = scale_px_other if min_idx == 0 else scale_px_min
```
