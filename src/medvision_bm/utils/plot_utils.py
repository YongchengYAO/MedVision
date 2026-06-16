import os

import matplotlib.lines as mlines
import numpy as np
from matplotlib import pyplot as plt


def _get_appropriate_scale(pixel_size, img_size, init_scale=10):
    """
    Calculate appropriate scale bar size in mm and pixels.
    Args:
        pixel_size (float): Size of one pixel in mm
        img_size (int): Smallest image dimension in pixels
        init_scale (int): Initial scale in mm (default 10mm)
    Returns:
        tuple: (scale_mm, scale_pixels) - Selected scale in mm and pixels
    """
    scales = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    scale_pixels_num = int(init_scale / pixel_size)
    min_pixels = img_size * 0.05
    max_pixels = img_size * 0.25
    if scale_pixels_num < min_pixels:
        for scale in scales:
            if scale > init_scale:
                return _get_appropriate_scale(pixel_size, img_size, scale)
    elif scale_pixels_num > max_pixels:
        for scale in reversed(scales):
            if scale < init_scale:
                return _get_appropriate_scale(pixel_size, img_size, scale)
    return init_scale, scale_pixels_num


def plot_tl_axes_on_image(
    image_2d,
    pixel_sizes,
    major_axis_pts,
    minor_axis_pts,
    slice_dim,
    slice_idx,
    fig_path,
    mask_2d=None,
    show_coords=False,
    gt_major_pts=None,
    gt_minor_pts=None,
):
    """
    Plot a 2D medical image slice with GT axes (dashed), model-predicted axes (solid),
    and optional mask contour.

    Display: 90° CCW rotation via imshow(image_2d.T, origin="lower").
        plot_x = idx_dim0 (originally rows, now horizontal, increasing left→right)
        plot_y = idx_dim1 (originally cols, now vertical,   increasing bottom→top)

    Coordinate conversion (model image space → array space → plot):
        Model reports (x_rel, y_rel) with origin lower-left:
            idx_dim1 = x_rel * W       → plot_y
            idx_dim0 = H * (1 - y_rel) → plot_x

    Args:
        image_2d:        (H, W) float array, normalized to [0, 1], original (non-resized) shape
        pixel_sizes:     [px_dim0, px_dim1] physical pixel sizes in mm (array space)
        major_axis_pts:  [(dim0_p1, dim1_p1), (dim0_p2, dim1_p2)] predicted, in array space
        minor_axis_pts:  [(dim0_p3, dim1_p3), (dim0_p4, dim1_p4)] predicted, in array space
        slice_dim:       0=Sagittal, 1=Coronal, 2=Axial
        slice_idx:       integer slice index
        fig_path:        full output path including filename
        mask_2d:         optional (H, W) binary array; plotted as green contour if provided
        gt_major_pts:    optional [(dim0_p1, dim1_p1), (dim0_p2, dim1_p2)] GT, in array space
        gt_minor_pts:    optional [(dim0_p3, dim1_p3), (dim0_p4, dim1_p4)] GT, in array space
    """
    img_height, img_width = image_2d.shape  # H=dim0, W=dim1
    # After 90° CCW: plot-x unit = 1 row (pixel_sizes[0] mm), plot-y unit = 1 col (pixel_sizes[1] mm)
    aspect_ratio = pixel_sizes[1] / pixel_sizes[0]

    base_size = 10
    # Rotated visible extents: horizontal=H, vertical=W
    img_aspect = img_height / img_width
    figsize = (
        (base_size * img_aspect, base_size)
        if img_aspect > 1
        else (base_size, base_size / img_aspect)
    )

    plt.figure(figsize=figsize)
    plt.imshow(
        image_2d.T,
        cmap="gray",
        origin="lower",
        aspect=aspect_ratio,
        zorder=-1,
    )

    _extra_legend_handles = []
    if mask_2d is not None:
        plt.contour(
            mask_2d.T,
            levels=[0.5],
            colors="#2ECC71",
            linewidths=4,
            zorder=0,
        )
        _extra_legend_handles.append(
            mlines.Line2D([], [], color="#2ECC71", linewidth=4, label="GT mask")
        )

    # In rotated plot space: plot_x = idx_dim0, plot_y = idx_dim1
    # GT axes (dashed) — drawn before predicted so predicted renders on top
    if gt_major_pts is not None:
        plt.plot(
            [gt_major_pts[0][0], gt_major_pts[1][0]],
            [gt_major_pts[0][1], gt_major_pts[1][1]],
            color="#A21CAF",
            linestyle="--",
            linewidth=4,
            zorder=2,
        )
        _extra_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="#A21CAF",
                linestyle="--",
                linewidth=4,
                label="GT major axis",
            )
        )
    if gt_minor_pts is not None:
        plt.plot(
            [gt_minor_pts[0][0], gt_minor_pts[1][0]],
            [gt_minor_pts[0][1], gt_minor_pts[1][1]],
            color="#4F46E5",
            linestyle="--",
            linewidth=4,
            zorder=2,
        )
        _extra_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="#4F46E5",
                linestyle="--",
                linewidth=4,
                label="GT minor axis",
            )
        )

    if major_axis_pts is not None:
        # Predicted axes (solid)
        # Major axis (P1→P2, orange)
        plt.plot(
            [major_axis_pts[0][0], major_axis_pts[1][0]],  # x: dim0
            [major_axis_pts[0][1], major_axis_pts[1][1]],  # y: dim1
            color="#F37020",
            linestyle="-",
            linewidth=3,
            label="Pred major axis",
            zorder=3,
        )
        # Minor axis (P3→P4, yellow)
        plt.plot(
            [minor_axis_pts[0][0], minor_axis_pts[1][0]],  # x: dim0
            [minor_axis_pts[0][1], minor_axis_pts[1][1]],  # y: dim1
            color="#FBBC05",
            linestyle="-",
            linewidth=3,
            label="Pred minor axis",
            zorder=3,
        )

        # Landmark dots P1–P4: scatter(x=dim0, y=dim1)
        colors = ["#4285F4", "#EA4335", "#FDB813", "#34A853"]
        offset_x = img_height * 0.015 if show_coords else 0
        for j, (dim0, dim1) in enumerate(
            [major_axis_pts[0], major_axis_pts[1], minor_axis_pts[0], minor_axis_pts[1]]
        ):
            plt.scatter(
                dim0,
                dim1,
                color=colors[j],
                edgecolors="black",
                marker="o",
                s=60,
                linewidth=1.5,
                zorder=4,
            )
            if show_coords:
                x_rel = dim1 / img_width
                y_rel = 1.0 - dim0 / img_height
                plt.annotate(
                    f"P{j + 1} ({x_rel:.3f}, {y_rel:.3f})",
                    xy=(dim0, dim1),
                    xytext=(dim0 + offset_x, dim1),
                    color=colors[j],
                    fontsize=7,
                    va="center",
                    zorder=5,
                )

    # L-shaped scale bar (lower-left corner).
    # With origin="lower", lower-left = small plot_x (dim0), small plot_y (dim1).
    # Horizontal arm grows +plot_x (+dim0), vertical arm grows +plot_y (+dim1).
    min_idx = np.argmin(image_2d.shape[:2])
    scale_mm, num_pixels_dim_min = _get_appropriate_scale(
        pixel_sizes[min_idx], image_2d.shape[min_idx], init_scale=10
    )
    num_pixels_dim_max = int(scale_mm / pixel_sizes[1 - min_idx])
    if min_idx == 0:
        scale_pixels_dim0, scale_pixels_dim1 = num_pixels_dim_min, num_pixels_dim_max
    else:
        scale_pixels_dim0, scale_pixels_dim1 = num_pixels_dim_max, num_pixels_dim_min
    start_x = int(img_height * 0.05)  # dim0
    start_y = int(img_width * 0.05)  # dim1
    end_x = start_x + scale_pixels_dim0
    end_y = start_y + scale_pixels_dim1
    plt.plot([start_x, end_x], [start_y, start_y], "w-", linewidth=4)
    plt.plot([start_x, start_x], [start_y, end_y], "w-", linewidth=4)
    plt.text(
        end_x + img_height * 0.01,
        start_y,
        f"{scale_mm} mm",
        color="white",
        horizontalalignment="left",
        fontsize=20,
    )

    if slice_dim == 0:
        plt.xlabel("Anterior →", fontsize=28)
        plt.ylabel("Superior →", fontsize=28)
    elif slice_dim == 1:
        plt.xlabel("Right →", fontsize=28)
        plt.ylabel("Superior →", fontsize=28)
    else:
        plt.xlabel("Right →", fontsize=28)
        plt.ylabel("Anterior →", fontsize=28)

    plt.tick_params(labelsize=20)
    _ax_handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=_extra_legend_handles + _ax_handles, loc="upper right", fontsize=16
    )
    plt.tight_layout(pad=1.5, rect=[0.05, 0.05, 0.95, 0.95])

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def plot_detection_on_image(
    image_2d,
    pixel_sizes,
    gt_box,
    pred_box,
    slice_dim,
    slice_idx,
    fig_path,
):
    """
    Plot a 2D medical image slice with GT and model-predicted bounding boxes.

    Display: 90° CCW rotation via imshow(image_2d.T, origin="lower").
        plot_x = idx_dim0 (originally rows, now horizontal)
        plot_y = idx_dim1 (originally cols, now vertical)

    Coordinate conversion (image space → array space):
        [x_min, y_min, x_max, y_max] normalized, origin lower-left:
            dim0_min = H * (1 - y_max)    dim1_min = x_min * W
            dim0_max = H * (1 - y_min)    dim1_max = x_max * W
        Rectangle: anchor=(dim0_min, dim1_min), width=dim0_max-dim0_min, height=dim1_max-dim1_min

    Args:
        image_2d:    (H, W) float array, normalized to [0, 1]
        pixel_sizes: [px_dim0, px_dim1] physical pixel sizes in mm
        gt_box:      [dim0_min, dim1_min, dim0_max, dim1_max] in array space, or None
        pred_box:    [dim0_min, dim1_min, dim0_max, dim1_max] in array space, or None
        slice_dim:   0=Sagittal, 1=Coronal, 2=Axial
        slice_idx:   integer slice index
        fig_path:    full output path including filename
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    img_height, img_width = image_2d.shape
    aspect_ratio = pixel_sizes[1] / pixel_sizes[0]

    base_size = 10
    img_aspect = img_height / img_width
    figsize = (
        (base_size * img_aspect, base_size)
        if img_aspect > 1
        else (base_size, base_size / img_aspect)
    )

    plt.figure(figsize=figsize)
    plt.imshow(
        image_2d.T,
        cmap="gray",
        origin="lower",
        aspect=aspect_ratio,
        zorder=-1,
    )

    ax = plt.gca()
    legend_handles = []

    if gt_box is not None:
        d0_min, d1_min, d0_max, d1_max = gt_box
        ax.add_patch(
            mpatches.Rectangle(
                (d0_min, d1_min),
                d0_max - d0_min,
                d1_max - d1_min,
                linewidth=4,
                edgecolor="#2ECC71",
                facecolor="none",
                linestyle="-",
                zorder=2,
            )
        )
        legend_handles.append(
            mlines.Line2D([], [], color="#2ECC71", ls="-", linewidth=4, label="GT bbox")
        )

    if pred_box is not None:
        d0_min, d1_min, d0_max, d1_max = pred_box
        ax.add_patch(
            mpatches.Rectangle(
                (d0_min, d1_min),
                d0_max - d0_min,
                d1_max - d1_min,
                linewidth=4,
                edgecolor="#F37020",
                facecolor="none",
                linestyle="-",
                zorder=3,
            )
        )
        legend_handles.append(
            mlines.Line2D(
                [], [], color="#F37020", ls="-", linewidth=4, label="Pred bbox"
            )
        )

    # L-shaped scale bar (lower-left corner) — identical to plot_tl_axes_on_image
    min_idx = np.argmin(image_2d.shape[:2])
    scale_mm, num_pixels_dim_min = _get_appropriate_scale(
        pixel_sizes[min_idx], image_2d.shape[min_idx], init_scale=10
    )
    num_pixels_dim_max = int(scale_mm / pixel_sizes[1 - min_idx])
    if min_idx == 0:
        scale_pixels_dim0, scale_pixels_dim1 = num_pixels_dim_min, num_pixels_dim_max
    else:
        scale_pixels_dim0, scale_pixels_dim1 = num_pixels_dim_max, num_pixels_dim_min
    start_x = int(img_height * 0.05)
    start_y = int(img_width * 0.05)
    end_x = start_x + scale_pixels_dim0
    end_y = start_y + scale_pixels_dim1
    plt.plot([start_x, end_x], [start_y, start_y], "w-", linewidth=4)
    plt.plot([start_x, start_x], [start_y, end_y], "w-", linewidth=4)
    plt.text(
        end_x + img_height * 0.01,
        start_y,
        f"{scale_mm} mm",
        color="white",
        horizontalalignment="left",
        fontsize=20,
    )

    if slice_dim == 0:
        plt.xlabel("Anterior →", fontsize=28)
        plt.ylabel("Superior →", fontsize=28)
    elif slice_dim == 1:
        plt.xlabel("Right →", fontsize=28)
        plt.ylabel("Superior →", fontsize=28)
    else:
        plt.xlabel("Right →", fontsize=28)
        plt.ylabel("Anterior →", fontsize=28)

    plt.tick_params(labelsize=20)
    plt.legend(handles=legend_handles, loc="upper right", fontsize=16)
    plt.tight_layout(pad=1.5, rect=[0.05, 0.05, 0.95, 0.95])

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def plot_ad_on_image(
    image_2d,
    pixel_sizes,
    metric_type,
    gt_pts,
    pred_pts,
    slice_dim,
    slice_idx,
    fig_path,
    show_coords=False,
):
    """
    Plot a 2D image slice with GT and model-predicted landmarks for an A/D task.

    Display: 90° CCW rotation via imshow(image_2d.T, origin="lower").
        plot_x = idx_dim0 (row), plot_y = idx_dim1 (col)

    Args:
        image_2d:    (H, W) array, normalized
        pixel_sizes: [px_dim0, px_dim1] in mm
        metric_type: "distance" or "angle"
        gt_pts:      dict from _get_gt_coords (array-space coords as [row, col] lists)
        pred_pts:    tuple from _parse_dist_preds / _parse_angle_preds (array-space tuples)
        slice_dim:   0=Sagittal, 1=Coronal, 2=Axial
        slice_idx:   integer slice index
        fig_path:    output path including filename
        show_coords: annotate each dot with its relative (x, y)
    """
    import matplotlib.lines as mlines

    _DOT_COLORS = ["#4285F4", "#EA4335", "#FDB813", "#34A853"]

    img_height, img_width = image_2d.shape
    aspect_ratio = pixel_sizes[1] / pixel_sizes[0]

    base_size = 10
    img_aspect = img_height / img_width
    figsize = (
        (base_size * img_aspect, base_size)
        if img_aspect > 1
        else (base_size, base_size / img_aspect)
    )

    plt.figure(figsize=figsize)
    plt.imshow(
        image_2d.T,
        cmap="gray",
        origin="lower",
        aspect=aspect_ratio,
        zorder=-1,
    )

    legend_handles = []

    if metric_type == "distance":
        gt_p1, gt_p2 = gt_pts["p1"], gt_pts["p2"]

        plt.plot(
            [gt_p1[0], gt_p2[0]],
            [gt_p1[1], gt_p2[1]],
            color="#A21CAF",
            linestyle="--",
            linewidth=4,
            zorder=2,
        )
        plt.scatter(
            [gt_p1[0], gt_p2[0]],
            [gt_p1[1], gt_p2[1]],
            color=[_DOT_COLORS[0], _DOT_COLORS[1]],
            edgecolors="black",
            s=80,
            linewidth=1.5,
            zorder=4,
        )

        legend_handles = [
            mlines.Line2D(
                [],
                [],
                color="#A21CAF",
                linestyle="--",
                linewidth=3,
                label="GT landmarks",
            ),
        ]

        if pred_pts is not None:
            pr_p1, pr_p2 = pred_pts
            plt.plot(
                [pr_p1[0], pr_p2[0]],
                [pr_p1[1], pr_p2[1]],
                color="#F37020",
                linestyle="-",
                linewidth=3,
                zorder=3,
            )
            plt.scatter(
                [pr_p1[0], pr_p2[0]],
                [pr_p1[1], pr_p2[1]],
                color=[_DOT_COLORS[2], _DOT_COLORS[3]],
                edgecolors="black",
                s=60,
                linewidth=1.5,
                zorder=4,
            )
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="#F37020",
                    linestyle="-",
                    linewidth=3,
                    label="Pred landmarks",
                )
            )

        if show_coords:
            offset_x = img_height * 0.015
            for pt, color in [(gt_p1, _DOT_COLORS[0]), (gt_p2, _DOT_COLORS[1])]:
                x_rel = pt[1] / img_width
                y_rel = 1.0 - pt[0] / img_height
                plt.annotate(
                    f"({x_rel:.3f}, {y_rel:.3f})",
                    xy=(pt[0], pt[1]),
                    xytext=(pt[0] + offset_x, pt[1]),
                    color=color,
                    fontsize=7,
                    va="center",
                    zorder=5,
                )
            if pred_pts is not None:
                for pt, color in [(pr_p1, _DOT_COLORS[2]), (pr_p2, _DOT_COLORS[3])]:
                    x_rel = pt[1] / img_width
                    y_rel = 1.0 - pt[0] / img_height
                    plt.annotate(
                        f"({x_rel:.3f}, {y_rel:.3f})",
                        xy=(pt[0], pt[1]),
                        xytext=(pt[0] + offset_x, pt[1]),
                        color=color,
                        fontsize=7,
                        va="center",
                        zorder=5,
                    )

    else:  # angle
        gt_l1p1, gt_l1p2 = gt_pts["l1p1"], gt_pts["l1p2"]
        gt_l2p1, gt_l2p2 = gt_pts["l2p1"], gt_pts["l2p2"]

        plt.plot(
            [gt_l1p1[0], gt_l1p2[0]],
            [gt_l1p1[1], gt_l1p2[1]],
            color="#A21CAF",
            linestyle="--",
            linewidth=4,
            zorder=2,
        )
        plt.plot(
            [gt_l2p1[0], gt_l2p2[0]],
            [gt_l2p1[1], gt_l2p2[1]],
            color="#4F46E5",
            linestyle="--",
            linewidth=4,
            zorder=2,
        )

        legend_handles = [
            mlines.Line2D(
                [], [], color="#A21CAF", linestyle="--", linewidth=3, label="GT line 1"
            ),
            mlines.Line2D(
                [], [], color="#4F46E5", linestyle="--", linewidth=3, label="GT line 2"
            ),
        ]

        if pred_pts is not None:
            pr_l1p1, pr_l1p2, pr_l2p1, pr_l2p2 = pred_pts
            plt.plot(
                [pr_l1p1[0], pr_l1p2[0]],
                [pr_l1p1[1], pr_l1p2[1]],
                color="#F37020",
                linestyle="-",
                linewidth=3,
                zorder=3,
            )
            plt.plot(
                [pr_l2p1[0], pr_l2p2[0]],
                [pr_l2p1[1], pr_l2p2[1]],
                color="#FBBC05",
                linestyle="-",
                linewidth=3,
                zorder=3,
            )

            all_pred_pts = [pr_l1p1, pr_l1p2, pr_l2p1, pr_l2p2]
            for j, pt in enumerate(all_pred_pts):
                plt.scatter(
                    pt[0],
                    pt[1],
                    color=_DOT_COLORS[j],
                    edgecolors="black",
                    s=60,
                    linewidth=1.5,
                    zorder=4,
                )
                if show_coords:
                    offset_x = img_height * 0.015
                    x_rel = pt[1] / img_width
                    y_rel = 1.0 - pt[0] / img_height
                    plt.annotate(
                        f"({x_rel:.3f}, {y_rel:.3f})",
                        xy=(pt[0], pt[1]),
                        xytext=(pt[0] + offset_x, pt[1]),
                        color=_DOT_COLORS[j],
                        fontsize=7,
                        va="center",
                        zorder=5,
                    )

            legend_handles += [
                mlines.Line2D(
                    [],
                    [],
                    color="#F37020",
                    linestyle="-",
                    linewidth=3,
                    label="Pred line 1",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color="#FBBC05",
                    linestyle="-",
                    linewidth=3,
                    label="Pred line 2",
                ),
            ]

    # L-shaped scale bar (lower-left corner)
    min_idx = np.argmin(image_2d.shape[:2])
    scale_mm, num_pixels_dim_min = _get_appropriate_scale(
        pixel_sizes[min_idx], image_2d.shape[min_idx], init_scale=10
    )
    num_pixels_dim_max = int(scale_mm / pixel_sizes[1 - min_idx])
    if min_idx == 0:
        scale_pixels_dim0, scale_pixels_dim1 = num_pixels_dim_min, num_pixels_dim_max
    else:
        scale_pixels_dim0, scale_pixels_dim1 = num_pixels_dim_max, num_pixels_dim_min
    start_x = int(img_height * 0.05)
    start_y = int(img_width * 0.05)
    end_x = start_x + scale_pixels_dim0
    end_y = start_y + scale_pixels_dim1
    plt.plot([start_x, end_x], [start_y, start_y], "w-", linewidth=4)
    plt.plot([start_x, start_x], [start_y, end_y], "w-", linewidth=4)
    plt.text(
        end_x + img_height * 0.01,
        start_y,
        f"{scale_mm} mm",
        color="white",
        horizontalalignment="left",
        fontsize=20,
    )

    if slice_dim == 0:
        plt.xlabel("Anterior →", fontsize=28)
        plt.ylabel("Superior →", fontsize=28)
    elif slice_dim == 1:
        plt.xlabel("Right →", fontsize=28)
        plt.ylabel("Superior →", fontsize=28)
    else:
        plt.xlabel("Right →", fontsize=28)
        plt.ylabel("Anterior →", fontsize=28)

    plt.tick_params(labelsize=20)
    plt.legend(handles=legend_handles, loc="upper right", fontsize=16)
    plt.tight_layout(pad=1.5, rect=[0.05, 0.05, 0.95, 0.95])

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
