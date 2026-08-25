import numpy as np
import cv2

from ..schemas import Instance2D

from dataclasses import replace
from collections.abc import Sequence

import cv2
import numpy as np


def _expand_mask_region(
    mask: np.ndarray,
    region: tuple[int, int, int, int],
    padding: int,
    image_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Expand a cropped mask region with zeros.
    """
    if padding == 0:
        return mask, region

    x0, y0, x1, y1 = region

    # Image coordinates cannot be negative.
    new_x0 = max(0, x0 - padding)
    new_y0 = max(0, y0 - padding)
    new_x1 = x1 + padding
    new_y1 = y1 + padding

    if image_shape is not None:
        image_h, image_w = image_shape

        new_x1 = min(image_w, new_x1)
        new_y1 = min(image_h, new_y1)

    expanded = np.zeros(
        (new_y1 - new_y0, new_x1 - new_x0),
        dtype=np.bool_,
    )

    offset_x = x0 - new_x0
    offset_y = y0 - new_y0

    h, w = mask.shape

    expanded[
        offset_y : offset_y + h,
        offset_x : offset_x + w,
    ] = mask

    return expanded, (new_x0, new_y0, new_x1, new_y1)


def _crop_mask_to_nonzero(
    mask: np.ndarray,
    region: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop a mask to the smallest rectangle containing all True pixels.
    """
    x0, y0, _, _ = region

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        # Preserve the distinction between an empty mask and mask=None.
        return (
            np.empty((0, 0), dtype=np.bool_),
            (x0, y0, x0, y0),
        )

    local_x0 = int(xs.min())
    local_y0 = int(ys.min())
    local_x1 = int(xs.max()) + 1
    local_y1 = int(ys.max()) + 1

    cropped = mask[
        local_y0:local_y1,
        local_x0:local_x1,
    ]

    cropped_region = (
        x0 + local_x0,
        y0 + local_y0,
        x0 + local_x1,
        y0 + local_y1,
    )

    return cropped, cropped_region


def _validate_mask_region(
    mask: np.ndarray,
    region: tuple[int, int, int, int],
    image_shape: tuple[int, int] | None,
) -> None:
    """Validate consistency between mask and mask_region."""
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-dimensional, got shape={mask.shape}."
        )

    x0, y0, x1, y1 = region
    h, w = mask.shape

    if x1 - x0 != w or y1 - y0 != h:
        raise ValueError(
            "mask and mask_region are inconsistent: "
            f"mask.shape={mask.shape}, mask_region={region}."
        )

    if x0 < 0 or y0 < 0:
        raise ValueError(
            f"mask_region must have non-negative coordinates: {region}."
        )

    if image_shape is not None:
        image_h, image_w = image_shape

        if x1 > image_w or y1 > image_h:
            raise ValueError(
                f"mask_region={region} exceeds image_shape={image_shape}."
            )


def mask_morphology(
    instance: Instance2D,
    kernel_sizes: Sequence[float],
    ratio_morphology: bool = False,
    image_shape: tuple[int, int] | None = None,
) -> Instance2D:
    """
    Apply morphological operations to an Instance2D mask.

    The mask is stored only inside ``instance.mask_region``. Before each
    morphological operation, the mask region is temporarily expanded so that
    the result is equivalent to applying the operation to a full-image mask
    as much as possible.

    Args:
        instance:
            Instance2D containing the cropped binary mask.

        kernel_sizes:
            Kernel sizes in order of application.

            - Positive value: dilation
            - Negative value: erosion
            - Zero: no operation

            If ``ratio_morphology=False``, values are interpreted as pixels.

            If ``ratio_morphology=True``, values are interpreted as ratios
            relative to the average mask bounding-box size. For example,
            ``0.03`` means 3% of the average of mask width and height.

        ratio_morphology:
            Whether to interpret ``kernel_sizes`` as ratios of the mask size.

        image_shape:
            Optional ``(height, width)`` of the original image.
            Providing this is recommended because morphology can expand
            ``mask_region``. It also allows the expanded region to be clipped
            correctly at the image boundaries.

    Returns:
        A new Instance2D whose mask and mask_region contain the result of the
        morphological operations. ``box`` and other attributes are preserved.
    """
    if instance.mask is None:
        return instance

    if instance.mask_region is None:
        raise ValueError("instance.mask_region must be specified when instance.mask is not None.")

    mask = instance.mask.astype(np.bool_, copy=False)
    region = instance.mask_region

    _validate_mask_region(mask, region, image_shape)

    # Remove unnecessary False margins from the input first.
    mask, region = _crop_mask_to_nonzero(mask, region)

    # Empty mask: morphology cannot make it non-empty again.
    if mask.size == 0:
        return replace(
            instance,
            mask=mask,
            mask_region=region,
        )

    # Resolve ratio-based kernel sizes once from the original mask size,
    # matching the behavior of the original implementation.
    if ratio_morphology:
        h, w = mask.shape
        bbox_avg = (h + w) / 2.0

        resolved_kernel_sizes = [
            0
            if k == 0
            else (
                int(np.ceil(k * bbox_avg))
                if k > 0
                else -int(np.ceil(abs(k) * bbox_avg))
            )
            for k in kernel_sizes
        ]
    else:
        resolved_kernel_sizes = [int(k) for k in kernel_sizes]

    for kernel_size in resolved_kernel_sizes:
        if kernel_size == 0:
            continue

        size = abs(kernel_size)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )

        # cv2 morphology needs pixels surrounding the cropped mask.
        #
        # k // 2 pixels on each side are enough to cover the support of
        # the structuring element. For even kernels this slightly
        # over-expands one side, which is harmless.
        padding = size // 2

        mask, region = _expand_mask_region(
            mask=mask,
            region=region,
            padding=padding,
            image_shape=image_shape,
        )

        # OpenCV morphology does not reliably support np.bool_,
        # so use uint8 internally.
        mask_u8 = mask.astype(np.uint8)

        if kernel_size > 0:
            mask_u8 = cv2.dilate(
                mask_u8,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
        else:
            mask_u8 = cv2.erode(
                mask_u8,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
            )

        mask = mask_u8 > 0

        # Crop the mask again to minimize memory usage.
        mask, region = _crop_mask_to_nonzero(mask, region)

        # Once the mask becomes empty, further morphology cannot restore it.
        if mask.size == 0:
            break

    return replace(
        instance,
        mask=mask,
        mask_region=region,
    )
