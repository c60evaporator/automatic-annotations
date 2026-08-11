import numpy as np
import cv2

def mask_morphology(mask: np.ndarray, 
                    kernel_sizes: list[int] = [3, -5],
                    ratio_morphology: bool = False,) -> np.ndarray:
    """
    Apply morphological operations to a binary mask.

    Args:
        mask (np.ndarray): HxW binary mask array.
        kernel_sizes (list[int]): List of kernel sizes sorted in order of application. Positive size represents dilation, whereas negative size represents erosion.
        ratio_morphology (bool): Whether to apply ratio-based morphology. If True, the applied kernel sizes are computed by multiplying `kernel_sizes` by the average of the mask bounding box height and width.

    Returns:
        np.ndarray: HxW binary mask after morphological operations.
    """
    # Calculate the kernel sizes based on the average bounding box size if ratio_morphology is True
    if ratio_morphology:
        # Calculate bounding rectangle of the mask
        rows = mask.any(axis=1)  # (H,) If a row has any True value, it is part of the mask
        cols = mask.any(axis=0)  # (W,) If a column has any True value, it is part of the mask
        nonempty = rows.any()  # If the mask has any True value, it is non-empty
        y0 = rows.argmax()
        y1 = rows.shape[0] - np.flip(rows, axis=0).argmax()
        x0 = cols.argmax()
        x1 = cols.shape[0] - np.flip(cols, axis=0).argmax()

        # Adjust kernel sizes based on the average bounding box size.
        bbox_avg = (y1 - y0 + x1 - x0) / 2
        kernel_sizes = [
            0 if k == 0 else (int(np.ceil(k * bbox_avg)) if k > 0 else -int(np.ceil(abs(k * bbox_avg))))
            for k in kernel_sizes
        ]

    for kernel_size in kernel_sizes:
        if kernel_size == 0:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(kernel_size), abs(kernel_size)))
        if kernel_size > 0:
            mask = cv2.dilate(mask, kernel)
        else:
            mask = cv2.erode(mask, kernel)
    return mask
