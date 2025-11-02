from __future__ import annotations

import numpy as np
from skimage import io, color, img_as_float
from pathlib import Path


def load_image(
    path: str | Path,
    as_gray: bool = True,
) -> np.ndarray:
    """
    Load an image from disk and return it as a normalized NumPy array.

    Parameters
    ----------
    path : str | Path
        Path to the image file on disk.
    as_gray : bool, optional
        If True, convert the image to grayscale (default: True).

    Returns
    -------
    np.ndarray
        Image array with float values in [0, 1].

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist or cannot be opened.
    ValueError
        If the image cannot be loaded or has an unsupported format.

    Examples
    --------
    >>> from utils import load_image
    >>> img = load_image("data/photo.jpg", as_gray=True)
    >>> img.shape
    (512, 512)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {path}")

    try:
        image = io.imread(path)
    except Exception as e:
        raise ValueError(f"Could not read image at {path}: {e}")

    if as_gray and image.ndim == 3:
        image = color.rgb2gray(image)

    return img_as_float(image)
