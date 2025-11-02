from __future__ import annotations

import numpy as np
from skimage import data, io, color, img_as_float
from pathlib import Path
from typing import Tuple, List


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


def download_medical_images(images_folder: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load medical images (2 MRI and 2 chest X-rays) for denoising experiments.
    
    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        A tuple containing:
        - A list of 4 images as NumPy arrays (float, range [0, 1]).
        - A list of corresponding descriptive names for each image.
    
    Notes
    -----
    - The MRI images are taken from the scikit-image sample data (brain MRIs).
    - The chest X-rays are loaded from ./images/ folder (bacteria_1.jpeg and virus_1.jpeg), downloaded from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download
    - All images are automatically converted to grayscale and normalized.
    """
    from pathlib import Path
    
    images: List[np.ndarray] = []
    names: List[str] = []
    
    # --- MRI IMAGES ---
    brain_volume = data.brain()
    images.extend([img_as_float(brain_volume[5]), img_as_float(brain_volume[7])])
    names.extend(["MRI_Brain_1", "MRI_Brain_2"])
    
    # --- CHEST X-RAY IMAGES from images folder ---
    xray_files = ["bacteria_1.jpeg", "virus_1.jpeg"]
    images_folder = Path(f"./{images_folder}")
    
    for xray_file in xray_files:
        xray_path = images_folder / xray_file
        
        try:
            xray_img = io.imread(xray_path)
            
            if xray_img.ndim == 3:
                xray_img = color.rgb2gray(xray_img)
            
            xray_img = img_as_float(xray_img)
            
            images.append(xray_img)
            name = xray_file.replace('.jpeg', '').replace('_', ' ').title()
            names.append(f"Xray_{name}")
            
        except Exception as e:
            print(f"Error loading {xray_path}: {e}")
            raise
    
    return images, names