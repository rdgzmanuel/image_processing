from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
from skimage.util import random_noise


class ImageDenoisingComparison:
    """
    Class for comparing different image denoising methods.

    Includes:
    - Classical Gaussian filtering
    - Variational denoising by Total Variation (TV Chambolle)

    Attributes
    ----------
    image_original : np.ndarray
        Original clean image.
    image_noisy : np.ndarray | None
        Noisy version of the original image.
    """

    def __init__(self, image: np.ndarray) -> None:
        """
        Initialize the class with a base image.

        Parameters
        ----------
        image : np.ndarray
            Original image as float in range [0, 1].
        """
        self.image_original: np.ndarray = image
        self.image_noisy: np.ndarray | None = None

    def add_gaussian_noise(self, var: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to the original image.

        Parameters
        ----------
        var : float, optional
            Variance of the Gaussian noise, by default 0.01.

        Returns
        -------
        np.ndarray
            Noisy image.
        """
        self.image_noisy = random_noise(self.image_original, mode="gaussian", var=var)
        return self.image_noisy

    def denoise_gaussian(self, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian filtering to denoise the image.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of the Gaussian kernel, by default 1.0.

        Returns
        -------
        np.ndarray
            Denoised image using Gaussian filter.
        """
        if self.image_noisy is None:
            raise ValueError("You must first generate the noisy image.")
        return gaussian(self.image_noisy, sigma=sigma)

    def denoise_tv(self, weight: float = 0.1) -> np.ndarray:
        """
        Apply Total Variation (TV Chambolle) denoising.

        Parameters
        ----------
        weight : float, optional
            Regularization weight. Larger values produce smoother images.
            Default is 0.1.

        Returns
        -------
        np.ndarray
            Denoised image using TV Chambolle.
        """
        if self.image_noisy is None:
            raise ValueError("You must first generate the noisy image.")
        return denoise_tv_chambolle(self.image_noisy, weight=weight)

    def compare_methods(
        self,
        sigma_gauss: float = 1.0,
        weight_tv: float = 0.1,
    ) -> None:
        """
        Compare Gaussian and TV denoising visually.

        Parameters
        ----------
        sigma_gauss : float, optional
            Sigma parameter for Gaussian filtering.
        weight_tv : float, optional
            Lambda (regularization) parameter for TV Chambolle.
        """
        img_gauss = self.denoise_gaussian(sigma=sigma_gauss)
        img_tv = self.denoise_tv(weight=weight_tv)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        ax = axes.ravel()

        ax[0].imshow(self.image_original, cmap="gray")
        ax[0].set_title("Original")
        ax[1].imshow(self.image_noisy, cmap="gray")
        ax[1].set_title("Noisy (Gaussian)")
        ax[2].imshow(img_gauss, cmap="gray")
        ax[2].set_title(f"Gaussian Filter (σ={sigma_gauss})")
        ax[3].imshow(img_tv, cmap="gray")
        ax[3].set_title(f"TV Chambolle (λ={weight_tv})")

        for a in ax:
            a.axis("off")

        plt.tight_layout()
        plt.show()

