from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import gaussian
from skimage.util import random_noise
import os


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

    def __init__(self, image: np.ndarray, noise_variance: float = 0.01) -> None:
        """
        Initialize the class with a base image.

        Parameters
        ----------
        image : np.ndarray
            Original image as float in range [0, 1].
        noise_variance : float, optional
            Variance of Gaussian noise.
        """
        self.image_original: np.ndarray = image
        self.image_noisy: np.ndarray | None = None
        self.noise_variance: float = noise_variance

    def add_gaussian_noise(self) -> np.ndarray:
        """
        Add Gaussian noise to the original image.

        Returns
        -------
        np.ndarray
            Noisy image.
        """
        self.image_noisy = random_noise(self.image_original, mode="gaussian", var=self.noise_variance)
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
        images_path: str,
        sigma_gauss: float = 1.0,
        weights_tv: list[float] = [0.05, 0.1, 0.15, 0.25],
    ) -> None:
        """
        Compare Gaussian and TV denoising visually.

        Parameters
        ----------
        images_path : str
            Directory where plots will be saved.
        sigma_gauss : float, optional
            Sigma parameter for Gaussian filtering.
        weights_tv : list[float], optional
            List of regularization weights (λ) for TV Chambolle.
        """
        os.makedirs(images_path, exist_ok=True)

        # Compute results
        img_gauss: np.ndarray = self.denoise_gaussian(sigma=sigma_gauss)
        images_tv: list[np.ndarray] = [self.denoise_tv(weight=w) for w in weights_tv]

        # --- First plot: comparison of main methods ---
        fig1, axes1 = plt.subplots(1, 4, figsize=(16, 5))
        ax = axes1.ravel()

        ax[0].imshow(self.image_original, cmap="gray")
        ax[0].set_title("Original")

        ax[1].imshow(self.image_noisy, cmap="gray")
        ax[1].set_title(f"Noisy (Gaussian noise, var={self.noise_variance})")

        ax[2].imshow(img_gauss, cmap="gray")
        ax[2].set_title(f"Gaussian Filter (σ={sigma_gauss})")

        plot_index: int = min(1, len(weights_tv) - 1)
        img_tv = images_tv[plot_index]
        ax[3].imshow(img_tv, cmap="gray")
        ax[3].set_title(f"TV Chambolle (λ={weights_tv[plot_index]})")

        for a in ax:
            a.axis("off")

        fig1.suptitle("Denoising Comparison: Gaussian vs. TV", fontsize=16, y=1.02)
        plt.tight_layout()
        save_path_1 = os.path.join(images_path, "denoising_comparison.jpg")
        plt.savefig(save_path_1, bbox_inches="tight", dpi=150)

        fig2, axes2 = plt.subplots(1, len(weights_tv), figsize=(4 * len(weights_tv), 4))
        if len(weights_tv) == 1:
            axes2 = [axes2]

        for ax, img, w in zip(axes2, images_tv, weights_tv):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"λ = {w}")
            ax.axis("off")

        fig2.suptitle("TV Denoising for Different λ Values", fontsize=16, y=1.02)
        plt.tight_layout()
        save_path_2 = os.path.join(images_path, "tv_lambda_comparison.jpg")
        plt.savefig(save_path_2, bbox_inches="tight", dpi=150)

        print(f"Saved comparison plots to:\n  {save_path_1}\n  {save_path_2}")


    def compare_tv_lambda_sweep(
        self,
        images: list[np.ndarray],
        image_names: list[str],
        weights_tv: list[float],
        save_path: str,
        cmap: str = "gray"
    ) -> None:
        """
        For each image in `images`, apply TV denoising for each weight in `weights_tv`,
        then create and save a plot comparing: original → noisy → denoised for each λ.

        Parameters
        ----------
        images : list[np.ndarray]
            List of original clean images.
        image_names : list[str]
            Corresponding names/descriptions for the images (for titles/filenames).
        weights_tv : list[float]
            List of λ (regularisation weights) for the TV denoising.
        save_path : str
            Directory path where result plots will be saved.
        cmap : str, optional
            Colormap for plotting images (default “gray”).
        """
        os.makedirs(save_path, exist_ok=True)

        for img, name in zip(images, image_names):
            self.image_original = img
            self.image_noisy = self.add_gaussian_noise()

            results = [ self.denoise_tv(weight=w) for w in weights_tv ]

            ncols = 2 + len(weights_tv)
            fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
            ax = axes.ravel()

            ax[0].imshow(self.image_original, cmap=cmap)
            ax[0].set_title(f"{name} – Original")
            ax[1].imshow(self.image_noisy, cmap=cmap)
            ax[1].set_title(f"{name} – Noisy (var={self.noise_variance})")

            for idx, (w, res_img) in enumerate(zip(weights_tv, results)):
                ax[2 + idx].imshow(res_img, cmap=cmap)
                ax[2 + idx].set_title(f"λ = {w:.3f}")

            for a in ax:
                a.axis("off")

            fig.suptitle(f"TV Denoising λ-Sweep: {name}", fontsize=14, y=1.02)
            plt.tight_layout()
            filename = os.path.join(save_path, f"tv_sweep_{name.replace(' ', '_')}.jpg")
            plt.savefig(filename, bbox_inches="tight", dpi=150)
