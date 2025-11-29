from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.util import random_noise

from .adaptive_tv import adaptive_tv_denoise, edge_weight_map
from .regularizer_tv import regularizer_denoise

Array = np.ndarray


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
        self.image_noisy = random_noise(
            image=self.image_original, mode="gaussian", var=self.noise_variance
        )
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
        return gaussian(image=self.image_noisy, sigma=sigma)

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
        return denoise_tv_chambolle(image=self.image_noisy, weight=weight)

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
        os.makedirs(name=images_path, exist_ok=True)

        # Compute results
        img_gauss: np.ndarray = self.denoise_gaussian(sigma=sigma_gauss)
        images_tv: list[np.ndarray] = [self.denoise_tv(weight=w) for w in weights_tv]

        # --- First plot: comparison of main methods ---
        fig1: Figure
        fig1, axes1 = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
        ax = axes1.ravel()

        ax[0].imshow(self.image_original, cmap="gray")
        ax[0].set_title("Original")

        ax[1].imshow(self.image_noisy, cmap="gray")
        ax[1].set_title(f"Noisy (Gaussian noise, var={self.noise_variance})")

        ax[2].imshow(img_gauss, cmap="gray")
        ax[2].set_title(f"Gaussian Filter (sigma={sigma_gauss})")

        plot_index: int = min(1, len(weights_tv) - 1)
        img_tv: Array = images_tv[plot_index]
        ax[3].imshow(img_tv, cmap="gray")
        ax[3].set_title(f"TV Chambolle (λ={weights_tv[plot_index]})")

        for a in ax:
            a.axis("off")

        fig1.suptitle(t="Denoising Comparison: Gaussian vs. TV", fontsize=16, y=1.02)
        plt.tight_layout()
        save_path_1: str = os.path.join(images_path, "denoising_comparison.jpg")
        plt.savefig(save_path_1, bbox_inches="tight", dpi=150)

        fig2: Figure
        fig2, axes2 = plt.subplots(
            nrows=1, ncols=len(weights_tv), figsize=(4 * len(weights_tv), 4)
        )
        if len(weights_tv) == 1:
            axes2 = [axes2]

        for ax, img, w in zip(axes2, images_tv, weights_tv, strict=True):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"λ = {w}")
            ax.axis("off")

        fig2.suptitle(t="TV Denoising for Different λ Values", fontsize=16, y=1.02)
        plt.tight_layout()
        save_path_2: str = os.path.join(images_path, "tv_lambda_comparison.jpg")
        plt.savefig(save_path_2, bbox_inches="tight", dpi=150)

        print(f"Saved comparison plots to:\n  {save_path_1}\n  {save_path_2}")

    def compare_tv_lambda_sweep(
        self,
        images: list[np.ndarray],
        image_names: list[str],
        weights_tv: list[float],
        save_path: str,
        cmap: str = "gray",
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
        os.makedirs(name=save_path, exist_ok=True)

        for img, name in zip(images, image_names, strict=True):
            self.image_original = img
            self.image_noisy = self.add_gaussian_noise()

            results: list[Array] = [self.denoise_tv(weight=w) for w in weights_tv]

            ncols: int = 2 + len(weights_tv)
            fig: Figure
            fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
            ax = axes.ravel()

            ax[0].imshow(self.image_original, cmap=cmap)
            ax[0].set_title(f"{name} - Original")
            ax[1].imshow(self.image_noisy, cmap=cmap)
            ax[1].set_title(f"{name} - Noisy (var={self.noise_variance})")

            for idx, (w, res_img) in enumerate(
                iterable=zip(weights_tv, results, strict=True)
            ):
                ax[2 + idx].imshow(res_img, cmap=cmap)
                ax[2 + idx].set_title(f"λ = {w:.3f}")

            for a in ax:
                a.axis("off")

            fig.suptitle(t=f"TV Denoising λ-Sweep: {name}", fontsize=14, y=1.02)
            plt.tight_layout()
            filename: str = os.path.join(
                save_path, f"tv_sweep_{name.replace(' ', '_')}.jpg"
            )
            print(f"saving {filename} ...")
            plt.savefig(filename, bbox_inches="tight", dpi=150)

    def denoise_tv_adaptive(
        self,
        lambda_data: float = 0.2,
        eps: float = 1e-3,
        dt: float = 0.2,
        iters: int = 200,
        sigma_edge: float = 1.0,
        k_percentile: float = 90.0,
        beta: float = 2.0,
        w_map: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Denoising TV con adaptación espacial (w(x) y/o λ(x)).

        Si w_map no se pasa, se calcula automáticamente a partir de la imagen ruidosa.
        """
        if self.image_noisy is None:
            raise ValueError(
                "Primero genera la imagen ruidosa con add_gaussian_noise()."
            )
        return adaptive_tv_denoise(
            f_noisy=self.image_noisy,
            lambda_data=lambda_data,
            w_map=w_map,
            eps=eps,
            dt=dt,
            iters=iters,
            sigma_edge=sigma_edge,
            k_percentile=k_percentile,
            beta=beta,
        )

    def compare_tv_adaptive_sweep(
        self,
        save_path: str,
        name: str,
        lambdas: list[float] = (0.05, 0.1, 0.2),
        sigma_edge: float = 1.0,
        k_percentile: float = 90.0,
        beta: float = 2.0,
        iters: int = 200,
        dt: float = 0.2,
        eps: float = 1e-3,
        cmap: str = "gray",
    ) -> None:
        """
        Compara el método TV adaptativo para varios λ sobre la imagen actual.
        Guarda una figura: original | noisy | resultados para λ de la lista.
        """
        os.makedirs(name=save_path, exist_ok=True)
        if self.image_noisy is None:
            raise ValueError(
                "Primero genera la imagen ruidosa con add_gaussian_noise()."
            )

        # Un único w(x) consistente por imagen ruidosa:
        w_map: Array = edge_weight_map(
            f=self.image_noisy, sigma=sigma_edge, k_percentile=k_percentile, beta=beta
        )

        results: list[Array] = [
            adaptive_tv_denoise(
                f_noisy=self.image_noisy,
                lambda_data=lmb,
                w_map=w_map,
                eps=eps,
                dt=dt,
                iters=iters,
            )
            for lmb in lambdas
        ]

        ncols: int = 2 + len(lambdas)
        fig: Figure
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
        ax = axes.ravel()

        ax[0].imshow(self.image_original, cmap=cmap)
        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[1].imshow(self.image_noisy, cmap=cmap)
        ax[1].set_title("Noisy")
        ax[1].axis("off")

        for i, (lmb, img) in enumerate(iterable=zip(lambdas, results, strict=True)):
            ax[2 + i].imshow(img, cmap=cmap)
            ax[2 + i].set_title(rf"Adapt. TV, $\lambda={lmb}$")
            ax[2 + i].axis("off")

        fig.tight_layout()
        fig.suptitle(t=f"TV Adaptive Sweep: {name}", fontsize=14, y=1.02)
        plt.tight_layout()
        filename: str = os.path.join(
            save_path, f"tv_sweep_{name.replace(' ', '_')}.jpg"
        )
        print(f"saving {filename} ...")
        plt.savefig(filename, bbox_inches="tight", dpi=150)

    def denoise_regularizer(
        self,
        lambda_data: float = 0.05,
        alpha: float = 1.0,
        beta: float = 0.05,
        dt: float = 0.05,
        iters: int = 400,
    ) -> Array:
        if self.image_noisy is None:
            raise ValueError(
                "Primero genera la imagen ruidosa con add_gaussian_noise()."
            )

        u_denoised: Array | tuple[Array, list[Array]] = regularizer_denoise(
            f_noisy=self.image_noisy,
            lambda_data=lambda_data,
            alpha=alpha,
            beta=beta,
            dt=dt,
            iters=iters,
            clip=True,
            return_history=False,
        )
        assert isinstance(u_denoised, np.ndarray)
        return u_denoised

    def compare_regularizer_sweep(
        self,
        alphas: list[float],
        betas: list[float],
        save_path: str,
        name: str,
        lambda_data: float = 0.5,
        total_time: float = 10.0,
        safety: float = 0.7,
        cmap: str = "gray",
    ) -> None:

        def _stable_dt(
            lambda_data: float, alpha: float, beta: float, safety: float = 0.7
        ) -> float:
            s: float = lambda_data + 8.0 * alpha + 64.0 * beta
            return safety * 2.0 / s

        os.makedirs(name=save_path, exist_ok=True)
        assert len(alphas) == len(betas)

        results: list[Array] = []
        labels: list[str] = []

        for a, b in zip(alphas, betas, strict=True):
            dt: float = _stable_dt(
                lambda_data=lambda_data, alpha=a, beta=b, safety=safety
            )
            iters = int(total_time / dt)

            u_ho: Array = self.denoise_regularizer(
                lambda_data=lambda_data,
                alpha=a,
                beta=b,
                dt=dt,
                iters=iters,
            )
            results.append(u_ho)
            labels.append(f"α={a:.3f}, β={b:.4f}, dt={dt:.3f}, iters={iters}")

        ncols: int = 2 + len(results)  # original, noisy, y cada resultado
        fig: Figure
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        axes[0].imshow(self.image_original, cmap=cmap)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(self.image_noisy, cmap=cmap)
        axes[1].set_title("Noisy")
        axes[1].axis("off")

        for i, (img, lab) in enumerate(
            iterable=zip(results, labels, strict=True), start=2
        ):
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(lab)
            axes[i].axis("off")

        fig.tight_layout()
        fig.suptitle(t=f"TV Regularizer Sweep: {name}", fontsize=14, y=1.02)
        plt.tight_layout()
        filename: str = os.path.join(
            save_path, f"tv_sweep_{name.replace(' ', '_')}.jpg"
        )
        print(f"saving {filename} ...")
        plt.savefig(filename, bbox_inches="tight", dpi=150)
