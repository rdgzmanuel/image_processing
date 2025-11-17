import os

import numpy as np

from src.denoising_comparison import ImageDenoisingComparison
from src.utils import load_image


def main(
    path: str,
    images_path: str,
    tv_weights: list[float],
    noise_variance: float,
    sigma_gaussian: float = 1.0,
) -> None:
    # Load input image
    image: np.ndarray = load_image(path=path, as_gray=False)

    # Initialize denoising object with the image and noise level
    denoiser = ImageDenoisingComparison(image=image, noise_variance=noise_variance)

    # Add Gaussian noise to the original image
    denoiser.add_gaussian_noise()

    # Run classic denoising methods comparison (Gaussian vs. TV)
    denoiser.compare_methods(
        images_path=images_path, sigma_gauss=sigma_gaussian, weights_tv=tv_weights
    )

    # Run Adaptive TV method comparison (spatially varying regularization)
    denoiser.compare_tv_adaptive_sweep(
        save_path=os.path.join(images_path, "results_adaptive"),
        lambdas=[0.05, 0.1, 0.2],
        sigma_edge=1.0,
        k_percentile=90.0,
        beta=2.0,
        iters=250,
        dt=0.2,
        title_suffix=f"(sigma_edge=1.0, k={90}p, β=2)",
    )


if __name__ == "__main__":
    # Define paths and parameters
    images_path = "images"
    path: str = os.path.join(images_path, "iniesta.jpg")
    noise_variance: float = 0.15
    sigma_gaussian: float = 0.25
    tv_weights: list[float] = [0.01, 0.05, 0.1, 0.15]

    # Execute full denoising pipeline
    main(
        path=path,
        images_path=images_path,
        tv_weights=tv_weights,
        noise_variance=noise_variance,
        sigma_gaussian=sigma_gaussian,
    )
