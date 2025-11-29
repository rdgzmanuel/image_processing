import os

import numpy as np

from src.denoising_comparison import ImageDenoisingComparison
from src.utils import download_medical_images

Array = np.ndarray


def main(images_folder: str) -> None:
    # Download or load medical images (MRI, X-Ray, etc.)
    images: list[Array]
    names: list[str]
    images, names = download_medical_images(images_folder=images_folder)
    denoiser = ImageDenoisingComparison(image=images[0])  # temporary initialization

    # Run classical TV denoising comparison for several lambda values
    weights_tv: list[float] = [0.05, 0.1, 0.2]
    save_dir = "results/medical_denoising/standard"
    print("running experiment compare_tv_lambda_sweep")
    denoiser.compare_tv_lambda_sweep(
        images=images,
        image_names=names,
        weights_tv=weights_tv,
        save_path=save_dir,
    )

    # Run Adaptive TV denoising comparison (spatially weighted version)
    save_dir_adapt = "results/medical_denoising/adaptive"
    os.makedirs(name=save_dir_adapt, exist_ok=True)
    print("running experiment compare_tv_adaptive_sweep")
    # Apply adaptive TV to each medical image and save results
    for img, name in zip(images, names, strict=True):
        denoiser.image_original = img
        denoiser.add_gaussian_noise()
        denoiser.compare_tv_adaptive_sweep(
            save_path=save_dir_adapt,
            name=name,
            lambdas=[0.05, 0.1, 0.2],
            sigma_edge=1.0,
            k_percentile=90.0,
            beta=2.0,
            iters=250,
            dt=0.2,
        )

    save_dir_ho = "results/medical_denoising/regularizer"
    os.makedirs(name=save_dir_ho, exist_ok=True)
    print("running experiment compare_tv_regularizer_sweep")
    for img, name in zip(images, names, strict=True):
        denoiser = ImageDenoisingComparison(image=img, noise_variance=0.01)
        denoiser.add_gaussian_noise()

        alphas: list[float] = [0.05, 0.1, 0.15]  # [0.2, 0.4, 0.6]
        ratio: float = 0.1
        betas: list[float] = [ratio * a for a in alphas]

        denoiser.compare_regularizer_sweep(
            alphas=alphas,
            betas=betas,
            lambda_data=0.1,  # 0.03,
            total_time=300.0,
            safety=0.05,  # 0.001,
            save_path=save_dir_ho,
            name=name,
        )


if __name__ == "__main__":
    # Define source folder and run the full medical denoising workflow
    images_folder = "images"
    main(images_folder=images_folder)
