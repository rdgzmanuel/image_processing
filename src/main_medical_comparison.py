from src.utils import download_medical_images
from src.denoising_comparison import ImageDenoisingComparison
import os
import numpy as np

def main(images_folder: str) -> None:
    # Download or load medical images (MRI, X-Ray, etc.)
    images, names = download_medical_images(images_folder)
    denoiser = ImageDenoisingComparison(images[0])  # temporary initialization

    # Run classical TV denoising comparison for several lambda values
    weights_tv = [0.05, 0.1, 0.2]
    save_dir = "results/medical_denoising"
    denoiser.compare_tv_lambda_sweep(
        images=images,
        image_names=names,
        weights_tv=weights_tv,
        save_path=save_dir,
    )

    # Run Adaptive TV denoising comparison (spatially weighted version)
    save_dir_adapt = "results/medical_denoising_adaptive"
    os.makedirs(save_dir_adapt, exist_ok=True)

    # Apply adaptive TV to each medical image and save results
    for img, name in zip(images, names):
        denoiser.image_original = img
        denoiser.add_gaussian_noise()
        denoiser.compare_tv_adaptive_sweep(
            save_path=os.path.join(save_dir_adapt, name),
            lambdas=[0.05, 0.1, 0.2],
            sigma_edge=1.0,
            k_percentile=90.0,
            beta=2.0,
            iters=250,
            dt=0.2,
            title_suffix=name,
        )

if __name__ == "__main__":
    # Define source folder and run the full medical denoising workflow
    images_folder = "images"
    main(images_folder)
