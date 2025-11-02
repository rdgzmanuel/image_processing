from src.utils import load_image
from src.denoising_comparison import ImageDenoisingComparison
import numpy as np
import os

def main(path: str, images_path: str, tv_weights: list[float], noise_variance, sigma_gaussian: float = 1.0) -> None:
    image: np.ndarray = load_image(path, as_gray=False)
    denoiser = ImageDenoisingComparison(image, noise_variance)
    denoiser.add_gaussian_noise()
    denoiser.compare_methods(images_path, sigma_gauss=sigma_gaussian, weights_tv=tv_weights)

if __name__ == "__main__":
    images_path: str = "images"
    image_name: str = "iniesta.jpg"
    path: str = os.path.join(images_path, image_name)
    noise_variance: float = 0.15  # 0.01
    sigma_gaussian: float = 0.25      # 1.0
    tv_weights: list[float] = [0.01, 0.05, 0.1, 0.15]       # 0.15

    main(path, images_path, tv_weights, noise_variance, sigma_gaussian)
