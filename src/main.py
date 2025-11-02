from utils import load_image
from denoising_comparison import ImageDenoisingComparison
import numpy as np

def main(path: str) -> None:
    image: np.ndarray = load_image(path, as_gray=True)
    denoiser = ImageDenoisingComparison(image)
    denoiser.add_gaussian_noise(var=0.01)
    denoiser.compare_methods(sigma_gauss=1.0, weight_tv=0.15)

if __name__ == "__main__":
    path: str = "images/sample_noisy.jpg"
    gaussian_variance: float = 0.01
    sigma_gaussian: float = 1.0
    tv_weight: float = 0.15

    main(path, gaussian_variance, sigma_gaussian, tv_weight)
