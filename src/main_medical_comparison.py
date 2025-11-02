from src.utils import download_medical_images
from src.denoising_comparison import ImageDenoisingComparison

def main(images_folder: str) -> None:
    images, names = download_medical_images(images_folder)
    denoiser = ImageDenoisingComparison(images[0], noise_variance=0.025)  # temporary init

    weights_tv = [0.05, 0.1, 0.2]
    save_dir = "results/medical_denoising"

    denoiser.compare_tv_lambda_sweep(
        images=images,
        image_names=names,
        weights_tv=weights_tv,
        save_path=save_dir,
    )

if __name__ == "__main__":
    images_folder: str = "images"
    main(images_folder)
