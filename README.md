# Image Processing via Total Variation

This project implements image denoising techniques based on the Calculus of Variations. It explores the classical Rudin–Osher–Fatemi (ROF) model, a Spatially Adaptive TV method for edge preservation, and Higher Order Regularizers to mitigate staircasing effects.

The implementation focuses on the balance between noise reduction and structural preservation, with specific applications to medical imaging (MRI and X-ray).

## Project Overview

The project follows the theoretical formulation presented in the report *“Image Processing via Calculus of Variations”*. It demonstrates how calculus of variations principles, historically used for problems like the Brachistochrone, can be applied to modern image processing.

**Key Methods:**

* **Gaussian Smoothing:** Baseline linear filtering for comparison.
* **Rudin–Osher–Fatemi (ROF):** Minimizes total variation to remove noise while maintaining sharp edges.
* **Spatially Adaptive TV:** Introduces a spatial weighting function $w(x)$ to inhibit diffusion near edges.
* **Higher Order Regularization:** Penalizes the Laplacian to produce smoother gradients and avoid piecewise constant artifacts.

## Installation

The project requires Python 3 and the following dependencies:

```bash
pip install numpy matplotlib scikit-image scipy
```

## Folder Structure

```text
image_processing/
│
├── src/
│   ├── utils.py                     # Utility functions (image loading, dataset handling)
│   ├── denoising_comparison.py      # Main class for denoising (Gaussian, ROF, Adaptive)
│   ├── adaptive_tv.py               # Implementation of the Spatially Adaptive TV model
│   ├── regularizer_tv.py            # Implementation of the Higher Order TV model
|   ├── main_denoising_comparison.py # Script for standard image comparison
|   ├── main_medical_comparison.py   # Script for medical dataset comparison
│
├── images/                          # Input images (e.g., MRI, X-ray, standard test images)
│
├── written_report/
│   ├── image_processing_english.pdf   # Theoretical report (English)
│   ├── image_processing_spanish.pdf   # Theoretical report (Spanish)
│
├── requirements.txt                 # Libraries dependencies
└── README.md
```

## Usage

### Standard Image Denoising

To run the comparison on a standard test image (e.g., `iniesta.jpg`):

```bash
python main_denoising_comparison.py
```

This script:
1.  Adds Gaussian noise to the input.
2.  Applies Gaussian filter and ROF denoising with varying $\lambda$.
3.  Applies Spatially Adaptive TV.
4.  Saves results to `images/results_adaptive/`.

### Medical Image Denoising

To run the analysis on medical datasets:

```bash
python main_medical_comparison.py
```

This script:
1.  Loads/downloads MRI (brain) and X-ray (lungs) datasets.
2.  Compares Classical TV vs. Adaptive TV.
3.  Saves results to `results/medical_denoising/`.

## Mathematical Formulation

### 1. Classical ROF Model
The model minimizes the following functional, balancing data fidelity ($L^2$ norm) and total variation regularization:

$$
J[u] = \frac{1}{2} \int_{\Omega} (u - f)^2 \, dx + \lambda \int_{\Omega} |\nabla u| \, dx
$$

### 2. Spatially Adaptive TV
To better preserve edges, we introduce a weight $w(x)$ that approaches 0 at edges and 1 in flat regions:

$$
E(u) = \int_{\Omega} w(x)\, |\nabla u| \, dx + \frac{1}{2} \int_{\Omega} \lambda(x)\, (u - f)^2 \, dx
$$

Where the weight is defined by the gradient of the smoothed image $G_\sigma * f$:

$$
w(x) = \exp\left(-\left(\frac{|\nabla (G_\sigma * f)|}{k}\right)^\beta\right)
$$

### 3. Higher Order Regularization
To address the "staircasing" effect (blocky artifacts) inherent in TV, we include a Laplacian term:

$$
J[u] = \frac{1}{2}\int_\Omega (u-f)^2\,dx + \frac{\alpha}{2}\int_\Omega |\nabla u|^2\,dx + \frac{\beta}{2}\int_\Omega (\Delta u)^2\,dx
$$

## References

* L. Rudin, S. Osher, and E. Fatemi, "Nonlinear total variation based noise removal algorithms," *Physica D*, 1992.
* Datasets: `scikit-image` library and Kaggle Chest X-Ray Images (Pneumonia).

## Authors

Joaquín Mir Macías, Miguel Montes Lorenzo, Manuel Rodríguez Villegas