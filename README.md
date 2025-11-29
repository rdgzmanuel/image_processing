# Image Denoising using Total Variation and Spatially Adaptive Models

This project implements image denoising techniques based on Total Variation (TV), including the classical Rudin–Osher–Fatemi (ROF) model and an extended Spatially Adaptive TV method for improved edge preservation.  
It also demonstrates applications on medical images (MRI brain scans and X-ray images).

## Project Overview

The project is inspired by variational methods in image processing and follows the theoretical formulation presented in the document *“De la Braquistócrona al Procesamiento de Imágenes”*.  
It shows how variational calculus principles can be applied to remove noise from images while maintaining important structures and edges.

Implemented methods:

- Gaussian smoothing (baseline denoising)
- Classical Total Variation (ROF) model
- Spatially Adaptive TV model (new contribution)
  - Uses a spatial weighting function w(x) to control local smoothing.
  - Reduces diffusion near edges while smoothing flat regions.

## Folder Structure

```bash
image_processing/
│
├── src/
│ ├── utils.py # Utility functions (image loading, dataset handling)
│ ├── denoising_comparison.py # Main class with denoising methods (Gaussian, ROF, Adaptive TV)
│ ├── adaptive_tv.py # Implementation of the spatially adaptive TV model
│ ├── regularizer_tv.py # Implementation of the high order TV model (regularizer)
│
├── images/ # Input images (e.g., MRI, X-ray, example photos)
│
├── results/ # Output images and comparison figures
│
├── main_denoising_comparison.py # Runs comparison on a standard image (e.g., iniesta.jpg)
├── main_medical_comparison.py # Runs denoising on medical datasets
│
├── Branquistócrona al Procesamiento de Imagen.pdf # Project report (theoretical background)
└── README.md
```

## How to Run

### 1. Standard Denoising Comparison

Run the comparison on a regular image (e.g., iniesta.jpg):

```python
python main_denoising_comparison.py
```

This will:

- Load the image and add Gaussian noise.
- Apply Gaussian filter and TV (ROF) denoising for different λ values.
- Apply Spatially Adaptive TV denoising for comparison.
- Save the output figures under `images/results_adaptive/`.

### 2. Medical Image Denoising

Run the adaptive denoising pipeline on medical images:

```python
python main_medical_comparison.py
```

This script:

- Downloads or loads MRI and X-ray images.
- Compares classical TV and Adaptive TV for multiple λ values.
- Saves results under `results/medical_denoising/` and `results/medical_denoising_adaptive/`.

## Adaptive TV Model Summary

The Spatially Adaptive TV model minimizes the following functional:

<!-- **E(u) = ∫Ω w(x)·|∇u| dx + (1/2) ∫Ω λ(x)·(u − f)² dx** -->
$$
E(u) = \int_{\Omega} w(x)\, |\nabla u| \, dx
\;+\; \frac{1}{2} \int_{\Omega} \lambda(x)\, (u - f)^2 \, dx
$$

where:

- **w(x)** = exp(−(|∇(Gσ * f)| / k)^β) controls diffusion near edges  
- **λ(x)** adjusts data fidelity (often constant)

This formulation smooths homogeneous regions while preserving sharp boundaries — particularly useful for medical imaging.

## Example Results

- MRI Brain Scan: preserves tissue boundaries while denoising.
- X-Ray (lungs): effectively removes noise without losing edge details.
- Standard Image (Lena/Iniesta): smoother textures, sharp edges.

Each method produces side-by-side comparisons with different λ values.

## References

- L. Rudin, S. Osher, and E. Fatemi, “Nonlinear total variation based noise removal algorithms,” Physica D, 1992.  
- Theoretical background in the project report *“De la Braquistócrona al Procesamiento de Imágenes”*.  
- Dataset sources:
  - scikit-image sample datasets.
  - Kaggle Chest X-Ray Images (Pneumonia) dataset.

## Author Credits

Developed by Joaquín Mir Macías, Miguel Montes Lorenzo, and Manuel Rodríguez Villegas
