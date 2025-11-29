from __future__ import annotations

import numpy as np

Array = np.ndarray


def _neumann_pad(u: Array) -> Array:
    """Extiende la imagen con condición de Neumann (modo 'edge')."""
    return np.pad(array=u, pad_width=((1, 1), (1, 1)), mode="edge")


def _laplacian(u: Array) -> Array:
    """Laplaciano discreto con stencil 5-puntos y bordes de Neumann."""
    up: Array = _neumann_pad(u=u)
    # kernel:
    #  0  1  0
    #  1 -4  1
    #  0  1  0
    center: Array = up[1:-1, 1:-1]
    north: Array = up[:-2, 1:-1]
    south: Array = up[2:, 1:-1]
    west: Array = up[1:-1, :-2]
    east: Array = up[1:-1, 2:]
    lap = north + south + west + east - 4.0 * center
    return lap


def regularizer_denoise(
    f_noisy: Array,
    lambda_data: float = 0.2,  # <-- NEW
    alpha: float = 1.0,
    beta: float = 0.02,
    dt: float = 0.2,
    iters: int = 400,
    clip: bool = True,
    return_history: bool = False,
) -> Array | tuple[Array, list[Array]]:
    # float64 for stability
    u: Array = f_noisy.astype(dtype=np.float64).copy()
    f: Array = f_noisy.astype(dtype=np.float64)

    history: list[Array] = []
    if return_history:
        history.append(u.copy())

    for _ in range(iters):
        lap_u: Array = _laplacian(u=u)
        bih_u: Array = _laplacian(u=lap_u)

        # ∂_t u = λ(f - u) + α Δu - β Δ²u
        du: Array = lambda_data * (f - u) + alpha * lap_u - beta * bih_u
        u = u + dt * du

        if clip:
            u = np.clip(a=u, a_min=0.0, a_max=1.0)

        if return_history:
            history.append(u.copy())

    return (u, history) if return_history else u
