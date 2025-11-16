from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple

Array = np.ndarray

def _neumann_pad(u: Array) -> Array:
    """Extiende con condición de Neumann (derivada normal nula) mediante 'edge'."""
    return np.pad(u, ((1, 1), (1, 1)), mode="edge")

def _forward_grad(u: Array) -> Tuple[Array, Array]:
    """Gradientes hacia delante con Neumann implícito en el borde."""
    up = _neumann_pad(u)
    ux = up[1:-1, 2:] - up[1:-1, 1:-1]
    uy = up[2:, 1:-1] - up[1:-1, 1:-1]
    return ux, uy

def _backward_div(px: Array, py: Array) -> Array:
    """Divergencia (diferencias hacia atrás) con Neumann implícito en el borde."""
    pxp = _neumann_pad(px)
    pyp = _neumann_pad(py)
    divx = pxp[1:-1, 1:-1] - pxp[1:-1, :-2]
    divy = pyp[1:-1, 1:-1] - pyp[:-2, 1:-1]
    return divx + divy

def edge_weight_map(
    f: Array,
    sigma: float = 1.0,
    k_percentile: float = 90.0,
    beta: float = 2.0,
) -> Array:
    """
    Mapa de pesos w(x) \in [0,1] que atenúa la regularización cerca de bordes.
    w(x) = exp(-( |∇(G_sigma * f)| / k )^beta)

    Parámetros:
      - f: imagen de referencia en [0,1] (mejor la ruidosa suavizada).
      - sigma: suavizado previo para robustez del gradiente.
      - k_percentile: percentil para escalar el umbral k automáticamente.
      - beta: controla cuán rápido cae w en bordes (típicamente 1–4).
    """
    f_s = gaussian_filter(f, sigma=sigma) if sigma > 0 else f
    gx, gy = _forward_grad(f_s)
    gmag = np.sqrt(gx**2 + gy**2)
    k = np.percentile(gmag, k_percentile) + 1e-12  # evitar 0
    w = np.exp(- (gmag / k) ** beta)
    # normalizar a [0,1] por seguridad numérica
    w = np.clip(w, 0.0, 1.0)
    return w

def adaptive_tv_denoise(
    f_noisy: Array,
    lambda_data: float | Array = 0.2,
    w_map: Optional[Array] = None,
    eps: float = 1e-3,
    dt: float = 0.2,
    iters: int = 200,
    sigma_edge: float = 1.0,
    k_percentile: float = 90.0,
    beta: float = 2.0,
    clip: bool = True,
    return_history: bool = False,
) -> Array | Tuple[Array, list[Array]]:
    """
    Denoising ROF *adaptativo espacialmente*:
      Min_u ∫ w(x) |∇u| dx + (1/2) ∫ λ(x) (u - f)^2 dx

    PDE de gradiente descendente:
      ∂_t u = div( w * ∇u / sqrt(|∇u|^2 + eps^2) ) + λ(x) (f - u)

    Parámetros:
      - f_noisy: imagen ruidosa en [0,1].
      - lambda_data: escalar o mapa λ(x) ≥ 0 (fidelidad a datos).
      - w_map: si se proporciona, se usa; en caso contrario se calcula con edge_weight_map.
      - eps: regularización en la norma para estabilizar |∇u|.
      - dt: paso temporal (estable ~0.1–0.25).
      - iters: iteraciones del descenso.
      - sigma_edge, k_percentile, beta: parámetros para construir w si no se pasa.
      - clip: recorta a [0,1] cada iteración.
      - return_history: si True, devuelve también una lista con snapshots.

    Devuelve:
      - u (y opcionalmente el historial si return_history=True).
    """
    u = f_noisy.astype(np.float64).copy()
    if w_map is None:
        w_map = edge_weight_map(
            f_noisy, sigma=sigma_edge, k_percentile=k_percentile, beta=beta
        )
    if np.isscalar(lambda_data):
        lam = float(lambda_data)
        def data_term(u: Array) -> Array:
            return lam * (f_noisy - u)
    else:
        lamap = lambda_data.astype(np.float64)
        def data_term(u: Array) -> Array:
            return lamap * (f_noisy - u)

    history: list[Array] = []

    for _ in range(iters):
        ux, uy = _forward_grad(u)
        grad_norm = np.sqrt(ux**2 + uy**2 + eps**2)

        px = w_map * (ux / grad_norm)
        py = w_map * (uy / grad_norm)

        div_p = _backward_div(px, py)

        u = u + dt * (div_p + data_term(u))
        if clip:
            u = np.clip(u, 0.0, 1.0)

        if return_history:
            history.append(u.copy())

    return (u, history) if return_history else u
