"""
fn_eval.py — Funciones de evaluación espacial y armónica del UNetDenoiser.

Métricas implementadas
-----------------------
Dominio espacial:
  · RMSE y bias sobre la región limpia (|b| > lat_cut_deg).
  · RMS por bandas de latitud de anchura configurable (réplica de la Figura 4 de Wang et al. 2022).
  · Correlación de Pearson entre predicción y target.


Dominio armónico:
  · Pseudo-C_ℓ con corrección f_sky de primer orden.
  · Ratio de espectros cl_pred / cl_target: desviación de amplitud por escala angular.
  · Espectro del residuo cl_residual / cl_target: ruido irreducible por multipolo.
  · Desviación σ_CNN vs varianza cósmica (réplica de la Figura 7 de
    Wang et al. 2022): requiere múltiples muestras del test set.
"""

import numpy as np
import healpy as hp

# ==============================================================================    
# Máscaras
# ==============================================================================

def create_galactic_mask(nside: int, lat_cut_deg: float = 20.0) -> np.ndarray:
    """
    Genera una máscara binaria excluyendo el plano galáctico.

    Parameters
    ----------
    nside       : int
                  Resolución HEALPix del mapa.
    lat_cut_deg : float
                  Píxeles con |b| < lat_cut_deg quedan enmascarados. Default: 20°.

    Returns
    -------
    np.ndarray, dtype=bool, shape (12*nside**2,)
        True  → píxel incluido (fuera del plano galáctico).
        False → píxel excluido (plano galáctico).
    
    Raises
    ------
    ValueError : si lat_cut_deg no está entre 0 y 90.
    """
    if not 0.0 <= lat_cut_deg <= 90.0:
        raise ValueError(
            f"'lat_cut_deg' debe estar en [0, 90], pero se recibió: {lat_cut_deg}"
        )
    
    npix     = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))

    galactic_lat_rad = np.pi / 2 - theta

    galactic_mask = np.ones(npix, dtype=bool)
    galactic_mask[np.abs(galactic_lat_rad) < np.radians(lat_cut_deg)] = False
    return galactic_mask

def _sky_fraction(valid_pixel_mask: np.ndarray) -> float:
    """Fracción de cielo visible f_sky = N_visibles / N_total."""
    return float(np.sum(valid_pixel_mask) / valid_pixel_mask.size)

# ==============================================================================    
# Análisis armónico
# ==============================================================================

def compute_pseudo_cl(healpix_map_1d: np.ndarray, valid_pixel_mask: np.ndarray) -> np.ndarray:
    """
    Calcula el pseudo-C_ℓ de un mapa enmascarado con corrección f_sky.

    Los píxeles enmascarados se rellenan con 0 antes de pasar a anafast,
    y el resultado se divide por f_sky (corrección de primer orden).

    Parameters
    ----------
    healpix_map_1d   : np.ndarray, shape (12*nside**2,)
                       Mapa HEALPix en esquema RING.
    valid_pixel_mask : np.ndarray, dtype=bool
                       Máscara (True = píxel válido).

    Returns
    -------
    np.ndarray
        Pseudo-C_ℓ corregido por f_sky.

    Raises
    ------
    ValueError : si valid_pixel_mask está vacío.
    """
    if valid_pixel_mask.sum() == 0:
        raise ValueError("La máscara excluye todos los píxeles."
                          " Reduce 'lat_cut_deg' o revisa la máscara de cobertura instrumental.")
    
    masked_map      = hp.ma(healpix_map_1d)
    masked_map.mask = ~valid_pixel_mask
    cl              = hp.anafast(masked_map.filled(0.0))
    sky_fraction    = _sky_fraction(valid_pixel_mask)
    return cl / max(sky_fraction, 1e-6)

def spectral_metrics(
    pred_map: np.ndarray,
    target_map: np.ndarray,
    residual_map: np.ndarray,
    valid_pixel_mask: np.ndarray,
) -> dict:
    """
    Calcula métricas espectrales completas para una muestra.

    Parameters
    ----------
    pred_map    : np.ndarray, shape (12*nside**2,)
    target_map       : np.ndarray, shape (12*nside**2,)
    residual_map     : np.ndarray, shape (12*nside**2,)
                       residual_map = target_map - predicted_map
    valid_pixel_mask : np.ndarray, dtype=bool

    Returns
    -------
    dict con claves:
        'ell'         : array de multipolos
        'cl_target'   : pseudo-C_ℓ del target
        'cl_pred'     : pseudo-C_ℓ de la predicción
        'cl_residual' : pseudo-C_ℓ del residuo
        'ratio'       : cl_pred / cl_target  (≈1 si la red es perfecta)
        'transfer'    : cl_residual / cl_target  (<<1 = ruido bien eliminado)
        'f_sky'       : fracción de cielo visible
    """
    cl_target   = compute_pseudo_cl(target_map,   valid_pixel_mask)
    cl_pred     = compute_pseudo_cl(pred_map,     valid_pixel_mask)
    cl_residual = compute_pseudo_cl(residual_map, valid_pixel_mask)
    ell         = np.arange(len(cl_target))
    sky_fraction = _sky_fraction(valid_pixel_mask)

    safe_denominator = np.maximum(cl_target, 1e-30)
    return {
        'ell'        : ell,
        'cl_target'  : cl_target,
        'cl_pred'    : cl_pred,
        'cl_residual': cl_residual,
        'ratio'      : cl_pred    / np.maximum(cl_target, safe_denominator),
        'transfer'   : cl_residual / np.maximum(cl_target, safe_denominator),
        'f_sky'      : sky_fraction,
    }

def deviation_vs_cosmic_variance(
    pred_maps: np.ndarray,
    target_maps: np.ndarray,
    valid_pixel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Réplica de la Figura 7 de Wang et al. 2022.

    Calcula la desviación estándar del error en C_ℓ sobre N muestras del
    test set, y la compara con la varianza cósmica.

    Parameters
    ----------
    pred_maps        : np.ndarray, shape (N, 12*nside**2)
    target_maps      : np.ndarray, shape (N, 12*nside**2)
    valid_pixel_mask : np.ndarray, dtype=bool

    Returns
    -------
    ell          : np.ndarray
    sigma_cnn    : np.ndarray
                   Desviación estándar de ΔC_ℓ sobre las N muestras.
    sigma_cosmic : np.ndarray
                   Varianza cósmica: sqrt(2/[(2ℓ+1) f_sky]) * C_ℓ_medio.
                   Con full_mask (f_sky=1) se reduce a sqrt(2/(2ℓ+1)) * C_ℓ_medio.

    Raises
    ------
    ValueError : si pred_maps y target_maps no tienen la misma forma.
    """
    if pred_maps.shape != target_maps.shape:
        raise ValueError(f"'pred_maps' y 'target_maps' deben tener la misma forma, "
                         f"pero se recibieron: {pred_maps.shape} vs {target_maps.shape}")

    delta_cl = []
    for pred, target in zip(pred_maps, target_maps):
        cl_pred   = compute_pseudo_cl(pred,   valid_pixel_mask)
        cl_target = compute_pseudo_cl(target, valid_pixel_mask)
        delta_cl.append(cl_pred - cl_target)

    delta_cl     = np.array(delta_cl)          # (N, lmax+1)
    sigma_cnn    = np.std(delta_cl, axis=0)

    cl_mean      = np.array([
        compute_pseudo_cl(t, valid_pixel_mask) for t in target_maps
    ]).mean(axis=0)
    ell          = np.arange(len(sigma_cnn))
    # El factor f_sky degrada la varianza cósmica sobre cielo parcial.
    # Con full_mask f_sky=1 y el término se anula (sqrt(2/(2ℓ+1)) * C_ℓ).
    sky_fraction = _sky_fraction(valid_pixel_mask)
    sigma_cosmic = np.sqrt(2 / np.maximum((2*ell + 1) * sky_fraction, 1e-6)) * cl_mean

    return ell, sigma_cnn, sigma_cosmic

# ==============================================================================
# Análisis espacial
# ==============================================================================

def rms_by_latitude_bands(
    residual_map: np.ndarray,
    nside: int,
    n_latitude_bands: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Réplica de la Figura 4 de Wang et al. 2022.

    Calcula el RMS del mapa de residuos en n_bands bandas de latitud
    galáctica de igual anchura entre -90° y +90°.

    Parameters
    ----------
    residual_map     : np.ndarray, shape (12*nside**2,)
    nside            : int
    n_latitude_bands : int
                       Número de bandas de latitud galáctica. Default: 18 (bandas de 10° cada una).

    Returns
    -------
    centers : np.ndarray, shape (n_latitude_bands,)
              Centro de cada banda en grados.
    rms     : np.ndarray, shape (n_latitude_bands,)
              RMS del residuo en cada banda.
    """
    if n_latitude_bands < 1:
        raise ValueError(f"'n_latitude_bands' debe ser al menos 1, pero se recibió: {n_latitude_bands}")
    
    npix              = hp.nside2npix(nside)
    theta, _          = hp.pix2ang(nside, np.arange(npix))
    galactic_lat_deg  = np.degrees(np.pi / 2 - theta)
    edges             = np.linspace(-90, 90, n_latitude_bands + 1)

    centers, rms = [], []
    for band_idx in range(n_latitude_bands):
        in_band = (galactic_lat_deg >= edges[band_idx]) & (galactic_lat_deg < edges[band_idx + 1])
        if in_band.sum() > 0:
            centers.append((edges[band_idx] + edges[band_idx + 1]) / 2)
            rms.append(np.sqrt(np.mean(residual_map[in_band] ** 2)))

    return np.array(centers), np.array(rms)


def spatial_metrics(
    pred_map: np.ndarray,
    target_map: np.ndarray,
    residual_map: np.ndarray,
    valid_pixel_mask: np.ndarray,
) -> dict:
    """
    Métricas espaciales escalares sobre la región enmascarada.

    Parameters
    ----------
    pred_map         : np.ndarray, shape (12*nside**2,)
    target_map       : np.ndarray, shape (12*nside**2,)
    residual_map     : np.ndarray, shape (12*nside**2,)
                       residual_map = target_map - pred_map
    valid_pixel_mask : np.ndarray, dtype=bool

    Returns
    -------
    dict con claves: 'rmse', 'bias', 'pearson_r'
    """
    r = np.corrcoef(pred_map[valid_pixel_mask], target_map[valid_pixel_mask])[0, 1]
    
    return {
        'rmse'     : float(np.sqrt(np.mean(residual_map[valid_pixel_mask] ** 2))),
        'bias'     : float(np.mean(residual_map[valid_pixel_mask])),
        'pearson_r': float(r),
    }