"""
data.py — Funciones para el preprocesamiento de mapas y generación de datasets.

Funciones principales
---------------------
load_and_degrade_map           : Lee un FITS HEALPix, suaviza y degrada a Nside objetivo.
healpix_to_2d                  : Proyecta mapa 1D (RING) → tensor 2D (1, 4N, 3N).
map_2d_to_healpix              : Operación inversa: tensor 2D → mapa 1D (RING).
generate_pysm_rm_dataset       : Genera dataset multifrecuencia (Q, U) → RM con PySM3
                                 usando el mapa de RM de Hutschenreuter et al. (2020) como ground truth.
generate_random_rm_dataset     : Genera dataset sintético con rotación aleatoria.
RMDataset                      : torch.utils.data.Dataset para el pipeline de extracción de RM.
"""

__all__ = [
    "load_and_degrade_map",
    "healpix_to_2d",
    "map_2d_to_healpix",
    "generate_pysm_rm_dataset",
    "generate_random_rm_dataset",
    "RMDataset",
]

import os
import math
import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
from torch.utils.data import Dataset
from scipy.constants import c
import torch
import functools

# ==============================================================================
# Funciones de validación
# ==============================================================================

def _validate_nside(nside: int) -> None:
    """
    Verifica que `nside` es una potencia de 2 válida para HEALPix.
 
    Parameters
    ----------
    nside : int
 
    Raises
    ------
    TypeError  : si nside no es entero.
    ValueError : si nside <= 0 o no es potencia de 2.
    """
    if not isinstance(nside, int):
        raise TypeError(
            f"'nside' debe ser un entero, pero se recibió {type(nside).__name__}: {nside!r}"
        )
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError(
            f"'nside' debe ser una potencia de 2 positiva (ej: 16, 32, 64, 128...), "
            f"pero se recibió: {nside}"
        )
    
# ==============================================================================
# Función interna: índice de Morton (curva Z-order) 
# - Krzysztof et al. 2024, "The HEALPix Primer"
# - Wang et al. 2022, "Recovering the CMB Signal with Machine Learning"
# - Sean E. Anderson, "Bit Twiddling Hacks"
# ==============================================================================
@functools.lru_cache(maxsize=16)
def _build_morton_index(face_resolution: int) -> np.ndarray:
    """
    Construye el índice de Morton para una cara HEALPix de tamaño 
    face_resolution × face_resolution.

    En el esquema NESTED de HEALPix, los píxeles dentro de cada cara siguen
    una curva de Morton (Z-order). Este índice permite reordenarlos en una
    cuadrícula 2D espacialmente coherente.

    En cada iteración se cuadruplican los bloques actuales,
    organizándolos en la disposición estándar de Morton:
        [bloque+3s  bloque+1s]
        [bloque+2s  bloque+0s]
    donde s es el número de píxeles en el bloque anterior.

    Parameters
    ----------
    face_resolution : int
                      Resolución de la cara (potencia de 2). Para un mapa con Nside global,
                      cada cara tiene face_resolution = Nside píxeles por lado.

    Returns
    -------
    np.ndarray, shape (face_resolution, face_resolution), dtype=float64
        Índice plano del píxel que debe colocarse en cada posición (i, j).
    
    Raises
    ------
    ValueError : si face_resolution no es una potencia de 2 positiva.
    """
    _validate_nside(face_resolution)
    
    # Base 2x2: Morton estándar
    morton_block = np.array([[3.0, 1.0],
                            [2.0, 0.0]])

    num_doublings = int(math.log2(face_resolution)) - 1

    # Cada iteración duplica la resolución:
    # si face_resolution=8 -> 2 iteraciones: 2->4->8
    # num_doublings = log2(8) - 1 = 3 - 1 = 2
    # 
    # si face_resolution=32 -> 4 iteraciones: 2->4->8->16->32
    # num_doublings = log2(32) - 1 = 5 - 1 = 4

    for _ in range(num_doublings):
        block_size   = morton_block.size  # número de píxeles en el bloque actual
        morton_block = np.block([
            [morton_block + 3 * block_size, morton_block + 1 * block_size],
            [morton_block + 2 * block_size, morton_block + 0 * block_size],
        ])
    return morton_block  # shape (face_resolution, face_resolution)

# Disposición geográfica de las 12 caras HEALPix en la cuadrícula plana: 
_HEALPIX_FACE_LAYOUT: list[list[int]] = [
    [2,  6,  9],    # norte
    [3,  7, 10],
    [0,  4, 11],
    [1,  5,  8],    # sur
]

# Número de filas y columnas de la cuadrícula plana
_GRID_N_ROWS = len(_HEALPIX_FACE_LAYOUT)        # 4
_GRID_N_COLS = len(_HEALPIX_FACE_LAYOUT[0])     # 3
 

# ==============================================================================
# Carga y degradación de mapas FITS
# ==============================================================================

def load_and_degrade_map(
    file_path: str | Path,
    fields: int | tuple[int],
    hdu: int = 1,
    target_nside: int = 64,
    smooth_fwhm_deg: float | None = None,
    unit_conversion: float = 1.0,
    power: float | None = None,
    bad_pixel_threshold: float | None = None,
) -> np.ndarray:
    """
    Lee un mapa HEALPix de un archivo FITS, detecta automáticamente su Nside
    y ordenamiento, suaviza opcionalmente y degrada al Nside objetivo.

    Parameters
    ----------
    file_path           : str or Path
                          Ruta al archivo FITS.
    fields              : int or tuple of int
                          Campo(s) a leer del HDU (índice base 0).
    hdu                 : int
                          Número de HDU del que leer los datos. Default: 1.
    target_nside        : int
                          Nside de salida. Default: 64.
    smooth_fwhm_deg     : float or None
                          FWHM del suavizado gaussiano en grados. Si None, no se suaviza.
    unit_conversion     : float
                          Factor multiplicativo de conversión de unidades. Default: 1.0.
                          Ejemplo: 1/1e3 para convertir de µK a mK.
    power               : float or None
                          Parámetro `power` de `hp.ud_grade`. Usar -2 para hitmaps
                          o 2 para mapas de varianza. Si None, usa el promedio por defecto.
    bad_pixel_threshold : float or None
                          Umbral para marcar píxeles con valores absurdos como hp.UNSEEN.
                          Útil para mapas HFI de Planck Legacy (100 GHz). Si None, no se filtra.

    Returns
    -------
    np.ndarray
        Mapa(s) en ordenamiento RING degradado(s) al Nside objetivo.
        Shape (12*Nside**2,) si fields es int; (N, 12*Nside**2) si es tupla.

    Raises
    ------
    FileNotFoundError : si file_path no existe.
    ValueError : si target_nside no es potencia de 2.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo FITS: '{file_path}'")

    _validate_nside(target_nside)

    with fits.open(file_path) as hdul:
        header   = hdul[hdu].header
        nside_in = header.get('NSIDE', None)
        ordering = header.get('ORDERING', 'RING').upper().strip()

    # Si no aparece nside en la cabecera, lo obtengo con el número de píxeles
    if nside_in is None:
        sample_field = fields[0] if hasattr(fields, '__len__') else fields
        sample_map   = hp.read_map(file_path, field=sample_field, hdu=hdu, verbose=False)
        nside_in     = hp.get_nside(sample_map)
        del sample_map
    else:
        nside_in = int(nside_in)

    is_nested_ordering = ordering == 'NESTED'

    maps = hp.read_map(file_path, field=fields, hdu=hdu, nest=is_nested_ordering) * unit_conversion

    if bad_pixel_threshold is not None:
        bad_pixels       = np.abs(maps) > bad_pixel_threshold
        maps[bad_pixels] = hp.UNSEEN

    if is_nested_ordering:
        maps = hp.reorder(maps, inp='NEST', out='RING')

    if smooth_fwhm_deg is not None:
        fwhm_rad = np.radians(smooth_fwhm_deg)
        if maps.ndim == 1:
            maps = hp.smoothing(maps, fwhm=fwhm_rad)
        else:
            maps = np.array([hp.smoothing(m, fwhm=fwhm_rad) for m in maps])

    if nside_in != target_nside:
        ud_grade_kwargs = {'nside_out': target_nside, 'order_in': 'RING'}
        if power is not None:
            ud_grade_kwargs['power'] = power
        maps = hp.ud_grade(maps, **ud_grade_kwargs)

    return maps


# ==============================================================================
# Proyecciones HEALPix <-> 2D
# ==============================================================================

def healpix_to_2d(map_ring_1d: np.ndarray, nside: int) -> np.ndarray:
    """
    Proyecta un mapa HEALPix 1D (esquema RING) a un tensor 2D plano.

    Trabajamos con ordenamiento NESTED para preservar la coherencia espacial 
    dentro de cada cara, reordenando cada una de estas via curva de Morton.
    Colocamos las 12 caras en la cuadrícula plana (4*Nside × 3*Nside).

    Parameters
    ----------
    map_ring_1d : np.ndarray, shape (12 * nside**2,)
                  Mapa HEALPix en esquema RING.
    nside       : int
                  Resolución del mapa (potencia de 2).

    Returns
    -------
    np.ndarray, shape (1, 4*nside, 3*nside)
        Tensor con dimensión de canal añadida.

    Raises
    ------
    ValueError : si el número de píxeles no coincide con el nside proporcionado.
    """
    _validate_nside(nside)

    if map_ring_1d.shape[0] != 12 * nside ** 2:
        raise ValueError(
            f"El mapa tiene {map_ring_1d.shape[0]} píxeles, "
            f"pero nside={nside} implica {12 * nside**2}."
        )

    pixels_per_face = nside ** 2

    map_nested_1d   = hp.reorder(map_ring_1d, r2n=True)
    morton_idx_2d   = _build_morton_index(nside).astype(int) # shape (nside, nside)
    morton_idx_flat = morton_idx_2d.flatten() # shape (nside**2,)

    grid_height, grid_width = _GRID_N_ROWS * nside, _GRID_N_COLS * nside
    flat_grid = np.empty((grid_height, grid_width), dtype=map_ring_1d.dtype)

    for grid_row, face_row_idx in enumerate(_HEALPIX_FACE_LAYOUT):
        for grid_col, face_idx in enumerate(face_row_idx):
            # Cada cara tiene nside**2 píxeles, luego: cara 2 si índice es 2*nside**2 a 2*nside**2 + nside**2
            face_start = face_idx * pixels_per_face
            face_end   = face_start + pixels_per_face

            face_pixels_flat = map_nested_1d[face_start:face_end]  # shape (nside**2,)

            # Se toman esos píxeles del array y se reordenan según el índice de Morton para formar la cara 2D
            face_2d = face_pixels_flat[morton_idx_flat].reshape(nside, nside).T  # shape (nside, nside)

            # Colocar la cara reordenada en la posición correcta de la cuadrícula
            row_start, col_start = grid_row * nside, grid_col * nside
            flat_grid[row_start:row_start + nside, col_start:col_start + nside] = face_2d

    return np.expand_dims(flat_grid, axis=0)   # (1, 4N, 3N)


def map_2d_to_healpix(flat_grid_2d: np.ndarray, nside: int) -> np.ndarray:
    """
    Operación inversa de healpix_to_2d: reconstruye el mapa HEALPix 1D (RING)
    desde la matriz 2D.

    Parameters
    ----------
    flat_grid_2d : np.ndarray, shape (4*nside, 3*nside) o (1, 4*nside, 3*nside)
                   Mapa plano producido por healpix_to_2d o la red.
    nside        : int
                   Resolución HEALPix del mapa de salida.

    Returns
    -------
    np.ndarray, shape (12 * nside**2,)
        Mapa HEALPix en esquema RING.

    Raises
    ------
    ValueError : si las dimensiones de flat_grid_2d no coinciden con nside.
    """

    # Eliminar dimensión de canal si existe
    if flat_grid_2d.ndim == 3:
        if flat_grid_2d.shape[0] != 1:
            raise ValueError(
                f"Si flat_grid_2d tiene 3 dimensiones, la primera debe ser 1 (canal), "
                f"pero se recibió con forma {flat_grid_2d.shape}."
            )
        flat_grid_2d = flat_grid_2d[0]  

    expected_height = _GRID_N_ROWS * nside
    expected_width  = _GRID_N_COLS * nside
    if flat_grid_2d.shape != (expected_height, expected_width):
        raise ValueError(
            f"Se esperaba shape ({expected_height}, {expected_width}) "
            f"para nside={nside}, pero se recibió {flat_grid_2d.shape}"
        )
    
    pixels_per_face = nside ** 2
    morton_idx_2d   = _build_morton_index(nside).astype(int) # shape (nside, nside)
    morton_idx_flat = morton_idx_2d.flatten() # shape (nside**2,)

    map_nested_1d = np.empty(12 * nside ** 2, dtype=flat_grid_2d.dtype)

    for grid_row, face_row_idx in enumerate(_HEALPIX_FACE_LAYOUT):
        for grid_col, face_idx in enumerate(face_row_idx):
            row_start, col_start = grid_row * nside, grid_col * nside

            # Deshacer la transpuesta aplicada en healpix_to_2d 
            face_2d_transposed = flat_grid_2d[row_start:row_start + nside, col_start:col_start + nside].T # shape (nside, nside)

            # Deshacer el reordenamiento de Morton para obtener los píxeles de la cara en orden plano
            face_pixels_flat = np.empty(pixels_per_face, dtype=flat_grid_2d.dtype)
            face_pixels_flat[morton_idx_flat] = face_2d_transposed.flatten()

            face_start = face_idx * pixels_per_face
            map_nested_1d[face_start:face_start + pixels_per_face] = face_pixels_flat

    return hp.reorder(map_nested_1d, n2r=True)


# ==========================================================================================
# Generación del dataset sintético haciendo uso de PySM3 + rotación aleatoria
# ==========================================================================================

def generate_random_rm_dataset(
    rm_fits_path: str,
    output_dir: str,
    n_noise_realizations: int,
    seed: int,
    nside: int,
    pysm_nside: int,
    pysm_preset: str,
    freqs_ghz: tuple[int, ...],
    split_ratio: float,
    sensitivity_mK: list[float],
    lknee: list[float],
    slope: list[float],
    smooth_fwhm_deg: float,
    rm_scale: float,
) -> dict:
    """
    Genera un dataset sintético multifrecuencia (Q, U) → RM usando PySM3.

    Pipeline
    --------
    1. Se carga el mapa de RM de Hutschenreuter et al. (2020) como ground truth,
       se suaviza y se degrada al `nside` objetivo. Es idéntico en todas las muestras.
    2. Para cada frecuencia ν ∈ `freqs_ghz`:
         · Se obtiene la emisión de sincrotrón con PySM3 (Q_ν, U_ν intrínsecos).
         · Se suavizan y degradan al `nside`. Son idénticos en todas las muestras.
    3. Para cada realización r = 0 … n_noise_realizations-1:
         · Por cada frecuencia ν_i, se genera un ángulo ψ aleatorio en [0, π),
           obteniendo (Q_ν^obs, U_ν^obs) como:
              Q_ν^obs = Q_ν · cos(2·ψ) + U_ν · sin(2·ψ)
              U_ν^obs = -Q_ν · sin(2·ψ) + U_ν · cos(2·ψ)
         · Se añade ruido a cada (Q_ν^obs, U_ν^obs):
              - Ruido blanco gaussiano, independiente para Q y U, con amplitud
                sigma_i propia de cada frecuencia (tomada de `sensitivity_mK`).
              - Ruido 1/f con espectro de potencia
                C_ℓ^{1/f} = C_ℓ^{white,i} · (ℓ_knee_i / ℓ)^|slope_i|
                donde C_ℓ^{white,i} = sigma_i² · 4π / npix_hi, y ℓ_knee_i
                y slope_i son los valores propios de la frecuencia ν_i.

    Estructura de muestras
    ----------------------
    Cada muestra contiene TODAS las frecuencias apiladas en el eje de canales
    en el orden [Q_ν1, U_ν1, Q_ν2, U_ν2, …, Q_νN, U_νN].

    Split train/val
    ---------------
    Se realiza a nivel de realización: ninguna realización aparece en ambos splits.

    Cada muestra se guarda como un archivo .npz con:
        - 'input'    : np.float32, shape (2·N_freqs, 4·nside, 3·nside)
                       Mapas Q, U apilados y divididos por qu_scale.
        - 'target'   : np.float32, shape (1, 4·nside, 3·nside)
                       Mapa de RM dividido por rm_scale. Idéntico en todas las muestras.
        - 'freqs'    : np.int32,   shape (N_freqs,), frecuencias en GHz.
        - 'sigma_0'  : np.float32, shape (N_freqs,), sensibilidad por frecuencia [mK].
        - 'real_i'   : np.int32,   índice de realización.
        - 'lknee'    : np.float32, shape (N_freqs,), ℓ_knee del ruido 1/f por frecuencia.
        - 'slope'    : np.float32, shape (N_freqs,), slope del ruido 1/f por frecuencia.
        - 'rm_scale' : np.float32, divisor aplicado al target.
        - 'qu_scale' : np.float32, divisor aplicado al input.
        - 'psi_obs'  : np.float32, shape (N_freqs,), ángulo de rotación aleatorio
                       ψ ∈ [0, π) aplicado a cada frecuencia [rad].

    Parameters
    ----------
    rm_fits_path          : str
                            Ruta al mapa FITS de RM (Hutschenreuter et al. 2020).
    output_dir            : str
                            Directorio raíz; se crearán subcarpetas train/ y val/.
    n_noise_realizations  : int
                            Número de realizaciones de ruido = número de muestras del dataset.
    seed                  : int
                            Semilla aleatoria para reproducibilidad.
    nside                 : int
                            Resolución HEALPix del dataset.
    pysm_nside            : int
                            Resolución interna de PySM3 (se degrada a `nside` tras suavizar).
                            Debe ser ≥ `nside`.
    pysm_preset           : str
                            Preset de PySM3 para el modelo de sincrotrón (p.ej. 's1').
    freqs_ghz             : tuple[int, ...]
                            Frecuencias a simular en GHz. Todas aparecen en TODAS las muestras.
                            Debe contener al menos 2 frecuencias.
    split_ratio           : float
                            Fracción de realizaciones destinadas a entrenamiento. Debe estar en (0, 1).
    sensitivity_mK        : list[float]
                            Sensibilidad instrumental [mK] para cada frecuencia.
                            Debe tener la misma longitud que `freqs_ghz`.
    lknee                 : list[float]
                            Valor de ℓ_knee del ruido 1/f para cada frecuencia.
                            Debe tener la misma longitud que `freqs_ghz`.
    slope                 : list[float]
                            Slope del ruido 1/f para cada frecuencia.
                            Debe tener la misma longitud que `freqs_ghz`.
    smooth_fwhm_deg       : float
                            FWHM del beam gaussiano [grados]. Se aplica a la señal
                            y al ruido blanco. No se aplica al ruido 1/f.
    rm_scale              : float
                            Divisor aplicado al target de RM para facilitar la optimización.
                            Debe ser positivo.

    Returns
    -------
    dict con claves:
        'n_train'    : int,       número de muestras en train/
        'n_val'      : int,       número de muestras en val/
        'output_dir' : str,       directorio raíz del dataset
        'nside'      : int,       resolución HEALPix
        'freqs_ghz'  : list[int], frecuencias simuladas
        'rm_scale'   : float,     divisor aplicado al target
        'qu_scale'   : float,     divisor aplicado al input (percentil 99.5 de |Q|, |U|)

    Raises
    ------
    ImportError : si PySM3 no está instalado.
    ValueError  : si los parámetros de entrada no son válidos.
    """
    try:
        import pysm3
        import pysm3.units as u
    except ImportError as exc:
        raise ImportError(
            "PySM3 es necesario para esta función. "
            "Instálalo con: pip install pysm3"
        ) from exc

    if pysm_nside < nside:
        raise ValueError(
            f"'pysm_nside' ({pysm_nside}) debe ser ≥ 'nside' ({nside})"
        )
    if len(freqs_ghz) < 2:
        raise ValueError(
            f"'freqs_ghz' debe contener al menos 2 frecuencias, se recibió: {freqs_ghz}"
        )
    if rm_scale <= 0:
        raise ValueError(
            f"'rm_scale' debe ser positivo, se recibió: {rm_scale}"
        )
    if len(sensitivity_mK) != len(freqs_ghz):
        raise ValueError(
            f"'sensitivity_mK' debe tener longitud {len(freqs_ghz)}, "
            f"se recibió: {len(sensitivity_mK)}"
        )
    if len(lknee) != len(freqs_ghz):
        raise ValueError(
            f"'lknee' debe tener longitud {len(freqs_ghz)}, "
            f"se recibió: {len(lknee)}"
        )
    if len(slope) != len(freqs_ghz):
        raise ValueError(
            f"'slope' debe tener longitud {len(freqs_ghz)}, "
            f"se recibió: {len(slope)}"
        )
    
    rng = np.random.default_rng(seed)

    train_dir = os.path.join(output_dir, 'train')
    val_dir   = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    print(f"Cargando y degradando mapa base de RM (Hutschenreuter 2020) a nside={nside}...")
    rm_lo = load_and_degrade_map(
        file_path=rm_fits_path,
        fields=0,
        target_nside=nside,
        smooth_fwhm_deg=smooth_fwhm_deg,
    )
    rm_lo = np.asarray(rm_lo, dtype=np.float64)
    print(f"RM ground-truth: min={rm_lo.min():7.1f}, "
          f"max={rm_lo.max():7.1f}, std={rm_lo.std():6.1f} [rad/m²]")
    target_tensor = healpix_to_2d(rm_lo / rm_scale, nside).astype(np.float32)

    print(f"Inicializando PySM3 (preset='{pysm_preset}', nside={pysm_nside})...")
    sky_model = pysm3.Sky(
        nside=pysm_nside,
        preset_strings=[pysm_preset],
        output_unit='mK_CMB',
    )

    npix_hi = hp.nside2npix(pysm_nside)
    npix_lo = hp.nside2npix(nside)
    n_freqs = len(freqs_ghz)

    grid_h  = _GRID_N_ROWS * nside
    grid_w  = _GRID_N_COLS * nside

    q0_hires = np.empty((n_freqs, npix_hi), dtype=np.float64)
    u0_hires = np.empty((n_freqs, npix_hi), dtype=np.float64)

    print(f"Obteniendo emisión PySM en {n_freqs} frecuencias...")
    for i, nu_ghz in enumerate(freqs_ghz):
        emission    = sky_model.get_emission(nu_ghz * u.GHz)
        q0_hires[i] = emission[1].value
        u0_hires[i] = emission[2].value
        print(f"ν = {nu_ghz:3d} GHz")

    qu_scale = float(np.percentile(
        np.concatenate([np.abs(q0_hires).ravel(),
                        np.abs(u0_hires).ravel()]), 99.5))
    qu_scale = max(qu_scale, 1e-6)
    print(f"Escala Q,U = {qu_scale:.3g} mK  |  Escala RM = {rm_scale:.1f} rad/m²")

    q_lo = np.empty((n_freqs, npix_lo), dtype=np.float64)
    u_lo = np.empty((n_freqs, npix_lo), dtype=np.float64)
    
    print("Suavizando y degradando mapas Q, U intrínsecos...")
    for i in range(n_freqs):
        q_lo[i] = hp.ud_grade(
            hp.smoothing(q0_hires[i], 
                         fwhm=np.radians(smooth_fwhm_deg)), 
            nside_out=nside)
        u_lo[i] = hp.ud_grade(
            hp.smoothing(u0_hires[i], 
                         fwhm=np.radians(smooth_fwhm_deg)), 
            nside_out=nside)

    all_real_idxs   = np.arange(n_noise_realizations)
    rng.shuffle(all_real_idxs)
    n_train         = int(round(split_ratio * n_noise_realizations))
    train_real_set  = set(all_real_idxs[:n_train].tolist())

    sigmas_per_freq = np.asarray(sensitivity_mK, dtype=np.float64)  # shape (n_freqs,)

    lmax_1f = 3 * pysm_nside - 1
    ell_1f  = np.arange(lmax_1f + 1, dtype=np.float64)
    Cl_white_per_freq = sigmas_per_freq ** 2 * (4.0 * np.pi / npix_hi)  # shape (n_freqs,)

    lknee_per_freq = np.asarray(lknee, dtype=np.float64)
    slope_per_freq = np.asarray(slope, dtype=np.float64)

    freqs_arr    = np.asarray(freqs_ghz, dtype=np.int32)
    input_tensor = np.empty((2 * n_freqs, grid_h, grid_w), dtype=np.float32)

    print(f"Generando {n_noise_realizations} muestras "
          f"({n_train} train / {n_noise_realizations - n_train} val)...")

    for r in range(n_noise_realizations):

        psi_obs_arr = np.empty(n_freqs, dtype=np.float32)

        for i in range(n_freqs):
            # Rotación aleatoria independiente por frecuencia
            psi_obs        = rng.uniform(0, np.pi)
            psi_obs_arr[i] = psi_obs 

            q_obs   = q_lo[i] * np.cos(2 * psi_obs) + u_lo[i] * np.sin(2 * psi_obs)
            u_obs   = -q_lo[i] * np.sin(2 * psi_obs) + u_lo[i] * np.cos(2 * psi_obs)

            noise_q = hp.ud_grade(
                hp.smoothing(rng.standard_normal(npix_hi) * sigmas_per_freq[i],
                             fwhm=np.radians(smooth_fwhm_deg)),
                nside_out=nside,
            )
            noise_u = hp.ud_grade(
                hp.smoothing(rng.standard_normal(npix_hi) * sigmas_per_freq[i],
                             fwhm=np.radians(smooth_fwhm_deg)),
                nside_out=nside,
            )

            ell_knee_i = float(lknee_per_freq[i])
            slope_i    = float(slope_per_freq[i])
            Cl_1f    = np.zeros(lmax_1f + 1)
            Cl_1f[1:] = Cl_white_per_freq[i] * (ell_knee_i / ell_1f[1:]) ** abs(slope_i)

            noise_q += hp.ud_grade(
                hp.synfast(Cl_1f, pysm_nside, lmax=lmax_1f, new=True),
                nside_out=nside,
            )
            noise_u += hp.ud_grade(
                hp.synfast(Cl_1f, pysm_nside, lmax=lmax_1f, new=True),
                nside_out=nside,
            )

            q_noisy = (q_obs + noise_q) / qu_scale
            u_noisy = (u_obs + noise_u) / qu_scale

            input_tensor[2*i]     = healpix_to_2d(q_noisy, nside)[0].astype(np.float32)
            input_tensor[2*i + 1] = healpix_to_2d(u_noisy, nside)[0].astype(np.float32)

        np.savez(
            os.path.join(train_dir if r in train_real_set else val_dir,
                         f"sample_r{r:04d}.npz"),
            input    = input_tensor,
            target   = target_tensor,
            freqs    = freqs_arr,
            real_i   = np.int32(r),
            sigma_0  = np.array(sigmas_per_freq, dtype=np.float32),
            lknee = np.array(lknee_per_freq, dtype=np.float32), 
            slope = np.array(slope_per_freq, dtype=np.float32),   
            rm_scale = np.float32(rm_scale),
            qu_scale = np.float32(qu_scale),
            psi_obs  = psi_obs_arr,
        )

        if (r + 1) % 20 == 0 or (r + 1) == n_noise_realizations:
            print(f"  {r + 1}/{n_noise_realizations} muestras generadas")

    print(
        f"\nDataset RM generado en '{output_dir}'\n"
        f"  Train: {n_train:4d} muestras\n"
        f"  Val: {n_noise_realizations - n_train:4d} muestras\n"
        f"  Input: ({2*n_freqs}, {grid_h}, {grid_w})\n"
        f"  Target: (1, {grid_h}, {grid_w})\n"
    )

    return {
        'n_train'    : n_train,
        'n_val'      : n_noise_realizations - n_train,
        'output_dir' : output_dir,
        'nside'      : nside,
        'freqs_ghz'  : list(freqs_ghz),
        'rm_scale'   : rm_scale,
        'qu_scale'   : qu_scale,
    }

# ==============================================================================
# Dataset PyTorch
# ==============================================================================

class RMDataset(Dataset):
    """
    torch.utils.data.Dataset para el problema (Q, U)_ν multifrecuencia → RM.

    Carga las muestras generadas por generate_pysm_rm_dataset. Cada muestra
    devuelve un par (input, target) donde:

        input  : torch.Tensor, shape (2·N_freqs, 4·nside, 3·nside)
                 Mapas Q y U observados (con ruido) apilados en el eje de
                 canales en el orden [Q_ν1, U_ν1, Q_ν2, U_ν2, …, Q_νN, U_νN],
                 ya divididos por qu_scale.
        target : torch.Tensor, shape (1, 4·nside, 3·nside)
                 Mapa de Medida de Rotación dividido por rm_scale.

    Atributos globales del dataset
    ------------------------------
    freqs_ghz  : list[int]
                 Frecuencias del stack, en GHz.
    lambda_sq  : np.ndarray, shape (N_freqs,), dtype=float32
                 Longitudes de onda al cuadrado, derivadas de freqs_ghz.
                 Útil para la consistency loss y para desrotar predicciones.
    rm_scale   : float
                 Divisor aplicado al target en el generador.
    qu_scale   : float
                 Divisor aplicado al input en el generador.
    n_channels : int
                 Número de canales de entrada = 2*N_freqs.

    Parameters
    ----------
    folder : str
             Ruta a la carpeta train/ o val/ con los archivos .npz generados
             por generate_pysm_rm_dataset.

    Raises
    ------
    FileNotFoundError : si no se encuentran archivos .npz en la carpeta.
    """

    def __init__(self, folder: str):
        self.files = sorted(
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if filename.endswith('.npz')
        )
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No se encontraron archivos .npz en '{folder}'."
            )

        with np.load(self.files[0]) as sample0:
            self.freqs_ghz  = sample0['freqs'].tolist()
            self.rm_scale   = float(sample0['rm_scale'])
            self.qu_scale   = float(sample0['qu_scale'])
            self.n_channels = int(sample0['input'].shape[0])

        lam_m          = c / (np.asarray(self.freqs_ghz, dtype=np.float64) * 1e9)
        self.lambda_sq = (lam_m ** 2).astype(np.float32)

        print(
            f"RMDataset cargado: {len(self.files)} muestras en '{folder}' | "
            f"canales = {self.n_channels} (={2*len(self.freqs_ghz)}) | "
            f"freqs = {self.freqs_ghz} GHz | "
            f"rm_scale = {self.rm_scale:.1f}, qu_scale = {self.qu_scale:.3g}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        input  : torch.Tensor, shape (2*N_freqs, 4*nside, 3*nside)
        target : torch.Tensor, shape (1, 4*nside, 3*nside)
        """
        data   = np.load(self.files[idx])
        input  = torch.from_numpy(data['input'])
        target = torch.from_numpy(data['target'])
        return input, target

    def get_metadata(self, idx: int) -> dict:
        """
        Devuelve los metadatos por muestra de una entrada del dataset.

        A diferencia de los atributos globales de la clase (freqs_ghz,
        rm_scale, etc.), este método devuelve los campos que sí varían
        por muestra: el σ_0 usado para el ruido y el índice de realización.

        Parameters
        ----------
        idx : int

        Returns
        -------
        dict con claves 'sigma_0', 'real_i', 'file'.
        """
        data = np.load(self.files[idx])
        return {
            'sigma_0': data['sigma_0'].tolist(),
            'lknee': data['lknee'].tolist(),
            'slope': data['slope'].tolist(),
            'real_i'    : int(data['real_i']),
            'psi_obs'   : data['psi_obs'].tolist()      if 'psi_obs'      in data else None,
            'euler_angles': data['euler_angles'].tolist() if 'euler_angles' in data else None,
            'file'      : self.files[idx],
        }