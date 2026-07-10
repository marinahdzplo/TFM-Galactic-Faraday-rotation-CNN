"""
eval.py — Evaluación espacial y armónica del UNetDenoiser.

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
import torch
import matplotlib.pyplot as plt

from utils.fn_eval import (
    create_galactic_mask,
    spatial_metrics,
    spectral_metrics,
    rms_by_latitude_bands,
    deviation_vs_cosmic_variance,
)
from utils.data import map_2d_to_healpix, RMDataset
from utils.model import UNetDenoiser
from utils.train import predict

# ==============================================================================
# Evaluación de muestras
# ==============================================================================
 
def evaluate_rm_sample(
    model: UNetDenoiser,
    dataset: RMDataset,
    sample_idx: int,
    checkpoint_path: str,
    freq_idx: int | None = None,
    nside: int = 16,
    lat_cut_deg: float = 20.0,
    save_path: str | None = None,
) -> dict:
    """
    Evalúa una muestra del RMDataset y genera una figura.
 
    Paneles (3 filas × 3 columnas):
        Fila 1: Q en una frecuencia central / U en una frecuencia central /
                RM target (ground truth).
        Fila 2: RM predicha / Residuo (target - pred) / Scatter pred-vs-target.
        Fila 3: Espectro C_ℓ de RM_target / RM_pred / RM_residual /
                RMS del residuo por banda de latitud /
                Ratio C_ℓ_pred/C_ℓ_target.
 
    Las unidades son rad/m². Los mapas se desescalan automáticamente
    multiplicando por rm_scale del dataset.
 
    Parameters
    ----------
    model       : UNetDenoiser
                  Entrenado, en eval() y en el dispositivo correcto.
    dataset     : RMDataset
    sample_idx  : int
                  Índice de la muestra a evaluar en el dataset.
    checkpoint_path : str
                  Ruta al checkpoint .pt del modelo, para leer 'target_scale'.
    nside       : int
                  Resolución HEALPix. Default: 16.
    lat_cut_deg : float
                  Corte de latitud galáctica para las métricas. Default: 20°.
    save_path   : str | None
                  Si se indica, guarda la figura en esa ruta.
 
    Returns
    -------
    dict con las métricas espaciales y espectrales escalares.
    """
    if not (0 <= sample_idx < len(dataset)):
        raise IndexError(f"sample_idx fuera de rango [0, {len(dataset)-1}]")
 
    input_tensor, target_tensor = dataset[sample_idx]
    meta     = dataset.get_metadata(sample_idx)
    rm_scale = dataset.rm_scale
    qu_scale = dataset.qu_scale
    freqs    = dataset.freqs_ghz
 
    # predict() multiplica por target_scale
    pred_tensor = predict(model, input_tensor.unsqueeze(0), checkpoint_path=checkpoint_path)
 
    pred_2d    = pred_tensor.squeeze().cpu().numpy()
    target_2d  = (target_tensor * rm_scale).squeeze().cpu().numpy()
 
    # Q, U para visualización — elegimos la frecuencia central del stack.
    mid      = freq_idx if freq_idx is not None else len(freqs) // 2
    q_mid_2d = (input_tensor[2*mid] * qu_scale).cpu().numpy()
    u_mid_2d = (input_tensor[2*mid + 1] * qu_scale).cpu().numpy()
 
    # Proyecciones a esfera y cálculo del residuo
    pred_map    = map_2d_to_healpix(pred_2d, nside)
    target_map  = map_2d_to_healpix(target_2d, nside)
    residual    = target_map - pred_map
    q_mid_map   = map_2d_to_healpix(q_mid_2d, nside)
    u_mid_map   = map_2d_to_healpix(u_mid_2d, nside)
 
    # Máscaras y métricas espaciales por región:
    # FULL_MASK   : todo el cielo
    # PLANE_MASK  : |b| < LAT_CUT (plano galáctico, donde se concentra la RM)
    # POLES_MASK  : |b| > LAT_CUT (polos, donde RM es más débil)
    npix       = hp.nside2npix(nside)
    poles_mask = create_galactic_mask(nside, lat_cut_deg=lat_cut_deg)  
    plane_mask = ~poles_mask                                       
    full_mask  = np.ones(npix, dtype=bool)

    spatial_full  = spatial_metrics(pred_map, target_map, residual, full_mask)
    spatial_plane = spatial_metrics(pred_map, target_map, residual, plane_mask)
    spatial_poles = spatial_metrics(pred_map, target_map, residual, poles_mask)
    spectral      = spectral_metrics(pred_map, target_map, residual, full_mask)
    band_c, band_r = rms_by_latitude_bands(residual, nside)

    sigma_str = ", ".join(f"{s:.3f}" for s in meta["sigma_0"])
    print(f'Muestra {sample_idx} (realización r={meta["real_i"]}, σ_0=[{sigma_str}] mK)')
    print(f'                   {"RMSE":>8}   {"bias":>8}   {"Pearson r":>10}  [rad/m²]')
    print(f'  Global:          {spatial_full ["rmse"]:8.2f}   {spatial_full ["bias"]:+8.2f}   {spatial_full ["pearson_r"]:10.4f}')
    print(f'  Plano (|b|<{lat_cut_deg:.0f}°): {spatial_plane["rmse"]:8.2f}   {spatial_plane["bias"]:+8.2f}   {spatial_plane["pearson_r"]:10.4f}')
    print(f'  Polos (|b|>{lat_cut_deg:.0f}°): {spatial_poles["rmse"]:8.2f}   {spatial_poles["bias"]:+8.2f}   {spatial_poles["pearson_r"]:10.4f}')

 
    # Figura 
    cmap = 'coolwarm'
    fig = plt.figure(figsize=(18, 14))
 
    # Fila 1
    hp.mollview(q_mid_map, title=f"Q observado (ν={freqs[mid]} GHz)  [mK]",
                sub=(3, 3, 1), cmap=cmap, norm='hist', fig=fig.number)
    hp.mollview(u_mid_map, title=f"U observado (ν={freqs[mid]} GHz)  [mK]",
                sub=(3, 3, 2), cmap=cmap, norm='hist', fig=fig.number)
    hp.mollview(target_map, title="RM target  [rad/m²]",
                sub=(3, 3, 3), cmap=cmap, fig=fig.number)
 
    # Fila 2
    hp.mollview(pred_map, title="RM predicha  [rad/m²]",
                sub=(3, 3, 4), cmap=cmap, fig=fig.number)
    hp.mollview(residual,
                title=f"Residuo",
                sub=(3, 3, 5), cmap=cmap, fig=fig.number)
 
    ax_sc = fig.add_subplot(3, 3, 6)
    ax_sc.hexbin(target_map[full_mask], pred_map[full_mask], gridsize=40,
                 cmap='viridis', bins='log')
    vmin = min(target_map[full_mask].min(), pred_map[full_mask].min())
    vmax = max(target_map[full_mask].max(), pred_map[full_mask].max())
    ax_sc.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1)
    ax_sc.set_xlabel('RM target [rad/m²]')
    ax_sc.set_ylabel('RM pred [rad/m²]')
    ax_sc.set_title(f'Scatter full (r={spatial_full["pearson_r"]:.3f})')
    ax_sc.grid(True, alpha=0.3)
 
    # Fila 3
    ax_cl = fig.add_subplot(3, 3, 7)
    ell = spectral['ell']
    ax_cl.plot(ell, spectral['cl_target'],   label='target',    color='black',  lw=2)
    ax_cl.plot(ell, spectral['cl_pred'],     label='pred',      color='orange', lw=2, ls='--')
    ax_cl.plot(ell, spectral['cl_residual'], label='residual',  color='red',    lw=1.5, alpha=0.7)
    ax_cl.set_yscale('log')
    ax_cl.set_xlabel(r'$\ell$')
    ax_cl.set_ylabel(r'pseudo-$C_\ell$  [rad$^2$/m$^4$]')
    ax_cl.set_title('Espectro de potencia de RM')
    ax_cl.legend(fontsize=8)
    ax_cl.grid(True, alpha=0.3)
 
    ax_rms = fig.add_subplot(3, 3, 8)
    ax_rms.plot(band_c, band_r, 'o-', color='steelblue', ms=4)
    ax_rms.axvline(-lat_cut_deg, color='gray', lw=0.8, ls='--')
    ax_rms.axvline( lat_cut_deg, color='gray', lw=0.8, ls='--')
    ax_rms.set_xlabel('Latitud galáctica [deg]')
    ax_rms.set_ylabel('RMS residuo [rad/m²]')
    ax_rms.set_title('RMS del residuo por banda de latitud')
    ax_rms.grid(True, alpha=0.3)
 
    ax_r = fig.add_subplot(3, 3, 9)
    ax_r.plot(ell, spectral['ratio'],    label=r'$C_\ell^{pred}/C_\ell^{target}$',
              color='orange', lw=2)
    ax_r.plot(ell, spectral['transfer'], label=r'$C_\ell^{res}/C_\ell^{target}$',
              color='red',    lw=1.5, alpha=0.8)
    ax_r.axhline(1.0, color='black', lw=0.8, ls='--')
    ax_r.set_ylim(0, 2)
    ax_r.set_xlabel(r'$\ell$')
    ax_r.set_ylabel('Ratio')
    ax_r.set_title('Ratio espectral (ideal: 1 y 0)')
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)
 
    plt.suptitle(
        f'Evaluación RM — realización {meta["real_i"]} | '
        f'Pearson (plano)={spatial_plane["pearson_r"]:.3f} '
        f'(global={spatial_full["pearson_r"]:.3f})',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
 
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figura guardada en: {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return {
        'sample_idx' : sample_idx,
        'real_i'     : int(meta['real_i']),
        'sigma_0'    : sigma_str,
        'full'       : spatial_full,
        'plane'      : spatial_plane,
        'poles'      : spatial_poles,
        'spectral'   : {
            'ell'         : spectral['ell'].tolist(),
            'cl_target'   : spectral['cl_target'].tolist(),
            'cl_pred'     : spectral['cl_pred'].tolist(),
            'cl_residual' : spectral['cl_residual'].tolist(),
            'ratio'       : spectral['ratio'].tolist(),
            'transfer'    : spectral['transfer'].tolist(),
        },
        'latitude_bands': {
            'centers' : band_c.tolist(),
            'rms'     : band_r.tolist(),
        },
    }
 
 
def _aggregate_spectral(
    pred_maps: np.ndarray,
    target_maps: np.ndarray,
    valid_pixel_mask: np.ndarray,
) -> dict:
    """
    Agrega las métricas espectrales sobre todas las muestras.

    Calcula, muestra a muestra, los pseudo-C_ℓ (target, pred, residuo) y los
    ratios C_ℓ^pred/C_ℓ^target y C_ℓ^res/C_ℓ^target; después promedia cada
    cantidad sobre las N muestras y estima su dispersión (percentiles 16-84).
    """
    cl_target, cl_pred, cl_residual, ratio, transfer = [], [], [], [], []
    for pred, target in zip(pred_maps, target_maps):
        res = target - pred
        m = spectral_metrics(pred, target, res, valid_pixel_mask)
        cl_target  .append(m['cl_target'])
        cl_pred    .append(m['cl_pred'])
        cl_residual.append(m['cl_residual'])
        ratio      .append(m['ratio'])
        transfer   .append(m['transfer'])

    def _stats(arr_list):
        a = np.array(arr_list)
        lo, hi = np.percentile(a, [16, 84], axis=0)
        return a.mean(axis=0), lo, hi

    ct_m, ct_lo, ct_hi = _stats(cl_target)
    cp_m, cp_lo, cp_hi = _stats(cl_pred)
    cr_m, cr_lo, cr_hi = _stats(cl_residual)
    ra_m, ra_lo, ra_hi = _stats(ratio)
    tr_m, tr_lo, tr_hi = _stats(transfer)

    return {
        'ell'             : np.arange(len(ct_m)),
        'cl_target_mean'  : ct_m, 'cl_target_lo'  : ct_lo, 'cl_target_hi'  : ct_hi,
        'cl_pred_mean'    : cp_m, 'cl_pred_lo'    : cp_lo, 'cl_pred_hi'    : cp_hi,
        'cl_residual_mean': cr_m, 'cl_residual_lo': cr_lo, 'cl_residual_hi': cr_hi,
        'ratio_mean'      : ra_m, 'ratio_lo'      : ra_lo, 'ratio_hi'      : ra_hi,
        'transfer_mean'   : tr_m, 'transfer_lo'   : tr_lo, 'transfer_hi'   : tr_hi,
    }


def _aggregate_rms_bands(
    pred_maps: np.ndarray,
    target_maps: np.ndarray,
    nside: int,
    n_latitude_bands: int = 18,
) -> dict:
    """
    Agrega el RMS del residuo por bandas de latitud sobre todas las muestras,
    promediando cada banda y estimando su dispersión (percentiles 16-84).
    """
    centers, rms_all = None, []
    for pred, target in zip(pred_maps, target_maps):
        res = target - pred
        c, r = rms_by_latitude_bands(res, nside, n_latitude_bands=n_latitude_bands)
        if centers is None:
            centers = c
        rms_all.append(r)

    rms_all = np.array(rms_all)
    rms_lo, rms_hi = np.percentile(rms_all, [16, 84], axis=0)
    return {
        'centers'  : centers,
        'rms_mean' : rms_all.mean(axis=0),
        'rms_lo'   : rms_lo,
        'rms_hi'   : rms_hi,
    }


def evaluate_rm_full_val(
    model: UNetDenoiser,
    dataset: RMDataset,
    checkpoint_path: str,
    nside: int = 16,
    lat_cut_deg: float = 20.0,
    max_samples: int | None = None,
    save_path: str | None = None,
) -> dict:
    """
    Evalúa la red sobre todas (o las primeras max_samples) muestras del
    dataset RM de validación. Devuelve un dict con listas de métricas por
    región (global, plano galáctico, polos) y la desviación σ_CNN vs
    varianza cósmica sobre el cielo entero.

    Returns
    -------
    dict con claves:
        'results_full', 'results_plane', 'results_poles' : list[dict] por muestra
                          (métricas espaciales RMSE/bias/pearson por región)
        'ell', 'sigma_cnn', 'sigma_cosmic'               : σ_CNN vs varianza cósmica
        'spectral_agg'   : dict con espectros medios (target/pred/residuo) y
                           ratios medios (ratio, transfer) con bandas 16-84
        'rms_bands_agg'  : dict con RMS medio del residuo por banda de latitud
        'all_pred', 'all_target'                         : list[np.ndarray] por muestra
    """
    n_eval = min(max_samples, len(dataset)) if max_samples else len(dataset)
    print(f"Evaluando {n_eval}/{len(dataset)} muestras...")

    rm_scale = dataset.rm_scale

    npix       = hp.nside2npix(nside)
    poles_mask = create_galactic_mask(nside, lat_cut_deg=lat_cut_deg)
    plane_mask = ~poles_mask
    full_mask  = np.ones(npix, dtype=bool)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'target_scale' not in ckpt:
        raise KeyError(
            f"El checkpoint '{checkpoint_path}' no contiene 'target_scale'."
        )
    target_scale = float(ckpt['target_scale'])
    del ckpt

    model.eval()
    device = next(model.parameters()).device

    results_full, results_plane, results_poles = [], [], []
    all_pred, all_target = [], []

    with torch.no_grad():
        for idx in range(n_eval):
            inp, tgt = dataset[idx]
            pred = model(inp.unsqueeze(0).to(device)) * target_scale

            pred_map   = map_2d_to_healpix(pred.squeeze().cpu().numpy(), nside)
            target_map = map_2d_to_healpix((tgt * rm_scale).squeeze().numpy(), nside)
            res = target_map - pred_map

            results_full .append(spatial_metrics(pred_map, target_map, res, full_mask))
            results_plane.append(spatial_metrics(pred_map, target_map, res, plane_mask))
            results_poles.append(spatial_metrics(pred_map, target_map, res, poles_mask))
            all_pred.append(pred_map)
            all_target.append(target_map)

    def _agg(results, key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)

    print(f'\n                      {"⟨RMSE⟩":>16}      {"⟨bias⟩":>14}      {"⟨Pearson r⟩":>16}')
    for name, res in [('Global',                            results_full),
                      (f'Plano (|b|<{lat_cut_deg:.0f}°)',   results_plane),
                      (f'Polos (|b|>{lat_cut_deg:.0f}°)',   results_poles)]:
        rm_m, rm_s = _agg(res, 'rmse')
        bi_m, bi_s = _agg(res, 'bias')
        pe_m, pe_s = _agg(res, 'pearson_r')
        print(f'  {name:<22} {rm_m:6.2f} ± {rm_s:5.2f}    '
              f'{bi_m:+6.2f} ± {bi_s:5.2f}    '
              f'{pe_m:.4f} ± {pe_s:.4f}')

    # σ_CNN vs varianza cósmica SOBRE TODO EL CIELO
    pred_arr, target_arr = np.array(all_pred), np.array(all_target)
    ell, sigma_cnn, sigma_cosmic = deviation_vs_cosmic_variance(
        pred_arr, target_arr, full_mask
    )

    # Métricas armónicas agregadas sobre el valset (cielo completo)
    spectral_agg = _aggregate_spectral(pred_arr, target_arr, full_mask)
    rms_bands_agg = _aggregate_rms_bands(pred_arr, target_arr, nside)

    # Figura: métricas ESPACIALES (histogramas por región) + ARMÓNICAS
    # (σ_CNN, espectro medio, ratio medio, RMS por bandas de latitud)
    rmse_plane    = [r['rmse']      for r in results_plane]
    rmse_poles    = [r['rmse']      for r in results_poles]
    pearson_plane = [r['pearson_r'] for r in results_plane]
    pearson_poles = [r['pearson_r'] for r in results_poles]
    bias_plane    = [r['bias']      for r in results_plane]
    bias_poles    = [r['bias']      for r in results_poles]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # ---------------- Métricas espaciales ----------------
    # Panel [0,0] — RMSE
    bins_rmse = np.linspace(
        min(min(rmse_plane), min(rmse_poles)),
        max(max(rmse_plane), max(rmse_poles)), 25,
    )
    axes[0, 0].hist(rmse_plane, bins=bins_rmse, color='steelblue', alpha=0.55,
                    label=f'plano ⟨⟩={np.mean(rmse_plane):.1f}', edgecolor='white')
    axes[0, 0].hist(rmse_poles, bins=bins_rmse, color='coral',     alpha=0.55,
                    label=f'polos ⟨⟩={np.mean(rmse_poles):.1f}', edgecolor='white')
    axes[0, 0].set_xlabel('RMSE [rad/m²]')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title(f'Distribución RMSE por región (N={n_eval})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel [0,1] — Pearson r
    bins_r = np.linspace(
        min(min(pearson_plane), min(pearson_poles)),
        max(max(pearson_plane), max(pearson_poles)), 25,
    )
    axes[0, 1].hist(pearson_plane, bins=bins_r, color='steelblue', alpha=0.55,
                    label=f'plano ⟨⟩={np.mean(pearson_plane):.3f}', edgecolor='white')
    axes[0, 1].hist(pearson_poles, bins=bins_r, color='coral',     alpha=0.55,
                    label=f'polos ⟨⟩={np.mean(pearson_poles):.3f}', edgecolor='white')
    axes[0, 1].set_xlabel('Pearson r')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title(f'Distribución Pearson r por región (N={n_eval})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel [1,0] — Bias
    bins_b = np.linspace(
        min(min(bias_plane), min(bias_poles)),
        max(max(bias_plane), max(bias_poles)), 25,
    )
    axes[1, 0].hist(bias_plane, bins=bins_b, color='steelblue', alpha=0.55,
                    label=f'plano ⟨⟩={np.mean(bias_plane):+.2f}', edgecolor='white')
    axes[1, 0].hist(bias_poles, bins=bins_b, color='coral',     alpha=0.55,
                    label=f'polos ⟨⟩={np.mean(bias_poles):+.2f}', edgecolor='white')
    axes[1, 0].axvline(0, color='black', lw=0.8, ls='--')
    axes[1, 0].set_xlabel('Bias [rad/m²]')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de bias por región')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ---------------- Métricas armónicas ----------------
    # Panel [1,1] — σ_CNN vs cósmica
    axes[1, 1].semilogy(ell, sigma_cnn,    label=r'$\sigma_{\ell,\mathrm{CNN}}$',
                        color='steelblue', lw=2)
    axes[1, 1].semilogy(ell, sigma_cosmic,  label='Varianza cósmica',
                        color='black',     lw=1.5, ls='--')
    axes[1, 1].set_xlabel(r'Multipolo $\ell$')
    axes[1, 1].set_ylabel(r'$\sigma_\ell$  [rad$^2$/m$^4$]')
    axes[1, 1].set_title(r'$\sigma_{\mathrm{CNN}}$ vs varianza cósmica (cielo completo)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Panel [2,0] — Espectro de potencia medio (target/pred/residuo) con banda 16-84
    ell_s = spectral_agg['ell']
    for key, color, label in [('cl_target',   'black',  'target'),
                              ('cl_pred',     'orange', 'pred'),
                              ('cl_residual', 'red',    'residual')]:
        axes[2, 0].plot(ell_s, spectral_agg[f'{key}_mean'], color=color, lw=2, label=label)
        axes[2, 0].fill_between(ell_s, spectral_agg[f'{key}_lo'], spectral_agg[f'{key}_hi'],
                                color=color, alpha=0.2)
    axes[2, 0].set_yscale('log')
    axes[2, 0].set_xlabel(r'Multipolo $\ell$')
    axes[2, 0].set_ylabel(r'pseudo-$C_\ell$  [rad$^2$/m$^4$]')
    axes[2, 0].set_title(f'Espectro medio de RM (N={n_eval})')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    # Panel [2,1] — Ratio espectral medio + RMS por bandas (eje gemelo)
    axes[2, 1].plot(ell_s, spectral_agg['ratio_mean'], color='orange', lw=2,
                    label=r'$C_\ell^{pred}/C_\ell^{target}$')
    axes[2, 1].fill_between(ell_s, spectral_agg['ratio_lo'], spectral_agg['ratio_hi'],
                            color='orange', alpha=0.2)
    axes[2, 1].plot(ell_s, spectral_agg['transfer_mean'], color='red', lw=1.5, alpha=0.8,
                    label=r'$C_\ell^{res}/C_\ell^{target}$')
    axes[2, 1].fill_between(ell_s, spectral_agg['transfer_lo'], spectral_agg['transfer_hi'],
                            color='red', alpha=0.15)
    axes[2, 1].axhline(1.0, color='black', lw=0.8, ls='--')
    axes[2, 1].set_ylim(0, 2)
    axes[2, 1].set_xlabel(r'Multipolo $\ell$')
    axes[2, 1].set_ylabel('Ratio')
    axes[2, 1].set_title('Ratio espectral medio (ideal: 1 y 0)')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Evaluación validación — N={n_eval} muestras', fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figura guardada en: {save_path}")
    else:
        plt.show()
    plt.close(fig)

    # Figura aparte: RMS del residuo por banda de latitud (valset completo)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(rms_bands_agg['centers'], rms_bands_agg['rms_mean'], 'o-',
             color='steelblue', ms=4, label=f'media (N={n_eval})')
    ax2.fill_between(rms_bands_agg['centers'], rms_bands_agg['rms_lo'],
                     rms_bands_agg['rms_hi'], color='steelblue', alpha=0.25,
                     label='16–84 %')
    ax2.axvline(-lat_cut_deg, color='gray', lw=0.8, ls='--')
    ax2.axvline( lat_cut_deg, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel('Latitud galáctica [deg]')
    ax2.set_ylabel('RMS residuo [rad/m²]')
    ax2.set_title('RMS del residuo por banda de latitud (valset completo)')
    # ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        base, ext = (save_path.rsplit('.', 1) + ['png'])[:2]
        rms_path = f"{base}_rms_bands.{ext}"
        plt.savefig(rms_path, dpi=150, bbox_inches='tight')
        print(f"  Figura guardada en: {rms_path}")
    else:
        plt.show()
    plt.close(fig2)

    return {
        'results_full'  : results_full,
        'results_plane' : results_plane,
        'results_poles' : results_poles,
        'ell'           : ell,
        'sigma_cnn'     : sigma_cnn,
        'sigma_cosmic'  : sigma_cosmic,
        'spectral_agg'  : spectral_agg,
        'rms_bands_agg' : rms_bands_agg,
        'all_pred'      : all_pred,
        'all_target'    : all_target,
    }

def plot_rm_recovery_diagnostics(
    all_pred: list[np.ndarray],
    all_target: list[np.ndarray],
    save_path: str | None = None,
) -> None:
    """
    Scatter acumulado pred-vs-target y residuo vs target. 
    Complementa a evaluate_rm_full_val.

    Parameters
    ----------
    all_pred, all_target : list of np.ndarray
                           Listas de mapas HEALPix 1D, una entrada por muestra de validación.
    save_path            : str or None
                           Si se indica, guarda la figura.
    """

    all_pred_arr   = np.array(all_pred)     # (N, npix)
    all_target_arr = np.array(all_target)
    all_res_arr    = all_target_arr - all_pred_arr
    n_eval = len(all_pred)

    t_flat = all_target_arr.ravel()
    p_flat = all_pred_arr.ravel()
    r_flat = all_res_arr.ravel()

    # Estadística compacta
    print(f'Residuo global: media={r_flat.mean():+.2f}, std={r_flat.std():.2f} rad/m²')
    print(f'Rango target:   [{t_flat.min():.0f}, {t_flat.max():.0f}] rad/m²')
    print(f'Rango pred:     [{p_flat.min():.0f}, {p_flat.max():.0f}] rad/m²')
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: scatter acumulado
    axes[0].hexbin(t_flat, p_flat, gridsize=60, cmap='viridis', bins='log', mincnt=1)
    lims = [min(t_flat.min(), p_flat.min()), max(t_flat.max(), p_flat.max())]
    axes[0].plot(lims, lims, 'r--', lw=1, label='identidad')
    axes[0].set_xlabel('RM target [rad/m²]')
    axes[0].set_ylabel('RM pred [rad/m²]')
    axes[0].set_title(f'Scatter acumulado — cielo completo ({n_eval} muestras)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: residuo vs target
    axes[1].hexbin(t_flat, r_flat, gridsize=60, cmap='coolwarm', bins='log', mincnt=1)
    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_xlabel('RM target [rad/m²]')
    axes[1].set_ylabel('Residuo (target − pred) [rad/m²]')
    axes[1].set_title('Residuo vs target')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Análisis de recuperación de RM', fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figura guardada en: {save_path}")
    else:
        plt.show()
    plt.close(fig)