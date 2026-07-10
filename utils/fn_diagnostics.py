"""
fn_diagnostics.py — Tests de interpretabilidad para modelos de RM.

Diseñados para distinguir, sobre un modelo ya entrenado, entre memorización
del target y uso genuino del input. Complementan las métricas estándar de
fn_eval.py, que solo miden la calidad de la predicción.

Tests implementados
-------------------
evaluate_on_zero_inputs     : sustituye los inputs por ceros.
                              Detecta memorización pura: si la red predice
                              bien sin ningún input, los sesgos de la red
                              son suficientes para reproducir el target.
evaluate_on_random_inputs   : sustituye los inputs por ruido gaussiano con
                              la misma std por canal. Detecta si la red
                              ignora el input.
evaluate_on_shuffled_inputs : permuta los píxeles dentro de cada canal.
                              Detecta si la red ignora la geometría espacial.
evaluate_channel_occlusion  : pone a cero los canales Q y U de una
                              frecuencia a la vez. Mide la importancia
                              relativa de cada frecuencia para la predicción.

Interpretación
--------------
En cada test se compara la predicción del modelo bajo el input perturbado
con dos referencias:
    · el TARGET del dataset (lo que la red debería predecir si todo fuera bien),
    · la PREDICCIÓN BASELINE con el input real (lo que la red predice realmente).

Si la red MEMORIZA: las predicciones bajo input perturbado se parecen mucho
                    a la baseline (la red ignora el input).
Si la red USA el input: las predicciones se degradan fuertemente respecto a
                        la baseline cuando el input se altera.

"""

import numpy as np
import torch
import torch.nn as nn

from utils.data import RMDataset, map_2d_to_healpix
from utils.fn_eval import spatial_metrics, create_galactic_mask


# ==============================================================================
# Helper interno: predicción en batch sin recargar el checkpoint
# ==============================================================================

def _load_target_scale(checkpoint_path: str) -> float:
    """Lee 'target_scale' de un checkpoint guardado por train_model."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'target_scale' not in ckpt:
        raise KeyError(
            f"El checkpoint '{checkpoint_path}' no contiene 'target_scale'."
        )
    return float(ckpt['target_scale'])


def _predict_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    target_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Forward sin gradiente, escalado a unidades físicas."""
    with torch.no_grad():
        return model(inputs.to(device)) * target_scale

# ==============================================================================
# Test 1 — Inputs vacíos
# ==============================================================================

def evaluate_on_zero_inputs(
    model: nn.Module,
    dataset: RMDataset,
    checkpoint_path: str,
    n_samples: int = 50,
    nside: int = 16,
) -> dict:
    """
    Sustituye los inputs por ceros, y mide cómo se degrada la predicción frente al baseline real.
    """
    n            = min(n_samples, len(dataset))

    target_scale = _load_target_scale(checkpoint_path)
    device       = next(model.parameters()).device
    rm_scale     = dataset.rm_scale
    
    full_mask    = np.ones(12 * nside ** 2, dtype=bool)

    model.eval()
    baseline, zero_results, pred_vs_pred = [], [], []

    for idx in range(n):
        inp, tgt = dataset[idx]
        inp_zero = torch.zeros_like(inp)

        pred_real = _predict_batch(model, inp.unsqueeze(0),      target_scale, device)
        pred_zero = _predict_batch(model, inp_zero.unsqueeze(0), target_scale, device)

        target_map     = map_2d_to_healpix((tgt * rm_scale).squeeze().numpy(), nside)
        pred_real_map  = map_2d_to_healpix(pred_real.squeeze().cpu().numpy(),  nside)
        pred_zero_map  = map_2d_to_healpix(pred_zero.squeeze().cpu().numpy(),  nside)

        baseline     .append(spatial_metrics(pred_real_map, target_map,
                                             target_map - pred_real_map, full_mask))
        zero_results .append(spatial_metrics(pred_zero_map, target_map,
                                             target_map - pred_zero_map, full_mask))
        pred_vs_pred .append(float(np.corrcoef(pred_real_map, pred_zero_map)[0, 1]))

    summary = {
        'rmse_baseline_mean'   : float(np.mean([r['rmse']      for r in baseline])),
        'rmse_zero_mean'       : float(np.mean([r['rmse']      for r in zero_results])),
        'pearson_baseline_mean': float(np.mean([r['pearson_r'] for r in baseline])),
        'pearson_zero_mean'    : float(np.mean([r['pearson_r'] for r in zero_results])),
        'pred_vs_pred_mean'    : float(np.mean(pred_vs_pred)),
    }

    print(f"\n--- Test 1: Inputs vacíos (N={n}) ---")
    print(f"  RMSE baseline (input real) : {summary['rmse_baseline_mean']:.2f}")
    print(f"  RMSE con input zeros       : {summary['rmse_zero_mean']:.2f}")
    print(f"  Pearson(pred_real, pred_zero): {summary['pred_vs_pred_mean']:.4f}")

    return {
        'baseline'             : baseline,
        'zero'                 : zero_results,
        'pearson_pred_vs_pred' : pred_vs_pred,
        'summary'              : summary,
    }

# ==============================================================================
# Test 2 — Inputs aleatorios
# ==============================================================================

def evaluate_on_random_inputs(
    model: nn.Module,
    dataset: RMDataset,
    checkpoint_path: str,
    n_samples: int = 50,
    nside: int = 16,
    seed: int = 0,
) -> dict:
    """
    Sustituye los inputs por ruido gaussiano con la misma media y std por canal,
    y mide cómo se degrada la predicción frente al baseline real.
    """
    rng = np.random.default_rng(seed)
    n   = min(n_samples, len(dataset))

    target_scale = _load_target_scale(checkpoint_path)
    device       = next(model.parameters()).device
    rm_scale     = dataset.rm_scale

    full_mask = np.ones(12 * nside ** 2, dtype=bool)

    model.eval()
    baseline, random_results, pred_vs_pred = [], [], []

    for idx in range(n):
        inp, tgt = dataset[idx]

        # Input "fake" gaussiano: misma media y std por canal que el original
        inp_fake = torch.empty_like(inp)
        for ch in range(inp.shape[0]):
            mu, sd = inp[ch].mean().item(), inp[ch].std().item()
            inp_fake[ch] = torch.from_numpy(
                rng.normal(loc=mu, scale=sd, size=inp[ch].shape).astype(np.float32)
            )

        pred_real = _predict_batch(model, inp.unsqueeze(0),     target_scale, device)
        pred_fake = _predict_batch(model, inp_fake.unsqueeze(0), target_scale, device)

        target_map     = map_2d_to_healpix((tgt * rm_scale).squeeze().numpy(), nside)
        pred_real_map  = map_2d_to_healpix(pred_real.squeeze().cpu().numpy(),  nside)
        pred_fake_map  = map_2d_to_healpix(pred_fake.squeeze().cpu().numpy(),  nside)

        baseline      .append(spatial_metrics(pred_real_map, target_map,
                                              target_map - pred_real_map, full_mask))
        random_results.append(spatial_metrics(pred_fake_map, target_map,
                                              target_map - pred_fake_map, full_mask))
        pred_vs_pred  .append(float(np.corrcoef(pred_real_map, pred_fake_map)[0, 1]))

    summary = {
        'rmse_baseline_mean'      : float(np.mean([r['rmse']      for r in baseline])),
        'rmse_random_mean'        : float(np.mean([r['rmse']      for r in random_results])),
        'pearson_baseline_mean'   : float(np.mean([r['pearson_r'] for r in baseline])),
        'pearson_random_mean'     : float(np.mean([r['pearson_r'] for r in random_results])),
        'pred_vs_pred_mean'       : float(np.mean(pred_vs_pred)),
    }

    print(f"\n--- Test 2: Inputs aleatorios (N={n}) ---")
    print(f"  RMSE baseline (input real)   : {summary['rmse_baseline_mean']:.2f}")
    print(f"  RMSE con input gaussiano     : {summary['rmse_random_mean']:.2f}")
    print(f"  Pearson(pred_real, pred_fake): {summary['pred_vs_pred_mean']:.4f}")

    return {
        'baseline'             : baseline,
        'random'               : random_results,
        'pearson_pred_vs_pred' : pred_vs_pred,
        'summary'              : summary,
    }


# ==============================================================================
# Test 3 — Inputs barajados espacialmente
# ==============================================================================

def evaluate_on_shuffled_inputs(
    model: nn.Module,
    dataset: RMDataset,
    checkpoint_path: str,
    n_samples: int = 50,
    nside: int = 16,
    seed: int = 0,
) -> dict:
    """
    Permuta los píxeles dentro de cada canal del input (mantiene la
    distribución marginal pero destruye la geometría espacial).

    Si la red sigue prediciendo bien, está ignorando estructura espacial
    y usando solo estadísticas globales del input.
    """
    rng = np.random.default_rng(seed)
    n   = min(n_samples, len(dataset))

    target_scale = _load_target_scale(checkpoint_path)
    device       = next(model.parameters()).device
    rm_scale     = dataset.rm_scale

    full_mask = np.ones(12 * nside ** 2, dtype=bool)

    model.eval()
    baseline, shuffled_results, pred_vs_pred = [], [], []

    for idx in range(n):
        inp, tgt = dataset[idx]

        # Permutar los píxeles dentro de cada canal por separado
        inp_shuf = torch.empty_like(inp)
        for ch in range(inp.shape[0]):
            flat       = inp[ch].numpy().ravel()
            perm       = rng.permutation(flat.size)
            inp_shuf[ch] = torch.from_numpy(flat[perm].reshape(inp[ch].shape))

        pred_real = _predict_batch(model, inp.unsqueeze(0),      target_scale, device)
        pred_shuf = _predict_batch(model, inp_shuf.unsqueeze(0), target_scale, device)

        target_map    = map_2d_to_healpix((tgt * rm_scale).squeeze().numpy(), nside)
        pred_real_map = map_2d_to_healpix(pred_real.squeeze().cpu().numpy(),  nside)
        pred_shuf_map = map_2d_to_healpix(pred_shuf.squeeze().cpu().numpy(),  nside)

        baseline        .append(spatial_metrics(pred_real_map, target_map,
                                                target_map - pred_real_map, full_mask))
        shuffled_results.append(spatial_metrics(pred_shuf_map, target_map,
                                                target_map - pred_shuf_map, full_mask))
        pred_vs_pred    .append(float(np.corrcoef(pred_real_map, pred_shuf_map)[0, 1]))

    summary = {
        'rmse_baseline_mean'      : float(np.mean([r['rmse']      for r in baseline])),
        'rmse_shuffled_mean'      : float(np.mean([r['rmse']      for r in shuffled_results])),
        'pearson_baseline_mean'   : float(np.mean([r['pearson_r'] for r in baseline])),
        'pearson_shuffled_mean'   : float(np.mean([r['pearson_r'] for r in shuffled_results])),
        'pred_vs_pred_mean'       : float(np.mean(pred_vs_pred)),
    }

    print(f"\n--- Test 3: Inputs barajados espacialmente (N={n}) ---")
    print(f"  RMSE baseline (input real)   : {summary['rmse_baseline_mean']:.2f}")
    print(f"  RMSE con input barajado      : {summary['rmse_shuffled_mean']:.2f}")
    print(f"  Pearson(pred_real, pred_shuf): {summary['pred_vs_pred_mean']:.4f}")

    return {
        'baseline'             : baseline,
        'shuffled'             : shuffled_results,
        'pearson_pred_vs_pred' : pred_vs_pred,
        'summary'              : summary,
    }
