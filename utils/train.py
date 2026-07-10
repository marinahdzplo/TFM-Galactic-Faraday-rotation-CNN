"""
train.py — Entrenamiento de la red.
 
train_model recibe el modelo ya instanciado y devuelve (model, history)
donde history es un dict serializable a JSON:
    modo estándar → {'mode':'standard', 'train':[...], 'val':[...], 'best_val':float, ...}
    modo CV       → {'mode':'kfold', 'k_folds':int, 'train':[[...]], 'val':[[...]],
                     'best_val':float, 'final_val_mean':float, 'final_val_std':float, ...}
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset

from utils.data import RMDataset

# ==============================================================================
# Funciones internas
# ==============================================================================

def _set_seed(seed: int) -> None:
    """Fija la semilla para reproducibilidad estricta."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
 

def _build_criterion(loss_fn: str) -> nn.Module:
    """
    Construye la función de pérdida pixel a pixel.

    Opciones
    --------
    'mse'   : MSE clásico. Penaliza más los errores grandes.
    'mae'   : L1. Más robusto a outliers, gradiente constante.
    'huber' : Smooth L1. MSE cerca de 0 y MAE lejos.

    Parameters
    ----------
    loss_fn : str
              Nombre de la pérdida ('mse', 'mae' o 'huber').

    Returns
    -------
    nn.Module
        Módulo de pérdida instanciado.

    Raises
    ------
    ValueError : si loss_fn no es una de las opciones válidas.
    """
    bases = {
        'mse'  : lambda: nn.MSELoss(),
        'mae'  : lambda: nn.L1Loss(),
        'huber': lambda: nn.SmoothL1Loss(beta=1.0),
    }
    name = loss_fn.lower()
    if name not in bases:
        raise ValueError(
            f"Loss desconocida: '{loss_fn}'. Opciones: {list(bases)}"
        )
    return bases[name]()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """
    Ejecuta una época completa (entrenamiento o evaluación).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer or None
                Si None, se ejecuta en modo evaluación (sin gradientes).

    Returns
    -------
    float
        Pérdida media por muestra sobre todo el dataset.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)

def _plot_learning_curve(
    train_losses: list, 
    val_losses: list, 
    save_path: str, 
    k_folds: int | None = None,
    loss_fn: str = 'mse',
) -> None:
    """
    Guarda la curva de aprendizaje en disco.
 
    Parameters
    ----------
    train_losses : list or list of lists
                   Pérdidas de entrenamiento. Lista de floats (modo estándar)
                   o lista de listas (modo k-fold).
    val_losses   : list or list of lists
                   Pérdidas de validación. Igual estructura que train_losses.
    save_path    : str
                   Ruta de salida de la figura (PNG).
    k_folds      : int or None
                   Si no es None, dibuja media ± std sobre los folds.
    loss_fn      : str
                   Nombre de la métrica para las etiquetas del eje Y.
    """
    metric  = loss_fn.upper()
    fig, ax = plt.subplots(figsize=(8, 5))

    if k_folds is None:
        # Modo estándar: una sola curva de train y una de val
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label=f'Train {metric}', color='steelblue')
        ax.plot(epochs, val_losses,   label=f'Val {metric}',   color='tomato')
    else:
        # Modo k-fold: media ± std sobre los folds
        max_len = max(len(x) for x in train_losses)

        train_arr = np.full((len(train_losses), max_len), np.nan, dtype=np.float64)
        val_arr   = np.full((len(val_losses),   max_len), np.nan, dtype=np.float64)

        for i, seq in enumerate(train_losses):
            train_arr[i, :len(seq)] = seq
        for i, seq in enumerate(val_losses):
            val_arr[i, :len(seq)] = seq

        epochs = range(1, max_len + 1)
        for arr, color, label in [
            (train_arr, 'steelblue', f'Train {metric} (media {k_folds} folds)'),
            (val_arr,   'tomato',    f'Val {metric} (media {k_folds} folds)'),
        ]:
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            ax.plot(epochs, mean, label=label, color=color)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel('Época')
    ax.set_ylabel(metric)
    ax.set_title('Curva de aprendizaje')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Curva de aprendizaje guardada en: {save_path}")

def _make_loaders(
    train_ds, 
    val_ds, 
    batch_size: int, 
    device: torch.device,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    
    """
    Construye los DataLoader de entrenamiento y validación.
 
    Parameters
    ----------
    num_workers : int
                  Procesos paralelos de carga de datos. Default: 4.
    """

    pin = device.type == 'cuda'

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader

def _print_epoch_progress(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    current_lr: float,
    loss_name: str,
) -> None:
    """Imprime el progreso de la época actual."""

    print(
        f"Epoch {epoch:3d}/{total_epochs} | "
        f"Train {loss_name.upper()}: {train_loss:.6f} | "
        f"Val {loss_name.upper()}: {val_loss:.6f} | "
        f"LR: {current_lr:.2e}"
    )

# ==============================================================================
# Entrenamiento principal
# ==============================================================================

def train_model(
    train_dir: str,
    val_dir: str,
    model: nn.Module,
    dataset_class: type = RMDataset,
    checkpoint_path: str = 'best_model.pt',
    curve_path: str = 'learning_curve.png',
    resume_from: str | None = None,
    device: str | None = None,
    num_workers: int = 4,
    k_folds: int | None = None,
    patience: int = 50,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    min_lr_ratio: float = 0.01,
    loss_fn: str = 'mse',
    seed: int = 42,
) -> tuple[nn.Module, dict]:   

    """
    Entrena el modelo y devuelve (modelo_con_mejores_pesos, historial_dict).
 
    Parameters
    ----------
    train_dir       : str
                      Carpeta con los archivos .npz de entrenamiento.
    val_dir         : str
                      Carpeta con los archivos .npz de validación.
    model           : nn.Module
                      Modelo ya instanciado.
    checkpoint_path : str
                      Ruta del archivo .pt donde se guarda el mejor modelo. Default: 'best_model.pt'.
    
    curve_path      : str
                      Ruta donde guardar la figura de la curva de aprendizaje. Default: 'learning_curve.png'.   
    resume_from     : str or None
                      Si se proporciona, reanuda el entrenamiento desde este checkpoint.
                      Incompatible con k_folds. Default: None.     
    device          : str or None
                      Dispositivo de cómputo ('cpu', 'cuda', 'mps'...).
                      Si None, usa CUDA si está disponible; si no, aborta con un mensaje claro.    
    num_workers     : int
                      Procesos paralelos de carga de datos. Usar 0 en Windows. Default: 4.   
    k_folds         : int or None
                      Si no es None, usa k-fold cross-validation en lugar del split estándar.
                      Debe ser ≥ 2. Incompatible con resume_from. Default: None.   
    patience        : int
                      Épocas sin mejora en val loss antes de aplicar early stopping. Default: 50.                                                                                                            
    epochs          : int
                      Número máximo de épocas de entrenamiento. Default: 50.
    batch_size      : int
                      Tamaño del minibatch. Default: 32.
    lr              : float
                      Tasa de aprendizaje inicial del optimizador Adam. Default: 1e-3.
    min_lr_ratio    : float
                      La tasa de aprendizaje mínima del scheduler es learning_rate * min_lr_ratio.
                      Default: 0.01 (decae hasta lr/100).            
    loss_fn         : str
                      Función de pérdida. Opciones: 'mse', 'mae', 'huber'.
                      Default: 'mse'.
    seed            : int
                      Semilla para reproducibilidad. Default: 42.
    
    Returns
    -------
    model   : nn.Module
              Modelo cargado con los mejores pesos encontrados.
    history : dict
              Historial de pérdidas, serializable a JSON.
 
    Raises
    ------
    ValueError : si los parámetros de entrada son inválidos.
    SystemExit : si no hay GPU y device=None (el entrenamiento no se realiza en CPU).
    """

    if epochs < 1:
        raise ValueError(f"'epochs' debe ser ≥ 1, se recibió: {epochs}")
    if batch_size < 1:
        raise ValueError(f"'batch_size' debe ser ≥ 1, se recibió: {batch_size}")
    if lr <= 0:
        raise ValueError(f"'lr' debe ser positivo, se recibió: {lr}")
    if not 0.0 < min_lr_ratio < 1.0:
        raise ValueError(
            f"'min_lr_ratio' debe estar en (0, 1), se recibió: {min_lr_ratio}"
        )
    if k_folds is not None and k_folds < 2:
        raise ValueError(f"'k_folds' debe ser ≥ 2 si se usa, se recibió: {k_folds}")
    if k_folds is not None and resume_from is not None:
        raise ValueError("'resume_from' no es compatible con 'k_folds'.")
 

    if device is None:
        if not torch.cuda.is_available():
            print(
                "ERROR: No se ha detectado GPU (CUDA). "
                "Pasa device='cpu' explícitamente si quieres entrenar en CPU. "
            )
            sys.exit(1)
        device = 'cuda'

    dev = torch.device(device)
    model = model.to(dev)
    print(f"Entrenando en dispositivo: {dev}")
    print(f"Semilla        : {seed}")
    print(f"Épocas máximas : {epochs}")
    print(f"Batch size     : {batch_size}")
    print(f"Learning rate  : {lr}")
    print(f"Función pérdida: {loss_fn}")
    print(f"Patience       : {patience}")

    if k_folds is None:
        return _train_standard(
            train_dir       = train_dir,
            val_dir         = val_dir,
            model           = model,
            dataset_class   = dataset_class,
            checkpoint_path = checkpoint_path,
            curve_path      = curve_path,
            resume_from     = resume_from,
            dev             = dev,
            num_workers     = num_workers,
            patience        = patience,
            epochs          = epochs,
            batch_size      = batch_size,
            lr              = lr,
            min_lr_ratio    = min_lr_ratio,
            loss_fn         = loss_fn,
            seed            = seed,
        )
    else:
        return _train_kfold(
            train_dir       = train_dir,
            val_dir         = val_dir,
            model           = model,
            dataset_class   = dataset_class,
            checkpoint_path = checkpoint_path,
            curve_path      = curve_path,
            dev             = dev,
            num_workers     = num_workers,
            k_folds         = k_folds,
            patience        = patience,
            epochs          = epochs,
            batch_size      = batch_size,
            lr              = lr,
            min_lr_ratio    = min_lr_ratio,
            loss_fn         = loss_fn,
            seed            = seed,
        )

# ==============================================================================
# Modo estándar
# ==============================================================================

def _train_standard(
    train_dir: str,
    val_dir: str,
    model: nn.Module,
    dataset_class: type,
    checkpoint_path: str,
    curve_path: str,
    resume_from: str | None,
    dev: torch.device,
    num_workers: int,
    patience: int,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr_ratio: float,
    loss_fn: str,
    seed: int = 42,
) -> tuple[nn.Module, dict]:
    """Entrenamiento estándar con conjuntos train/val fijos."""

    train_ds = dataset_class(train_dir)
    val_ds   = dataset_class(val_dir)

    target_scale = float(train_ds.rm_scale)
    print(f"  target_scale leído del dataset: {target_scale}")

    train_loader, val_loader = _make_loaders(
        train_ds, val_ds, batch_size, dev, num_workers
    )
 
    criterion = _build_criterion(loss_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * min_lr_ratio,
    )
 
    start_epoch       = 0
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    stopped_at_epoch  = epochs

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=dev)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"Reanudando desde epoch {start_epoch} "
              f"(mejor val loss: {best_val_loss:.6f})")

    train_history, val_history = [], []

    for epoch in range(start_epoch, epochs):
        tl = _run_epoch(model, train_loader, criterion, optimizer, dev)
        vl = _run_epoch(model, val_loader,   criterion, None,      dev)
        scheduler.step()

        train_history.append(float(tl))
        val_history.append(float(vl))

        _print_epoch_progress(
            epoch + 1, epochs, tl, vl, scheduler.get_last_lr()[0], loss_fn
        )

        if vl < best_val_loss:
            best_val_loss = vl
            epochs_no_improve = 0
            torch.save({
                'epoch'         : epoch,
                'model'         : model.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'scheduler'     : scheduler.state_dict(),
                'best_val_loss' : best_val_loss,
                'target_scale'  : target_scale,
                'seed'          : seed,
            }, checkpoint_path)
            print(f"  -> Checkpoint guardado (val: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_at_epoch = epoch + 1
                print(f"  -> Early stopping tras {patience} épocas sin mejora.")
                break

    _plot_learning_curve(train_history, val_history, curve_path, k_folds=None, loss_fn=loss_fn)

    model.load_state_dict(torch.load(checkpoint_path, map_location=dev)['model'])

    return model, {
        'mode'    : 'standard',
        'seed'    : seed,
        'train'   : train_history,
        'val'     : val_history,
        'best_val': float(best_val_loss),
        'loss_fn' : loss_fn,
        'patience' : patience,
        'stopped_epoch' : stopped_at_epoch,
    }

# ==============================================================================
# Modo k-fold CV
# ==============================================================================

def _train_kfold(
    train_dir: str,
    val_dir: str,
    model: nn.Module,
    dataset_class: type,
    checkpoint_path: str,
    curve_path: str,
    dev: torch.device,
    num_workers: int,
    k_folds: int,
    patience: int,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr_ratio: float,
    loss_fn: str,
    seed: int = 42
) -> tuple[nn.Module, dict]:
    """Entrenamiento con k-fold cross-validation."""

    train_ds0 = dataset_class(train_dir)
    val_ds0   = dataset_class(val_dir)

    target_scale = float(train_ds0.rm_scale)
    print(f"  target_scale leído del dataset: {target_scale}")

    full_ds   = ConcatDataset([train_ds0, val_ds0])
    indices   = np.arange(len(full_ds))
    kf        = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
 
    criterion = _build_criterion(loss_fn)
 
    # Guardar estado inicial del modelo para reiniciarlo en cada fold
    init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    all_train, all_val = [], []
    best_global        = float('inf')
    stopped_epochs     =  []

    print(f"\nIniciando {k_folds}-fold CV ({len(full_ds)} muestras)\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"{'─'*52}\n  Fold {fold+1}/{k_folds}  "
              f"(train: {len(train_idx)}, val: {len(val_idx)})\n{'─'*52}")

        # Reiniciar el modelo a su estado inicial antes de cada fold
        model.load_state_dict({k: v.to(dev) for k, v in init_state.items()})

        train_loader, val_loader = _make_loaders(
            Subset(full_ds, train_idx), Subset(full_ds, val_idx), batch_size, dev, num_workers
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * min_lr_ratio,
        )

        fold_train, fold_val = [], []
        fold_best            = float('inf')
        epochs_no_improve = 0
        stopped_at_epoch = epochs

        for epoch in range(epochs):
            tl = _run_epoch(model, train_loader, criterion, optimizer, dev)
            vl = _run_epoch(model, val_loader,   criterion, None,      dev)
            scheduler.step()
            fold_train.append(float(tl))
            fold_val.append(float(vl))

            _print_epoch_progress(
                epoch + 1, epochs, tl, vl, scheduler.get_last_lr()[0], loss_fn
            )

            if vl < fold_best:
                fold_best = vl
                epochs_no_improve = 0
                if vl < best_global:
                    best_global = vl
                    torch.save({
                        'fold': fold, 'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_global,
                        'target_scale': target_scale,
                        'seed'          : seed,
                    }, checkpoint_path)
                    print(f"  -> Mejor global (fold {fold+1}, val: {best_global:.6f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stopped_at_epoch = epoch + 1
                    print(f"  -> Early stopping en fold {fold+1} tras "
                          f"{patience} épocas sin mejora.")
                    break

        stopped_epochs.append(stopped_at_epoch)
        all_train.append(fold_train)
        all_val.append(fold_val)
        print(f"  Fold {fold+1} completado. Mejor val: {fold_best:.6f}\n")

    final_vals = [h[-1] for h in all_val]
    mean_val   = float(np.mean(final_vals))
    std_val    = float(np.std(final_vals))

    print(f"\n{'═'*52}\n  CV completada  |  "
          f"Val final: {mean_val:.6f} ± {std_val:.6f}\n{'═'*52}\n")

    _plot_learning_curve(all_train, all_val, curve_path, k_folds=k_folds, loss_fn=loss_fn)

    model.load_state_dict(torch.load(checkpoint_path, map_location=dev)['model'])

    return model, {
        'mode'           : 'kfold',
        'seed'           : seed,
        'k_folds'        : k_folds,
        'train'          : all_train,
        'val'            : all_val,
        'best_val'       : float(best_global),
        'final_val_mean' : mean_val,
        'final_val_std'  : std_val,
        'loss_fn'        : loss_fn,
        'patience'       : patience,
        'stopped_epochs' : stopped_epochs,
    }

# ==============================================================================
# Inferencia
# ==============================================================================

def predict(
    model:  nn.Module,
    noisy_tensor: torch.Tensor,
    checkpoint_path: str,
    device: str | None = None,
) -> torch.Tensor:
    """
    Aplica el modelo a un mapa ruidoso y deshace el escalado del target.
 
    Parameters
    ----------
    model        : nn.Module
                   Modelo entrenado en modo evaluación.
    noisy_tensor : torch.Tensor, shape (1, H, W) o (1, 1, H, W)
                   Mapa ruidoso ya dividido por qu_scale (como en el dataset).
    checkpoint_path : str
                      Ruta al .pt guardado por train_model. Se lee 'target_scale' de él.
    device       : str or None
                   Dispositivo de inferencia. Si None, usa CUDA si está disponible. Default: None.
 
    Returns
    -------
    torch.Tensor, shape (1, 1, H, W)
        Mapa reconstruido en las unidades originales (rad/m^2).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=dev)
    if 'target_scale' not in ckpt:
        raise KeyError(
            f"El checkpoint '{checkpoint_path}' no contiene 'target_scale'. "
            f"Puede ser un checkpoint antiguo generado antes de la unificación de escalas."
        )
    target_scale = float(ckpt['target_scale'])

    model.eval()

    if noisy_tensor.ndim == 3:
        noisy_tensor = noisy_tensor.unsqueeze(0) # añadir dimensión de batch

    if noisy_tensor.ndim != 4:
        raise ValueError(
            f"'noisy_tensor' debe tener 3 o 4 dimensiones, "
            f"pero se recibió shape {tuple(noisy_tensor.shape)}"
        )  
    
    with torch.no_grad():
        pred_scaled = model(noisy_tensor.to(dev))

    return pred_scaled * target_scale