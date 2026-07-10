# TFM-Galactic-Faraday-rotation-CNN

Reconstrucción bidimensional del mapa de Medida de Rotación (RM) de la Vía Láctea a partir de mapas multifrecuencia de los parámetros de Stokes $Q$ y $U$ en el régimen de microondas, mediante una red neuronal convolucional de tipo U-Net.

Código asociado al Trabajo de Fin de Máster *«Extracción de la Rotación de Faraday galáctica mediante Deep Learning»*.

---

## Descripción

La rotación de Faraday es el principal trazador observacional del campo magnético galáctico, y su caracterización resulta relevante para la cosmología, ya que el efecto Faraday mezcla los modos E y B del fondo cósmico de microondas e introduce una contaminación espuria en la búsqueda de modos B primordiales.

La reconstrucción del cielo de Faraday se ha apoyado tradicionalmente en catálogos de fuentes puntuales o en sondeos de emisión difusa a bajas frecuencias. Este trabajo aborda, hasta donde alcanza nuestro conocimiento por primera vez, la reconstrucción bidimensional del mapa de RM a partir de la emisión de sincrotrón polarizada en el rango de microondas.

Ante la ausencia de una verdad observacional continua sobre el cielo, se genera un conjunto de datos sintético que combina:

- **Señal**: emisión de sincrotrón polarizada simulada con [PySM3](https://pysm3.readthedocs.io/).
- **Variabilidad**: rotación del plano de polarización con un ángulo $\psi \in [0,\pi)$ muestreado de forma aleatoria e independiente para cada canal de frecuencia.
- **Ruido**: modelo instrumental de ruido blanco y $1/f$ derivado de las propiedades de seis canales de QUIJOTE, WMAP y Planck (11–44 GHz).
- **Objetivo**: mapa de RM de [Hutschenreuter et al. (2020)](https://doi.org/10.1051/0004-6361/202140486) como referencia supervisada.

La reconstrucción de RM se realiza a partir de una U-Net de cuatro niveles, adaptada a la geometría esférica mediante una proyección del esquema HEALPix a un tensor bidimensional de $64\times48$ píxeles ($N_\mathrm{side}=16$).

### Resultados principales

Sobre un conjunto de evaluación independiente:

| Métrica | Valor |
|---|---|
| Correlación de Pearson (cielo completo) | $0.9921 \pm 0.0039$ |
| RMSE medio | $7.84 \pm 1.81$ rad m⁻² |
| Sesgo | $+0.15 \pm 0.28$ rad m⁻² |

---

## Estructura del repositorio

```
.
├── utils/
│   ├── __init__.py
│   ├── data.py             # Preprocesado HEALPix, generación del dataset, RMDataset
│   ├── model.py            # Arquitectura UNetDenoiser (ConvBlock, UpBlock)
│   ├── train.py            # Bucle de entrenamiento, early stopping, k-fold, predict
│   ├── eval.py             # Evaluación espacial y armónica, figuras de diagnóstico
│   ├── fn_eval.py          # Métricas: máscaras, pseudo-C_ℓ, RMSE/bias/Pearson
│   ├── fn_diagnostics.py   # Tests de interpretabilidad (memorización vs. uso del input)
│   └── faraday2020v2.fits  # Mapa de RM de Hutschenreuter et al. (2020)
└── README.md
└── requirements.txt
└── LICENSE.txt
```

> **Nota**: los módulos importan mediante `from utils.data import ...`, por lo que deben residir en un paquete `utils/`. Ejecuta los scripts desde la raíz del repositorio.

---

## Instalación

```bash
git clone https://github.com/marinahdzplo/TFM-Galactic-Faraday-rotation-CNN.git
cd TFM-Galactic-Faraday-rotation-CNN

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencias

```
numpy
scipy
torch
healpy
astropy
pysm3
scikit-learn
matplotlib
```

`healpy` no está disponible para Windows, en ese caso se recomienda WSL o conda-forge.

## Uso

### 1. Generación del conjunto de datos

```python
from utils.data import generate_random_rm_dataset

generate_random_rm_dataset(
    rm_fits_path         = "data/faraday2020v2.fits",
    output_dir           = "dataset/",
    n_noise_realizations = 500,
    seed                 = 42,
    nside                = 16,        # resolución operativa de la red
    pysm_nside           = 256,       # resolución de simulación previa al degradado
    pysm_preset          = "s1",
    freqs_ghz            = (11, 17, 19, 23, 30, 44),
    split_ratio          = 0.8,
    sensitivity_mK       = [1.3710, 1.4895, 1.6695, 1.4350, 0.0065, 0.0086],
    lknee                = [3.81, 3.35, 8.34, 5.88, 3.42, 1.59],
    slope                = [1.95, 1.73, 1.34, 1.13, 0.92, 0.88],
    smooth_fwhm_deg      = 1.0,
    rm_scale             = 100.0,
)
```

Genera `dataset/train/` y `dataset/val/` con muestras `.npz`. Cada muestra contiene el tensor de entrada `(12, 64, 48)` —pares $(Q,U)$ de las seis frecuencias— y el objetivo `(1, 64, 48)`.

La variabilidad entre realizaciones proviene del ángulo de rotación $\psi$, muestreado de manera independiente para cada frecuencia, y de las dos componentes de ruido. La señal de sincrotrón intrínseca y el mapa objetivo son idénticos en todas las muestras; el ángulo aplicado se almacena en el campo `psi_obs` de cada `.npz`.

### 2. Entrenamiento

```python
from utils.model import UNetDenoiser
from utils.train import train_model

model = UNetDenoiser(
    in_ch        = 12,     # 2 × 6 frecuencias
    out_ch       = 1,      # mapa de RM
    base_ch      = 32,
    use_residual = False,  # obligatorio si in_ch != out_ch
    use_maxpool  = False,  # downsampling por convolución con stride
)

model, history = train_model(
    train_dir       = "dataset/train",
    val_dir         = "dataset/val",
    model           = model,
    checkpoint_path = "best_model.pt",
    epochs          = 300,
    batch_size      = 32,
    lr              = 1e-3,
    loss_fn         = "mae",
    patience        = 50,
    seed            = 42,
)
```

### 3. Evaluación

```python
from utils.data import RMDataset
from utils.eval import evaluate_rm_full_val, plot_rm_recovery_diagnostics

test_ds = RMDataset("dataset/val")
results = evaluate_rm_full_val(
    model           = model,
    dataset         = test_ds,
    checkpoint_path = "best_model.pt",
    nside           = 16,
    lat_cut_deg     = 20.0,
)
plot_rm_recovery_diagnostics(results)
```

Devuelve métricas espaciales por región (cielo completo, plano galáctico $|b|<20^\circ$ y polos), el ratio de espectros $C_\ell^\mathrm{pred}/C_\ell^\mathrm{target}$, el RMS del residuo por bandas de latitud y la comparación $\sigma_\mathrm{CNN}$ frente a la varianza cósmica.

### 4. Tests de interpretabilidad

Dado que el mapa objetivo es común a todas las muestras, estos tests verifican que la red se apoya en la señal polarizada de entrada y no en la memorización completa de la geometría del objetivo:

```python
from utils.fn_diagnostics import (
    evaluate_on_zero_inputs,      # ¿predice sin input? → memorización pura
    evaluate_on_random_inputs,    # ¿ignora el input?
    evaluate_on_shuffled_inputs,  # ¿ignora la geometría espacial?
)

evaluate_on_zero_inputs(model, test_ds, "best_model.pt", n_samples=50)
```

---

## Detalles metodológicos

**Proyección HEALPix → 2D.** Las convoluciones estándar asumen una malla cartesiana. Siguiendo a [Wang et al. (2022)](https://doi.org/10.3847/1538-4365/ac5f4a), las doce caras del esquema NESTED se reorganizan en una cuadrícula de $4\times3$ caras, preservando la vecindad topológica dentro de cada cara.

**Elecciones de arquitectura.** Se emplea `PReLU` en lugar de `ReLU`, ya que $Q$, $U$ y la propia RM adoptan valores negativos con significado físico (estados de polarización ortogonales y orientación del campo a lo largo de la línea de visión). El sobremuestreo se realiza por interpolación bilineal en lugar de convolución transpuesta, para evitar artefactos de tipo *checkerboard* en los bordes de las caras HEALPix.

**Función de pérdida.** El error absoluto medio (MAE) ofrece un gradiente de magnitud constante, más robusto frente a los valores extremos del plano galáctico que el error cuadrático.

---

## Referencias

- Hutschenreuter, S., et al. (2022). *The Galactic Faraday rotation sky 2020*. **A&A**, 657, A43.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*. **MICCAI**, 234–241.
- Wang, G.-J., et al. (2022). *Recovering the CMB Signal with Machine Learning*. **ApJS**, 260, 13.
- Górski, K. M., et al. (2005). *HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere*. **ApJ**, 622, 759.
- Thorne, B., et al. (2017). *The Python Sky Model: software for simulating the Galactic microwave sky*. **MNRAS**, 469, 2821.
- Rubiño-Martín, J. A., et al. (2023). *QUIJOTE scientific results – IV. A northern sky survey in intensity and polarization at 10–20 GHz with the Multi-Frequency Instrument*. **MNRAS**, 519, 3383.

---

## Licencia

Este proyecto se distribuye bajo la [GPL-3.0 License](LICENSE). Consulta el fichero `LICENSE` para más detalles.

El mapa `faraday2020v2.fits` procede de Hutschenreuter et al. (2020) y se rige por los términos de sus autores originales.
