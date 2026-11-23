# Predicción de defectos de software bajo desbalance de clases y escasez de etiquetas - Entrega 2

Proyecto semestral de ISWZ3402 Inteligencia Artificial II (UDLA) - Grupo 9:
Enzo Cortez, Ricardo Hidalgo y Sebastián Panchi.

Código de la Entrega 2 (métodos y resultados preliminares). Evalúa un perceptrón multicapa (MLP) frente a
regresión logística (LR) y bosque aleatorio (RF) sobre los conjuntos NASA MDP
depurados (versión D''), con y sin balanceo de clases.

## Estado

| Pregunta | Qué cubre | Estado |
|---|---|---|
| RQ1 | MLP vs. LR y RF | Implementado y ejecutado (configuración inicial del MLP) |
| RQ2 | Sin balanceo vs. SMOTE | Implementado y ejecutado |
| RQ2 | Sobremuestreo con GAN (CTGAN) | En el proyecto de la Entrega 3 |
| RQ3 | Semisupervisado con 10/20/30 % de etiquetas | En el proyecto de la Entrega 3 |
| - | Ajuste de hiperparámetros (validación anidada) | En el proyecto de la Entrega 3 |

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta raíz del proyecto, en este orden:

```bash
python scripts/01_descargar_datos.py      # descarga los 6 conjuntos D'' a data/raw/MDP_D2/
python scripts/02_verificar_datos.py      # control de calidad contra la Tabla 1
python scripts/03_eda.py                  # análisis exploratorio
python scripts/04_experimento_rq1_rq2.py  # validación cruzada 10 x 5 (unos 20-40 min)
python scripts/05_analisis.py             # tablas y pruebas estadísticas
python scripts/06_figuras.py              # figura para el artículo
python -m pytest -q                       # pruebas (datos, fuga de datos, estadística)
```

El paso 4 guarda un CSV por configuración en `results/folds/` y salta las que ya
existen; si se interrumpe, basta con volver a ejecutarlo. Con `--overwrite` rehace
todo. Para pruebas rápidas: `--datasets CM1 --models LR MLP`.

## Estructura

```
src/
  config.py      rutas, conjuntos, semilla y protocolo (10 x 5)
  data.py        descarga, lectura .arff y control de calidad
  pipelines.py   preprocesamiento + balanceo + modelo en un solo Pipeline
  metrics.py     AUC, F1, exhaustividad, precisión, MCC y G-mean
  stats.py       Wilcoxon, prueba t corregida, delta de Cliff y Holm
scripts/         pasos 01 a 06
tests/           pruebas con pytest
results/         salidas (CSV y JSON)
figures/         figuras
data/            datos descargados (no se versionan)
```

## Decisiones metodológicas

- **Datos.** Versión D'' de Shepperd et al. (2013), la que sus autores recomiendan.
  El script descarga la réplica pública de GitHub mantenida por C. Tantithamthavorn.
  Falta compararla con el archivo oficial de figshare
  (https://doi.org/10.6084/m9.figshare.6071675.v1, licencia CC BY 4.0): descargar
  `MDP.zip`, descomprimir y ejecutar
  `python scripts/02_verificar_datos.py --figshare-dir "<carpeta con los .arff de D''>"`.
- **Sin fuga de datos.** La transformación log(1 + x), el escalado y SMOTE están
  dentro de un `Pipeline` de imbalanced-learn, así que en cada partición se ajustan
  solo con los datos de entrenamiento y SMOTE nunca toca la partición de prueba.
  Las pruebas en `tests/` lo comprueban.
- **Particiones pareadas.** Todas las configuraciones usan las mismas 50 particiones
  (semilla 42), lo que permite compararlas por pares.
- **MLP.** Configuración inicial fija: una capa oculta de 32 neuronas, ReLU, Adam y
  regularización L2 (alpha = 0,001). Se desactivó la parada temprana de scikit-learn
  porque vigila la exactitud, una métrica engañosa con clases desbalanceadas: en una
  prueba inicial con CM1 detenía el entrenamiento demasiado pronto (AUC de 0,60).
  El ajuste formal de hiperparámetros, con validación cruzada anidada, se hace en el proyecto de la Entrega 3.
- **Estadística.** Se reporta el Wilcoxon propuesto en el Entregable 1 y, como
  complemento, la prueba t corregida para validación cruzada repetida (Nadeau y
  Bengio, 2003), porque las 50 particiones no son independientes y el Wilcoxon sin
  corregir tiende a dar p-valores demasiado pequeños. Ambos con corrección de Holm.
  Tamaño del efecto: delta de Cliff (umbrales de Romano et al., 2006).

## Hallazgo sobre los datos

D'' conserva 7 módulos de PC4 con `LOC_TOTAL = 0` pero `NUMBER_OF_LINES` entre
1 y 38: un valor implausible que sobrevivió a la depuración, del tipo que
documentan Petrić et al. (2016). Los 7 son no defectuosos (0,6 % de PC4). Por
ahora se mantienen para no alterar D''; el grupo debe decidir si los elimina y
declararlo en el artículo.

## Salidas

| Archivo | Contenido |
|---|---|
| `results/verificacion_datos.json` | Conteos, chequeos de calidad y huella SHA-256 de cada archivo |
| `results/eda_resumen.csv` | Tamaño, desbalance y asimetría antes y después de log(1 + x) |
| `results/eda_correlaciones.csv` | 5 métricas más correlacionadas con la etiqueta (Spearman) |
| `results/folds/*.csv` | Métricas por partición (50 por configuración) |
| `results/resumen_metricas.csv` | Media y desviación estándar por conjunto, modelo y balanceo |
| `results/rq1_comparaciones.csv` | MLP vs. LR y RF, sin balanceo |
| `results/rq2_comparaciones.csv` | SMOTE vs. sin balanceo, por modelo |
| `results/run_info.json` | Versiones de las librerías y protocolo usados |
| `figures/fig_mcc_por_conjunto.png` | MCC por conjunto y modelo |

## Referencias de los métodos

- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239-281.
- Petrić, J., Bowes, D., Hall, T., Christianson, B., & Baddoo, N. (2016). The jinx on the NASA software defect data sets. *EASE 2016*.
- Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). Appropriate statistics for ordinal level data. *Annual meeting of the Florida Association of Institutional Research*.
- Shepperd, M., Song, Q., Sun, Z., & Mair, C. (2013). Data quality: Some comments on the NASA software defect datasets. *IEEE Transactions on Software Engineering, 39*(9), 1208-1215.
