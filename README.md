#  Transformer Federado para Análisis de Tráfico de Red
**Proyecto del curso de Inteligencia Artificial (C++)**  
**Integrantes:** Paolo Jesús Mostajo Alor · Alexander Carpio Mamani · Anthony Briceño Quiroz

---

##  Problema
Los sistemas de detección de intrusos (IDS) necesitan grandes volúmenes de tráfico de red para entrenar modelos robustos. Centralizar esos datos:
- puede ser **costoso** y **lento**,
- **afecta la privacidad**,
- y no siempre es posible en escenarios distribuidos (distintas organizaciones o redes).

---

##  Enfoque propuesto
Implementamos un **modelo tipo Transformer** para la detección de ataques en el dataset **NSL‑KDD**, y lo usamos en un esquema de **aprendizaje federado**:

- Cada cliente entrena localmente un pequeño **MLP en CUDA** sobre la representación generada por el Transformer.
- Solo se comparten los **pesos del MLP** (no los datos crudos).
- Un servidor central realiza un **promedio de pesos (FedAvg)** y evalúa el modelo global.

**Ventajas principales**
- **Privacidad:** los datos permanecen en cada cliente.
- **Menor comunicación:** se envían solo parámetros, no todo el dataset.
- **Mejor generalización:** el modelo global ha visto patrones de varios clientes.

---

##  Dataset: NSL‑KDD
- Dataset clásico para detección de intrusiones.  
- En este proyecto se utilizan los archivos:
  - `NSL_KDD-master/KDDTrain+.txt`
  - `NSL_KDD-master/KDDTest+.txt`

Cada registro de conexión se transforma en **4 tokens**:
- Token 0: one‑hot de `protocol_type`, `service` y `flag`.
- Tokens 1–3: subconjuntos de atributos numéricos normalizados (duración, bytes, conteos, etc.).

Estos 4 tokens son la secuencia de entrada del Transformer.

---

##  Arquitectura del modelo

### 1. TokenEmbedding (`transformer.h`)
Convierte cada uno de los 4 tokens en un vector de dimensión `d_model = 64`:
- Para cada token se aplica una matriz de pesos distinta.
- La salida es una secuencia de tamaño `4 × 64`.

### 2. Bloque Transformer (`transformer.h`)
Implementado desde cero en C++:
- **MultiHeadAttention** con varias cabezas de atención.
- **FeedForward** totalmente conectada con activación ReLU.
- **LayerNorm** y conexiones residuales.

Se obtiene una representación por token; para clasificación usamos el primer token (`H[0]`) como vector “CLS”.

### 3. Clasificador MLP en CUDA (`transformer.cu` / `average.cu`)
Un MLP pequeño implementado con kernels CUDA:
- Capa densa 64 → 128 con ReLU.
- Capa densa 128 → `num_classes`.
- Softmax + cross‑entropy para la pérdida.
- Optimizador **Adam** implementado también en CUDA.

---

##  Entrenamiento centralizado (`transformer.cu`)

El archivo `transformer.cu` entrena el modelo de forma **centralizada**:
1. Carga `KDDTrain+.txt` y `KDDTest+.txt` desde `NSL_KDD-master/`.
2. Preprocesa el dataset y construye los 4 tokens por ejemplo (`NSLKDD`).
3. Pasa cada ejemplo por:
   - `TokenEmbedding` → `Transformer` → vector CLS (64).
   - MLP en CUDA (batch de tamaño configurable).
4. Entrena el MLP con Adam durante varias épocas.
5. Evalúa en el conjunto de prueba y muestra la **accuracy**.
6. Guarda los pesos del MLP en `mlp_model.bin`:
   - `MLP_CUDA::save_weights("mlp_model.bin")`.

Este flujo sirve como baseline centralizado y como base para los clientes federados.

---

##  Aprendizaje federado con 3 clientes (`average.cu` + Colab)

La parte federada se centra en el **MLP final**:

1. **Clientes (por ejemplo, 3 Colabs)**
   - Cada Colab entrena un MLP con la misma arquitectura (`MLP_CUDA` o equivalente) sobre un subconjunto de datos.
   - Al finalizar, guarda sus pesos en un archivo binario, por ejemplo:
     - `mlp_model_1.bin`
     - `mlp_model_2.bin`
     - `mlp_model_3.bin`
   - Solo se envían estos archivos al servidor (no los datos).

2. **Servidor de agregación (`average.cu`)**
   - Carga el conjunto de prueba `KDDTest+.txt`.
   - Crea un modelo `MLP_CUDA mlp_avg(...)` y pone todos sus pesos en cero (`zero_weights()`).
   - Para cada archivo `mlp_model_i.bin`:
     - Crea un MLP temporal y carga los pesos.
     - Suma sus pesos al acumulador con `accumulate_mlp(mlp_avg, tmp)`.
   - Al final divide todos los pesos acumulados entre K (número de clientes) con `divide_mlp(mlp_avg, K)`:
     - Esto implementa **Federated Averaging (FedAvg)**:
       \[
       W_{\text{global}} = \frac{1}{K} \sum_{k=1}^{K} W_k
       \]
   - Evalúa `mlp_avg` sobre el conjunto de prueba usando el mismo flujo:
     - `TokenEmbedding` → `Transformer` → CLS → `mlp_avg.predict`.
   - Imprime la **accuracy final del modelo federado**.

En resumen: el servidor construye un modelo global a partir del promedio de los pesos entrenados en 3 clientes distintos.

---

##  Archivos principales del proyecto
- `transformer.h`  
  Implementación del embedding de tokens, capas Transformer (atención multi‑cabeza, feedforward, layernorm) y el contenedor `Transformer`.

- `transformer.cu`  
  Entrenamiento centralizado de Transformer + MLP en CUDA sobre NSL‑KDD, y guardado de pesos en `mlp_model.bin`.

- `average.cu`  
  Carga el conjunto de prueba, recibe 3 modelos MLP entrenados (`mlp_model_1.bin`, `mlp_model_2.bin`, `mlp_model_3.bin`), realiza FedAvg y evalúa la accuracy final.

- `finalTransformer.ipynb`  
  Notebook de Colab para entrenar el modelo en la nube (por ejemplo, simulando cada cliente federado) y exportar los archivos `.bin` de pesos.

---

##  Requisitos
- **CUDA** con soporte para `nvcc` (probado en versiones recientes).
- **Compilador C++17** (por ejemplo, `g++`).
- Dataset **NSL‑KDD** ubicado en `NSL_KDD-master/` con los archivos:
  - `KDDTrain+.txt`
  - `KDDTest+.txt`
- (Opcional) Google Colab para entrenar los modelos de cada cliente y descargar los `.bin`.

---

##  Cómo compilar y ejecutar

Cómo compilar y ejecutar
Compilación básica (ejemplo):

usar el finalTransformer.ipynb en 3 colabs diferentes, donde se entrenaran 3 veces el modelo transformer.cu, luego descargar el mlp_model.bin y cambiarle a mlp_model_1.bin, y asi sucevivamente con los 3 modelos, mlp_model_2.bin y mlp_model_3.bin. Luego subir a cualquier .ipynb esos 3 puntos bin, y correr el average.cu, y se verá el desempeño colaborativo de los 3 modelos, y como mejora.
---

