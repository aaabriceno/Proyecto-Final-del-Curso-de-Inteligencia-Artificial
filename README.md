# 🧠 Transformer Federado para Análisis de Tráfico de Red
**Proyecto del curso de Inteligencia Artificial (C++)**  
**Integrantes:** Paolo Jesús Mostajo Alor · Alexander Carpio Mamani · Anthony Briceño Quiroz

---

## 🚩 Problema
Los IDS (Intrusion Detection Systems) requieren grandes volúmenes de tráfico de red para entrenar modelos robustos. Centralizar esos datos:
- es **lento** y **costoso**,
- **vulnera la privacidad**,
- y dificulta escalar a múltiples dominios/redes.

## 💡 Solución
Entrenar un **Transformer Federado**: cada nodo/cliente aprende localmente sobre su propio tráfico y solo comparte **pesos del modelo** (no los datos crudos).  
Agregamos con **FedAvg** en un servidor central y repetimos por rondas.

**Ventajas**  
- **Privacidad:** los datos nunca salen de cada cliente.  
- **Eficiencia:** se envían solo parámetros, no millones de paquetes.  
- **Escalabilidad:** más nodos ⇒ más inteligencia global.

---

## 🧱 Arquitectura (visión general)

       ┌────────────────────────────────────────────────────┐
       │                    Servidor (FedAvg)               │
       │ 1) Envía modelo global  4) Promedia pesos (FedAvg) │
       └───────────────▲───────────────────────────▲────────┘
                       │                           │
    ┌──────────────────┘                           └──────────────────┐
    │                                                              │
┌───────┴────────┐ ┌────────────┴───────┐
│ Cliente 1 │ 2) Entrena localmente con su dataset │ Cliente 2 │
│ (NSL-KDD split) │ 3) Devuelve pesos actualizados │ (NSL-KDD split) │
└─────────────────┘ └────────────────────┘



---

## 📊 Dataset
- **NSL-KDD** (Kaggle, 2019).  
  Archivos típicos: `KDDTrain+.txt`, `KDDTest+.txt`.  
  *Uso:* convertir a numérico (one-hot/label encode), normalizar, y **particionar** en 2–3 subconjuntos (uno por cliente).

> Referencia: Kaggle – “NSL-KDD” (2019).

---

## 🧩 Metodología
1. **Simular 2 o 3 clientes** federados (nodos) en **una sola máquina**.  
2. **Preprocesar NSL-KDD** (encoding + normalización + split por cliente).  
3. Entrenar un **Transformer Encoder pequeño** en cada cliente (C++ con kernels en CUDA).  
4. Implementar **servidor de agregación** con **FedAvg()**.  
5. **Comparar** contra un entrenamiento **centralizado** (mismo modelo, datos fusionados).  
6. Reportar **precisión/F1** y **costos de comunicación** (tamaño de pesos por ronda).

---

## 🗂️ Estructura sugerida del repositorio
├── CMakeLists.txt
├── include/
│ ├── transformer.hpp
│ ├── fedavg.hpp
│ └── dataloader.hpp
├── src/
│ ├── transformer.cu
│ ├── client.cpp
│ ├── server.cpp
│ └── centralized.cpp
├── data/
│ ├── raw/ # KDDTrain+.txt, KDDTest+.txt
│ ├── processed/ # *.csv / *.bin normalizados
│ └── splits/ # client1.csv, client2.csv, client3.csv
├── scripts/
│ ├── preprocess_nslkdd.py
│ ├── split_clients.py
│ └── run_federated.sh
├── configs/
│ ├── model.yaml # d_model, n_heads, n_layers, ff_dim, dropout…
│ ├── train_fed.yaml # rounds, local_epochs, batch_size, lr…
│ └── train_central.yaml
└── README.md




---

## ⚙️ Requisitos
- **CMake ≥ 3.24**
- **CUDA ≥ 12.x**, toolkit y driver compatibles
- **GCC/Clang** con soporte C++17
- **Python 3.9+** (solo para *scripts* de preprocesamiento)
- (Opcional) **vcpkg/conan** para gestionar dependencias C++ si se usan

---
