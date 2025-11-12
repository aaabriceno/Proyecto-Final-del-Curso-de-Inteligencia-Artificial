#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "common.cuh" // Un archivo para utilidades como el kernel de LayerNorm

// --- KERNELS BÁSICOS (Simplificados) ---

// Kernel para multiplicar dos matrices (A * B^T) -> Usado en Self-Attention
__global__ void matmul_transpose_kernel(const double* A, const double* B, double* C, int N, int M, int K);

// Kernel para aplicar Softmax a lo largo de las filas
__global__ void softmax_kernel(double* data, int N, int M);

// Kernel para la red Feed-Forward (puedes adaptar tu 'forward_kernel' del MLP)
__global__ void feed_forward_kernel(const double* W1, const double* b1, const double* W2, const double* b2, const double* input, double* output, int input_size, int ff_size, int output_size);


// --- CLASES DEL MODELO ---

class MultiHeadAttention {
public:
    int d_model, num_heads;
    double *d_W_q, *d_W_k, *d_W_v, *d_W_o; // Pesos para Q, K, V y salida
    
    MultiHeadAttention(int model_dim, int heads);
    ~MultiHeadAttention();

    // El forward tomará el input y devolverá el output de la atención
    void forward(const double* input, double* output);
};

class TransformerEncoderLayer {
public:
    int d_model, num_heads, d_ff;
    MultiHeadAttention mha;
    // Añadir aquí LayerNorm, FFN y conexiones residuales.
    
    TransformerEncoderLayer(int model_dim, int heads, int ff_dim);

    void forward(const double* input, double* output);
};

class TransformerModel {
public:
    std::vector<TransformerEncoderLayer> layers;
    int num_layers;
    // Otros componentes como la capa de embedding de entrada y la capa final de clasificación.

    TransformerModel(int n_layers, int d_model, int num_heads, int d_ff, int input_vocab_size, int output_classes);
    
    // Función para obtener todos los pesos del modelo en un solo vector (serializar)
    std::vector<double> get_weights();

    // Función para cargar pesos desde un vector (deserializar)
    void set_weights(const std::vector<double>& weights);
};

#endif // TRANSFORMER_CUH
