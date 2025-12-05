#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <unordered_set>
#include "transformer.h"
using namespace std;

#define BLOCK 256

__global__ void softmax_kernel(const double* X,
    double* Y, int batch, int classes){

    int r = blockIdx.x;
    if (r >= batch) return;

    X += r * classes;
    Y += r * classes;

    double mx = -1e300;
    for (int c = 0; c < classes; c++)
        mx = max(mx, X[c]);

    double sum = 0.0;
    for (int c = 0; c < classes; c++)
        sum += exp(X[c] - mx);

    for (int c = 0; c < classes; c++)
        Y[c] = exp(X[c] - mx) / sum;
}

__global__ void cross_entropy_kernel(const double* pred,
    const int* labels, double* loss_out, int batch, int classes){

    __shared__ double cache[BLOCK];
    int tid = threadIdx.x;

    double local_sum = 0.0;

    for (int i = tid; i < batch; i += BLOCK) {
        int lab = labels[i];
        double p = pred[i * classes + lab];
        p = max(p, 1e-12);
        local_sum += -log(p);
    }

    cache[tid] = local_sum;
    __syncthreads();

    if (tid == 0) {
        double total = 0.0;
        for (int i = 0; i < BLOCK; i++)
            total += cache[i];
        loss_out[0] = total / batch;
    }
}

__global__ void compute_final_dZ_kernel(const double* pred,
    const int* labels, double* dZ, int batch, int classes){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * classes;
    if (idx >= total) return;

    int r = idx / classes;
    int c = idx % classes;

    double val = pred[idx];
    if (labels[r] == c)
        val -= 1.0;

    dZ[idx] = val / batch;
}


__global__ void fc_forward_kernel(const double* X,
                                  const double* W,
                                  const double* b,
                                  double* Z,
                                  int batch, int in, int out)
{
    int r = blockIdx.x;
    int c = threadIdx.x;

    if (r >= batch || c >= out) return;

    double sum = b[c];
    for (int k = 0; k < in; k++)
        sum += X[r * in + k] * W[c * in + k];

    Z[r * out + c] = sum;
}

__global__ void relu_forward_kernel(double* A, const double* Z, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        A[idx] = (Z[idx] > 0.0 ? Z[idx] : 0.0);
}

__global__ void relu_backward_kernel(double* dZ, const double* Z, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        dZ[idx] *= (Z[idx] > 0 ? 1.0 : 0.0);
}

__global__ void compute_dW_kernel(double* dW,
                                  const double* dZ,
                                  const double* X,
                                  int batch, int in, int out)
{
    int i = blockIdx.x;  // output neuron
    int j = threadIdx.x; // input neuron

    if (i >= out || j >= in) return;

    double sum = 0.0;

    for (int r = 0; r < batch; r++)
        sum += dZ[r * out + i] * X[r * in + j];

    dW[i * in + j] = sum / batch;
}

__global__ void compute_dB_kernel(double* dB,
                                  const double* dZ,
                                  int batch, int out)
{
    int i = threadIdx.x;
    if (i >= out) return;

    double sum = 0.0;

    for (int r = 0; r < batch; r++)
        sum += dZ[r * out + i];

    dB[i] = sum / batch;
}

__global__ void compute_dX_kernel(double* dX,
                                  const double* dZ,
                                  const double* W,
                                  int batch, int in, int out)
{
    int r = blockIdx.x;
    int j = threadIdx.x;

    if (r >= batch || j >= in) return;

    double sum = 0.0;

    for (int i = 0; i < out; i++)
        sum += dZ[r * out + i] * W[i * in + j];

    dX[r * in + j] = sum;
}

__global__ void adam_update_kernel(double* W, double* m, double* v,
                                   const double* g, double lr,
                                   double b1, double b2,
                                   double b1t, double b2t,
                                   int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    double gi = g[i];
    double mi = m[i];
    double vi = v[i];

    mi = b1 * mi + (1 - b1) * gi;
    vi = b2 * vi + (1 - b2) * gi * gi;

    double m_hat = mi / (1 - b1t);
    double v_hat = vi / (1 - b2t);

    W[i] -= lr * m_hat / (sqrt(v_hat) + 1e-8);

    m[i] = mi;
    v[i] = vi;
}


inline void adam_update(double* W, double* mW, double* vW,
                        double* b, double* mB, double* vB,
                        const double* dW, const double* dB,
                        int wsize, int out,
                        double lr, double b1, double b2,
                        double b1t, double b2t){
    int threads = 256;

    int gw = (wsize + threads - 1) / threads;
    adam_update_kernel<<<gw, threads>>>(
        W, mW, vW, dW, lr, b1, b2, b1t, b2t, wsize);

    int gb = (out + threads - 1) / threads;
    adam_update_kernel<<<gb, threads>>>(
        b, mB, vB, dB, lr, b1, b2, b1t, b2t, out);
}


struct DenseLayerCUDA {

    int in, out;

    double *W, *b;
    double *mW, *vW, *mB, *vB;

    double *Z, *A;
    double *dZ, *dW, *dB, *dX;

    DenseLayerCUDA(int _in, int _out) : in(_in), out(_out)
    {
        cudaMalloc(&W, sizeof(double)*in*out);
        cudaMalloc(&b, sizeof(double)*out);

        cudaMalloc(&mW, sizeof(double)*in*out);
        cudaMalloc(&vW, sizeof(double)*in*out);
        cudaMalloc(&mB, sizeof(double)*out);
        cudaMalloc(&vB, sizeof(double)*out);

        cudaMalloc(&dW, sizeof(double)*in*out);
        cudaMalloc(&dB, sizeof(double)*out);

        cudaMemset(mW, 0, sizeof(double)*in*out);
        cudaMemset(vW, 0, sizeof(double)*in*out);
        cudaMemset(mB, 0, sizeof(double)*out);
        cudaMemset(vB, 0, sizeof(double)*out);
    }

    void alloc_batch(int batch)
    {
        cudaMalloc(&Z, sizeof(double)*batch*out);
        cudaMalloc(&A, sizeof(double)*batch*out);
        cudaMalloc(&dZ, sizeof(double)*batch*out);
        cudaMalloc(&dX, sizeof(double)*batch*in);
    }

    void init_params()
    {
        std::vector<double> hW(in*out);
        std::vector<double> hB(out, 0.0);

        std::mt19937 gen(123);
        std::normal_distribution<double> dist(0, sqrt(2.0 / in));

        for (double &x : hW) x = dist(gen);

        cudaMemcpy(W, hW.data(), sizeof(double)*hW.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(b, hB.data(), sizeof(double)*out, cudaMemcpyHostToDevice);
    }

    void forward(const double* X, int batch)
    {
        fc_forward_kernel<<<batch, out>>>(X, W, b, Z, batch, in, out);

        int total = batch * out;
        relu_forward_kernel<<<(total+255)/256, 256>>>(A, Z, total);
    }

    void backward(const double* X, const double* dA, int batch)
    {
        int total = batch * out;

        cudaMemcpy(dZ, dA, sizeof(double)*total, cudaMemcpyDeviceToDevice);
        relu_backward_kernel<<<(total+255)/256, 256>>>(dZ, Z, total);

        compute_dW_kernel<<<out, in>>>(dW, dZ, X, batch, in, out);
        compute_dB_kernel<<<1, out>>>(dB, dZ, batch, out);

        compute_dX_kernel<<<batch, in>>>(dX, dZ, W, batch, in, out);
    }

};

struct MLP_CUDA {

    int input_dim;
    int hidden_dim;
    int num_classes;
    int batch;

    DenseLayerCUDA L1, L2;

    double* d_input;
    int* d_labels;
    double* d_softmax;
    double* d_loss;
    double* dZ_final;

    double lr = 0.001, beta1 = 0.9, beta2 = 0.999, t = 1.0;

    MLP_CUDA(int in_dim, int hidden, int out_dim, int batch_size)
    : input_dim(in_dim), hidden_dim(hidden), num_classes(out_dim), batch(batch_size),
      L1(in_dim, hidden), L2(hidden, out_dim)
    {
        cudaMalloc(&d_input,   sizeof(double)*batch*input_dim);
        cudaMalloc(&d_labels,  sizeof(int)*batch);
        cudaMalloc(&d_softmax, sizeof(double)*batch*out_dim);
        cudaMalloc(&d_loss,    sizeof(double));
        cudaMalloc(&dZ_final,  sizeof(double)*batch*out_dim);

        L1.alloc_batch(batch);
        L2.alloc_batch(batch);

        L1.init_params();
        L2.init_params();
    }



    void forward_batch(const vector<vector<double>>& Xb,
                       const vector<int>& Yb)
    {
        cudaMemcpy(d_input, Xb[0].data(),
                   sizeof(double)*batch*input_dim, cudaMemcpyHostToDevice);

        cudaMemcpy(d_labels, Yb.data(),
                   sizeof(int)*batch, cudaMemcpyHostToDevice);

        L1.forward(d_input, batch);
        L2.forward(L1.A, batch);

        softmax_kernel<<<batch,1>>>(L2.A, d_softmax, batch, num_classes);
        cross_entropy_kernel<<<1,BLOCK>>>(d_softmax, d_labels, d_loss, batch, num_classes);
    }

    double get_loss() {
        double h;
        cudaMemcpy(&h, d_loss, sizeof(double), cudaMemcpyDeviceToHost);
        return h;
    }

    /* ---------- Backward + Adam ---------- */
    void backward_update()
    {
        compute_final_dZ_kernel<<< (batch*num_classes+255)/256, 256 >>>(
            d_softmax, d_labels, dZ_final, batch, num_classes);

        L2.backward(L1.A, dZ_final, batch);
        L1.backward(d_input, L2.dX, batch);

        double b1t = pow(beta1, t);
        double b2t = pow(beta2, t);

        adam_update(L1.W, L1.mW, L1.vW,
                    L1.b, L1.mB, L1.vB,
                    L1.dW, L1.dB,
                    L1.in*L1.out, L1.out,
                    lr, beta1, beta2, b1t, b2t);

        adam_update(L2.W, L2.mW, L2.vW,
                    L2.b, L2.mB, L2.vB,
                    L2.dW, L2.dB,
                    L2.in*L2.out, L2.out,
                    lr, beta1, beta2, b1t, b2t);

        t += 1.0;
    }

    int predict(const vector<double>& x){
        // Copiar input CLS a GPU (batch=1)
        cudaMemcpy(d_input, x.data(),
                sizeof(double)*input_dim, cudaMemcpyHostToDevice);

        // Forward capa 1
        L1.forward(d_input, 1);

        // Forward capa 2
        L2.forward(L1.A, 1);

        // Softmax
        softmax_kernel<<<1,1>>>(L2.A, d_softmax, 1, num_classes);

        vector<double> out(num_classes);
        cudaMemcpy(out.data(), d_softmax,
                sizeof(double)*num_classes, cudaMemcpyDeviceToHost);

        return int(max_element(out.begin(), out.end()) - out.begin());
    }   
    
    void load_weights(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Error cargando modelo: " << filename << "\n";
            return;
        }

        // ---- CAPA 1 ----
        int W1_size = L1.in * L1.out;
        int b1_size = L1.out;

        std::vector<double> hW1(W1_size);
        std::vector<double> hb1(b1_size);

        in.read((char*)hW1.data(), sizeof(double)*W1_size);
        in.read((char*)hb1.data(), sizeof(double)*b1_size);

        cudaMemcpy(L1.W, hW1.data(), sizeof(double)*W1_size, cudaMemcpyHostToDevice);
        cudaMemcpy(L1.b, hb1.data(), sizeof(double)*b1_size, cudaMemcpyHostToDevice);

        // ---- CAPA 2 ----
        int W2_size = L2.in * L2.out;
        int b2_size = L2.out;

        std::vector<double> hW2(W2_size);
        std::vector<double> hb2(b2_size);

        in.read((char*)hW2.data(), sizeof(double)*W2_size);
        in.read((char*)hb2.data(), sizeof(double)*b2_size);

        cudaMemcpy(L2.W, hW2.data(), sizeof(double)*W2_size, cudaMemcpyHostToDevice);
        cudaMemcpy(L2.b, hb2.data(), sizeof(double)*b2_size, cudaMemcpyHostToDevice);

        in.close();
        std::cout << "Modelo cargado desde " << filename << "\n";
    }
    void zero_weights() {

        int W1_size = L1.in * L1.out;
        int b1_size = L1.out;

        int W2_size = L2.in * L2.out;
        int b2_size = L2.out;

        // Poner en cero todos los pesos y biases
        cudaMemset(L1.W, 0, sizeof(double) * W1_size);
        cudaMemset(L1.b, 0, sizeof(double) * b1_size);

        cudaMemset(L2.W, 0, sizeof(double) * W2_size);
        cudaMemset(L2.b, 0, sizeof(double) * b2_size);

        // También ponemos los momentos del Adam en cero
        cudaMemset(L1.mW, 0, sizeof(double) * W1_size);
        cudaMemset(L1.vW, 0, sizeof(double) * W1_size);
        cudaMemset(L1.mB, 0, sizeof(double) * b1_size);
        cudaMemset(L1.vB, 0, sizeof(double) * b1_size);

        cudaMemset(L2.mW, 0, sizeof(double) * W2_size);
        cudaMemset(L2.vW, 0, sizeof(double) * W2_size);
        cudaMemset(L2.mB, 0, sizeof(double) * b2_size);
        cudaMemset(L2.vB, 0, sizeof(double) * b2_size);
    }
    
};

struct NSLKDD {

    vector<vector<double>> num_raw;
    vector<string> proto_raw, service_raw, flag_raw;
    vector<string> label_raw;

    vector<string> proto_vocab, service_vocab, flag_vocab;
    unordered_map<string,int> proto_map, service_map, flag_map;
    unordered_map<string,int> label_map;

    vector<array<vector<double>,4>> X_tokens;
    vector<int> y;

    vector<string> split(const string& s) {
        vector<string> out;
        string t;
        stringstream ss(s);
        while (getline(ss, t, ',')) out.push_back(t);
        return out;
    }

    vector<string> unique_vec(const vector<string>& v) {
        unordered_set<string> s(v.begin(), v.end());
        return vector<string>(s.begin(), s.end());
    }

    void load_file(const string& file) {
        ifstream f(file);
        if (!f.is_open()) throw runtime_error("Error abriendo " + file);

        string line;
        while (getline(f, line)) {

            if (line.size() < 10) continue;
            auto col = split(line);
            if (col.size() != 43) continue;

            proto_raw.push_back(col[1]);
            service_raw.push_back(col[2]);
            flag_raw.push_back(col[3]);

            vector<double> nums;
            nums.push_back(stod(col[0])); // duration
            for (int i = 4; i <= 40; i++)
                nums.push_back(stod(col[i]));
            num_raw.push_back(nums);

            label_raw.push_back(col[41]);
        }
    }

    void build_maps() {
        proto_vocab   = unique_vec(proto_raw);
        service_vocab = unique_vec(service_raw);
        flag_vocab    = unique_vec(flag_raw);

        for (int i = 0; i < proto_vocab.size(); i++)  proto_map[proto_vocab[i]] = i;
        for (int i = 0; i < service_vocab.size(); i++) service_map[service_vocab[i]] = i;
        for (int i = 0; i < flag_vocab.size(); i++)    flag_map[flag_vocab[i]] = i;
    }

    void process() {
        build_maps();

        int N = num_raw.size();
        int numf = num_raw[0].size();

        vector<double> mn(numf, 1e18), mx(numf, -1e18);
        for (auto &row : num_raw)
            for (int i = 0; i < numf; i++) {
                mn[i] = min(mn[i], row[i]);
                mx[i] = max(mx[i], row[i]);
            }

        X_tokens.resize(N);

        int P = proto_vocab.size();
        int S = service_vocab.size();
        int F = flag_vocab.size();

        for (int i = 0; i < N; i++) {

            // ---------- TOKEN 0 : ONE-HOT ----------
            vector<double> t0(P + S + F, 0.0);
            t0[ proto_map[proto_raw[i]] ] = 1.0;
            t0[ P + service_map[service_raw[i]] ] = 1.0;
            t0[ P + S + flag_map[flag_raw[i]] ] = 1.0;

            auto norm = [&](int idx){
                double d = mx[idx] - mn[idx];
                if (d < 1e-9) d = 1;
                return (num_raw[i][idx] - mn[idx]) / d;
            };

            // ---------- TOKEN 1 ----------
            vector<double> t1 = {
                norm(4-1),  norm(5-1),  norm(22-1),
                norm(23-1), norm(11-1), norm(32-1),
                norm(33-1)
            };

            // ---------- TOKEN 2 ----------
            vector<double> t2;
            int content_list[5] = {12,13,14,15,16};
            for (int k=0;k<5;k++) t2.push_back(norm(content_list[k]-1));

            // ---------- TOKEN 3 ----------
            vector<double> t3 = {
                norm(0), norm(6-1), norm(7-1),
                norm(8-1), norm(21-1), norm(28-1)
            };

            X_tokens[i] = { t0,t1,t2,t3 };

            if (!label_map.count(label_raw[i]))
                label_map[label_raw[i]] = label_map.size();

            y.push_back(label_map[label_raw[i]]);
        }
    }

    int seq_len() const { return 4; }
    int num_classes() const { return label_map.size(); }
};


void accumulate_mlp(MLP_CUDA &acc, const MLP_CUDA &m) {

    int W1_size = acc.L1.in * acc.L1.out;
    int b1_size = acc.L1.out;

    int W2_size = acc.L2.in * acc.L2.out;
    int b2_size = acc.L2.out;

    vector<double> aW1(W1_size), aB1(b1_size);
    vector<double> aW2(W2_size), aB2(b2_size);

    vector<double> mW1(W1_size), mB1(b1_size);
    vector<double> mW2(W2_size), mB2(b2_size);

    // descargo acumulador
    cudaMemcpy(aW1.data(), acc.L1.W, sizeof(double)*W1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(aB1.data(), acc.L1.b, sizeof(double)*b1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(aW2.data(), acc.L2.W, sizeof(double)*W2_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(aB2.data(), acc.L2.b, sizeof(double)*b2_size, cudaMemcpyDeviceToHost);

    // descargo modelo nuevo
    cudaMemcpy(mW1.data(), m.L1.W, sizeof(double)*W1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mB1.data(), m.L1.b, sizeof(double)*b1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mW2.data(), m.L2.W, sizeof(double)*W2_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mB2.data(), m.L2.b, sizeof(double)*b2_size, cudaMemcpyDeviceToHost);

    // sumo
    for (int i = 0; i < W1_size; i++) aW1[i] += mW1[i];
    for (int i = 0; i < b1_size; i++) aB1[i] += mB1[i];
    for (int i = 0; i < W2_size; i++) aW2[i] += mW2[i];
    for (int i = 0; i < b2_size; i++) aB2[i] += mB2[i];

    // regreso a GPU
    cudaMemcpy(acc.L1.W, aW1.data(), sizeof(double)*W1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc.L1.b, aB1.data(), sizeof(double)*b1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc.L2.W, aW2.data(), sizeof(double)*W2_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc.L2.b, aB2.data(), sizeof(double)*b2_size, cudaMemcpyHostToDevice);
}


// divide el acumulador entre K
void divide_mlp(MLP_CUDA &acc, int K) {

    int W1_size = acc.L1.in * acc.L1.out;
    int b1_size = acc.L1.out;

    int W2_size = acc.L2.in * acc.L2.out;
    int b2_size = acc.L2.out;

    vector<double> W1(W1_size), B1(b1_size);
    vector<double> W2(W2_size), B2(b2_size);

    cudaMemcpy(W1.data(), acc.L1.W, sizeof(double)*W1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B1.data(), acc.L1.b, sizeof(double)*b1_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(W2.data(), acc.L2.W, sizeof(double)*W2_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B2.data(), acc.L2.b, sizeof(double)*b2_size, cudaMemcpyDeviceToHost);

    for (double &x : W1) x /= K;
    for (double &x : B1) x /= K;
    for (double &x : W2) x /= K;
    for (double &x : B2) x /= K;

    cudaMemcpy(acc.L1.W, W1.data(), sizeof(double)*W1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc.L1.b, B1.data(), sizeof(double)*b1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc.L2.W, W2.data(), sizeof(double)*W2_size, cudaMemcpyHostToDevice);
    cudaMemcpy(acc.L2.b, B2.data(), sizeof(double)*b2_size, cudaMemcpyHostToDevice);
}


int main() {
    NSLKDD test;
    test.load_file("NSL_KDD-master/KDDTest+.txt");
    test.process();

    int T0 = test.proto_vocab.size()
           + test.service_vocab.size()
           + test.flag_vocab.size();

    int T1_dim = 7, T2_dim = 5, T3_dim = 6;
    int d_model = 64;
    int classes = test.num_classes();

    TokenEmbedding emb(T0, T1_dim, T2_dim, T3_dim, d_model);
    Transformer tr(4, 64, 4, 1);

    MLP_CUDA mlp_avg(64, 128, classes, 1);

    mlp_avg.zero_weights();

    vector<string> files = {
        "mlp_model_1.bin",
        "mlp_model_2.bin",
        "mlp_model_3.bin"
    };

    int K = files.size();
    cout << "Cargando " << K << " modelos...\n";

    for (string f : files) {

        MLP_CUDA tmp(64, 128, classes, 1);
        tmp.load_weights(f);

        accumulate_mlp(mlp_avg, tmp);
    }

    divide_mlp(mlp_avg, K);
    cout << "Promedio completado.\n";


    cout << "Evaluando ensemble promedio...\n";

    int correct = 0;
    int N = test.X_tokens.size();

    for (int i = 0; i < N; i++) {

        auto tok = emb.forward(test.X_tokens[i]);
        auto H   = tr.forward(tok);

        vector<double> cls = H[0];

        int pred = mlp_avg.predict(cls);
        if (pred == test.y[i]) correct++;
    }

    double acc = 100.0 * correct / N;

    cout << "Accuracy final (ensemble) = " << acc << "%\n";
    return 0;
}