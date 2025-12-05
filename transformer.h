#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

struct TokenEmbedding {
    int d_model;
    int dims[4];   // [T0, T1, T2, T3]

    vector<vector<vector<double>>> W;

    TokenEmbedding(int d0, int d1, int d2, int d3, int d_model_=64)
        : d_model(d_model_)
    {
        dims[0] = d0;
        dims[1] = d1;
        dims[2] = d2;
        dims[3] = d3;

        W.resize(4);

        std::mt19937 gen(1234);
        std::normal_distribution<double> dist(0.0, 0.02);

        for (int t = 0; t < 4; t++) {
            W[t] = vector<vector<double>>(d_model, vector<double>(dims[t]));
            for (int i = 0; i < d_model; i++)
                for (int j = 0; j < dims[t]; j++)
                    W[t][i][j] = dist(gen);
        }
    }

    vector<vector<double>> forward(const array<vector<double>,4>& tok) {
        vector<vector<double>> out(4, vector<double>(d_model, 0.0));
        for (int t=0; t<4; t++)
            for (int i=0; i<d_model; i++)
                for (int j=0; j<dims[t]; j++)
                    out[t][i] += W[t][i][j] * tok[t][j];
        return out;
    }

    void backward(const array<vector<double>,4>& tok,
                  const vector<vector<double>>& grad,
                  double lr)
    {
        for (int t=0; t<4; t++)
            for (int i=0; i<d_model; i++)
                for (int j=0; j<dims[t]; j++)
                    W[t][i][j] -= lr * grad[t][i] * tok[t][j];
    }


};


inline double randf(){ return ((double)rand()/RAND_MAX - 0.5)*0.02; }

vector<double> softmax_cpu(const vector<double>& v){
    double mx = -1e9;
    for(double x:v) mx = max(mx,x);

    vector<double> e(v.size());
    double sum = 0;
    for(int i=0;i<v.size();i++){
        e[i] = exp(v[i]-mx);
        sum += e[i];
    }
    for(int i=0;i<v.size();i++)
        e[i] /= sum;
    return e;
}


struct LayerNorm{
    int d;
    vector<double> m_cache, inv_cache;

    LayerNorm(int d_):d(d_){}

    vector<double> forward(const vector<double>& x){
        double m=0;
        for(double v:x) m+=v;
        m/=d;

        double var=0;
        for(double v:x) var+=(v-m)*(v-m);
        var/=d;

        double inv = 1.0/sqrt(var+1e-6);

        m_cache.assign(d,m);
        inv_cache.assign(d,inv);

        vector<double> y(d);
        for(int i=0;i<d;i++) y[i]=(x[i]-m)*inv;
        return y;
    }

    vector<double> backward(const vector<double>& dy,
                            const vector<double>& x){
        vector<double> dx(d);
        double m=m_cache[0];
        double inv=inv_cache[0];

        double s1=0, s2=0;
        for(int i=0;i<d;i++){
            s1+=dy[i];
            s2+=dy[i]*(x[i]-m);
        }
        for(int i=0;i<d;i++){
            dx[i] = inv*( dy[i] - s1/d - (x[i]-m)*s2/d );
        }
        return dx;
    }
};


struct Linear{
    int in,out;
    vector<vector<double>> W;
    vector<double> b;

    vector<vector<double>> gW;
    vector<double> gb;

    Linear(int in_, int out_): in(in_),out(out_){
        W.resize(out, vector<double>(in));
        b.resize(out);
        gW.resize(out, vector<double>(in,0));
        gb.resize(out,0);

        for(auto &row:W)
            for(double &v:row) v = randf();
        for(double &v:b) v=randf();
    }

    vector<double> forward(const vector<double>& x){
        vector<double> y(out);
        for(int i=0;i<out;i++){
            double s = b[i];
            for(int j=0;j<in;j++) s += W[i][j]*x[j];
            y[i]=s;
        }
        return y;
    }

    vector<double> backward(const vector<double>& x,
                            const vector<double>& dy){
        vector<double> dx(in,0.0);

        for(int i=0;i<out;i++){
            gb[i]+=dy[i];
            for(int j=0;j<in;j++){
                gW[i][j]+=dy[i]*x[j];
                dx[j]+=W[i][j]*dy[i];
            }
        }
        return dx;
    }

    void step(double lr){
        for(int i=0;i<out;i++){
            b[i]-=lr*gb[i];
            gb[i]=0;
            for(int j=0;j<in;j++){
                W[i][j]-=lr*gW[i][j];
                gW[i][j]=0;
            }
        }
    }
};


struct FeedForward {
    Linear L1,L2;
    vector<double> h;

    FeedForward(int d): L1(d,2*d),L2(2*d,d){}

    vector<double> forward(const vector<double>& x){
        h=L1.forward(x);
        for(double &v:h) if(v<0) v=0;
        return L2.forward(h);
    }

    vector<double> backward(const vector<double>& x,
                            const vector<double>& dy){
        auto d2 = L2.backward(h,dy);

        for(int i=0;i<d2.size();i++)
            if(h[i]<=0) d2[i]=0;

        return L1.backward(x,d2);
    }

    void step(double lr){
        L1.step(lr);
        L2.step(lr);
    }
};


struct MultiHeadAttention{
    int d,h,dk;
    Linear Wq,Wk,Wv,Wo;

    vector<vector<double>> Q,K,V;
    vector<vector<double>> scores, att;

    MultiHeadAttention(int d_, int h_)
        : d(d_), h(h_), dk(d_/h_),
          Wq(d_,d_),Wk(d_,d_),Wv(d_,d_),Wo(d_,d_) {}

    vector<vector<double>> forward(const vector<vector<double>>& X){
        int T=X.size();

        Q.resize(T);
        K.resize(T);
        V.resize(T);

        for(int t=0;t<T;t++){
            Q[t]=Wq.forward(X[t]);
            K[t]=Wk.forward(X[t]);
            V[t]=Wv.forward(X[t]);
        }

        double scale = 1.0/sqrt((double)dk);

        scores.assign(T, vector<double>(T,0));
        for(int i=0;i<T;i++)
            for(int j=0;j<T;j++){
                double s=0;
                for(int k=0;k<d;k++)
                    s+=Q[i][k]*K[j][k];
                scores[i][j]=s*scale;
            }

        att.resize(T);
        for(int i=0;i<T;i++)
            att[i]=softmax_cpu(scores[i]);

        vector<vector<double>> O(T, vector<double>(d,0));
        for(int i=0;i<T;i++)
            for(int j=0;j<T;j++)
                for(int k=0;k<d;k++)
                    O[i][k]+=att[i][j]*V[j][k];

        for(int i=0;i<T;i++)
            O[i]=Wo.forward(O[i]);

        return O;
    }

    vector<vector<double>> backward(const vector<vector<double>>& X,
                                    const vector<vector<double>>& dO){
        int T=X.size();
        vector<vector<double>> dX(T, vector<double>(d,0));

        // backward of Wo
        vector<vector<double>> dO2(T);
        for(int i=0;i<T;i++)
            dO2[i]=Wo.backward(att[i], dO[i]);

        vector<vector<double>> dV(T, vector<double>(d,0));
        vector<vector<double>> dAtt(T, vector<double>(T,0));

        for(int i=0;i<T;i++)
            for(int j=0;j<T;j++)
                for(int k=0;k<d;k++){
                    dV[j][k]+=dO2[i][k]*att[i][j];
                    dAtt[i][j]+=dO2[i][k]*V[j][k];
                }

        vector<vector<double>> dS(T, vector<double>(T,0));
        for(int i=0;i<T;i++){
            double s=0;
            for(int j=0;j<T;j++)
                s+=dAtt[i][j]*att[i][j];
            for(int j=0;j<T;j++)
                dS[i][j]=att[i][j]*( dAtt[i][j]-s );
        }

        double scale = 1.0/sqrt((double)dk);

        vector<vector<double>> dQ(T, vector<double>(d,0));
        vector<vector<double>> dK(T, vector<double>(d,0));

        for(int i=0;i<T;i++)
            for(int j=0;j<T;j++)
                for(int k=0;k<d;k++){
                    dQ[i][k]+= dS[i][j]*K[j][k]*scale;
                    dK[j][k]+= dS[i][j]*Q[i][k]*scale;
                }

        for(int i=0;i<T;i++){
            auto dq = Wq.backward(X[i], dQ[i]);
            auto dk = Wk.backward(X[i], dK[i]);
            auto dv = Wv.backward(X[i], dV[i]);

            for(int k=0;k<d;k++)
                dX[i][k]+= dq[k]+dk[k]+dv[k];
        }

        return dX;
    }

    void step(double lr){
        Wq.step(lr);
        Wk.step(lr);
        Wv.step(lr);
        Wo.step(lr);
    }
};


struct TransformerLayer{
    MultiHeadAttention attn;
    FeedForward ff;
    LayerNorm ln1,ln2;

    vector<vector<double>> Xc,A,ln1c;

    TransformerLayer(int d, int h)
        : attn(d,h), ff(d), ln1(d), ln2(d) {}

    vector<vector<double>> forward(const vector<vector<double>>& X){
        Xc = X;
        A = attn.forward(X);

        int T=X.size();
        vector<vector<double>> R1(T, vector<double>(A[0].size()));

        for(int i=0;i<T;i++)
            for(int j=0;j<A[0].size();j++)
                R1[i][j]=X[i][j]+A[i][j];

        ln1c.resize(T);
        for(int i=0;i<T;i++)
            ln1c[i]=ln1.forward(R1[i]);

        vector<vector<double>> FF(T);
        for(int i=0;i<T;i++)
            FF[i]=ff.forward(ln1c[i]);

        vector<vector<double>> R2(T, vector<double>(FF[0].size()));

        for(int i=0;i<T;i++)
            for(int j=0;j<FF[0].size();j++)
                R2[i][j]=ln1c[i][j]+FF[i][j];

        vector<vector<double>> out(T);
        for(int i=0;i<T;i++)
            out[i]=ln2.forward(R2[i]);

        return out;
    }

    vector<vector<double>> backward(const vector<vector<double>>& dOut){
        int T = dOut.size();

        // 1) Backward de LN2
        vector<vector<double>> dR2(T);
        for(int i=0;i<T;i++)
            dR2[i] = ln2.backward(dOut[i], ln1c[i]);

        // 2) Dividimos gradientes
        vector<vector<double>> dFF(T), dLN1(T);
        for (int i=0;i<T;i++) {
            dFF[i]  = dR2[i];
            dLN1[i] = dR2[i];
        }

        // 3) Backward del FeedForward
        vector<vector<double>> dLN1_cache(T);
        for (int i=0;i<T;i++)
            dLN1_cache[i] = ff.backward(ln1c[i], dFF[i]);

        // 4) Residual + LN1 backward
        vector<vector<double>> dR1(T);
        for(int i=0;i<T;i++)
            dR1[i].resize(dLN1_cache[i].size());

        for(int i=0;i<T;i++)
            for(int k=0;k<dLN1_cache[i].size();k++)
                dR1[i][k] = dLN1_cache[i][k] + dLN1[i][k];

        vector<vector<double>> dA(T);
        for (int i=0;i<T;i++)
            dA[i] = ln1.backward(dR1[i], Xc[i]);

        // 5) Backward de la atención
        auto dX = attn.backward(Xc, dA);

        // 6) Residual final
        for (int i=0;i<T;i++)
            for (int k=0;k<dX[i].size();k++)
                dX[i][k] += dR1[i][k];

        return dX;
    }

    void step(double lr){
        attn.step(lr);
        ff.step(lr);
    }

};


struct Transformer{
    int seq_len, d_model, heads, layers;
    vector<TransformerLayer> L;

    Transformer(int sl, int dm=64, int h=4, int ly=1)
        : seq_len(sl), d_model(dm), heads(h), layers(ly)
    {
        for(int i=0;i<layers;i++)
            L.emplace_back(dm,heads);
    }

    vector<vector<double>> forward(const vector<vector<double>>& X){
        auto H=X;
        for(auto &layer:L)
            H = layer.forward(H);
        return H;
    }

    vector<vector<double>> backward(const vector<vector<double>>& dH){
        auto g=dH;
        for(int i=layers-1;i>=0;i--)
            g = L[i].backward(g);
        return g;
    }

    void step(double lr){
        for(auto &layer:L)
            layer.step(lr);
    } 
};
