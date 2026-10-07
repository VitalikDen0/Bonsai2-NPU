#pragma once
void bonsai_rmsnorm(const float* x, const float* w, int n, float* out);
void bonsai_rope(float* q, float* k, int heads, int head_dim, int pos);
void bonsai_swiglu(const float* gate, const float* up, int n, float* out);
float bonsai_softmax(float* x, int n);
int bonsai_sample(const float* logits, int vocab, float temp,
                  int top_k, float top_p, unsigned* seed);
void bonsai_fwht1024(const float* x, const float* signs, int n, int inverse, float* out);

