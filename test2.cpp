#include <cstddef>
#include <stdio.h>
#include <cmath>

extern "C" {
    void __enzyme_autodiff(...);

    static inline float sigmoid(float f) {
        return 1 / (1 + exp(-f));
    }

    void f(float* __restrict z, size_t z_len, float* __restrict zb, size_t zb_len, float* __restrict c, size_t c_len, float* __restrict f, size_t f_len, float* __restrict i, size_t i_len, float* __restrict g, size_t g_len, float* out __restrict) {

        for(int idx=0; idx<z_len; idx++) {
            i[idx] = sigmoid(i[idx]);
            g[idx] = tanh(g[idx]);
            f[idx] = sigmoid(f[idx]);

            float*

            if (z[idx] == 1.) {
                out[idx] = i[idx] * g[idx];
            } else if (zb[idx] == 0.) {
                out[idx] = c[idx];
            } else {
                out[idx] = f[idx] * c[idx] + i[idx] * g[idx];
            }
        }
    }
    #ifdef TF_ENZYME
    int diffe_dupnoneed;
    int diffe_dup;

    void diffef(float* __restrict z, float* __restrict dz, size_t z_len, float* __restrict zb, float* __restrict dzb, size_t zb_len, float* __restrict c, float* __restrict dc, size_t c_len, float* __restrict f, float* __restrict df, size_t f_len, float* __restrict i, float* __restrict di, size_t i_len, float* __restrict g, float* __restrict dg, size_t g_len, float* dout __restrict) {
        __enzyme_autodiff(f, diffe_dup, z, dz, z_len, diffe_dup, zb, dzb, zb_len, diffe_dup, c, dc, c_len, diffe_dup, f, df, f_len, diffe_dup, i, di, i_len, diffe_dup, g, dg, g_len, diffe_dupnoneed, (float*)0, dout);
    }
    #endif
}
