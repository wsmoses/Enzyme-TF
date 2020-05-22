#include <cstddef>
#include <stdio.h>
#include <cmath>
#include <inttypes.h>
#include <sys/time.h>

static inline float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

extern "C" {
    void __enzyme_autodiff(...);

    static inline float sigmoid(float f) {
        return 1 / (1 + expf(-f));
    }

    void f(float* __restrict z, uint64_t z_len, float* __restrict zb, uint64_t zb_len, float* __restrict c, uint64_t c_len, float* __restrict f, uint64_t f_len, float* __restrict i, uint64_t i_len, float* __restrict g, uint64_t g_len, float* __restrict out) {
        
            //printf("starting idx=%d z_len=%d \n", idx, z_len);
        for(int jdx=0; jdx<z_len; jdx++) {
            for(int idx=0; idx<z_len; idx++) {    
                float ii2 = i[idx+z_len*jdx];
                float gg2 = g[idx+z_len*jdx];
                float ff2 = f[idx+z_len*jdx];
                ii2 = sigmoid(ii2);
                gg2 = tanhf(gg2);
                ff2 = sigmoid(ff2);
                float cc2 = c[idx+z_len*jdx];
                
                if (z[idx] == 1.) {
                    out[idx+z_len*jdx] = ii2 * gg2;
                } else if (zb[idx] == 0.) {
                    out[idx+z_len*jdx] = cc2;
                } else {
                    out[idx+z_len*jdx] = ff2 * cc2 + ii2 * gg2;
                }
            }
        }

    }


    #ifdef TF_ENZYME
    int diffe_dupnoneed;
    int diffe_dup;

    __attribute__((noinline))
    void diffef(float* __restrict z, float* __restrict dz, uint64_t z_len, float* __restrict zb, float* __restrict dzb, uint64_t zb_len, float* __restrict c, float* __restrict dc, uint64_t c_len, float* __restrict af, float* __restrict df, uint64_t f_len, float* __restrict i, float* __restrict di, uint64_t i_len, float* __restrict g, float* __restrict dg, uint64_t g_len, float* __restrict dout) {

        //struct timeval start, end;
        //gettimeofday(&start, NULL);
        __enzyme_autodiff(f,
            diffe_dupnoneed, z, dz,
            z_len,
            diffe_dupnoneed, zb, dzb,
            zb_len,
            diffe_dupnoneed, c, dc,
            c_len,
            diffe_dupnoneed, af, df,
            f_len,
            diffe_dupnoneed, i, di,
            i_len,
            diffe_dupnoneed, g, dg,
            g_len,
            diffe_dupnoneed, (float*)0, dout);
       // printf("dg[0]=%f dout[0]=%f\n", dg[0], dout[0]);

        //gettimeofday(&end, NULL);
        //printf("enzyme forwardreverse %f\n", tdiff(&start, &end));
    }
   
    /*
    int main() {
        float g[2] = { 0.39729846f, 0.6661664f };
        float dout[2] = {1.0f, 1.0f };
        float dg[2] = { 0.f, 0.f };
        diffef(0,0,0,  0,0,0,   0,0,0,  0,0,0,   0,0,0,  g, dg, 2,   dout);
        printf("dg[0]=%f dg[1]=%f, dout[0]=%f, dout[1]=%f\n", dg[0], dg[1], dout[0], dout[1]);
    }
    */

    #endif
}
