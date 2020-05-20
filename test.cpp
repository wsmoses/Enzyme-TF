#include <cstddef>
#include <stdio.h>
extern "C" {
    void __enzyme_autodiff(...);

    void f(float* inp, size_t n, float* out) {
        printf("calling f\n");
        for(unsigned i=0; i<n; i++) {
            out[i] = inp[i] * 2;
        }
    }
    #ifdef TF_ENZYME 
    int diffe_dupnoneed;
    int diffe_dup;

    void diffef(float* inp, float* dinp, size_t n, float* dout) {
        __enzyme_autodiff(f, diffe_dup, inp, dinp, n, diffe_dupnoneed, (float*)0, dout);
    }
    #endif
}
