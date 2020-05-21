CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

ZERO_OUT_SRCS = $(wildcard tf_enzyme/cc/kernels/*.cc) $(wildcard tf_enzyme/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = tf_enzyme/python/ops/_enzyme_ops.so

# zero_out op for CPU
zero_out_op: $(ZERO_OUT_TARGET_LIB)

$(ZERO_OUT_TARGET_LIB): $(ZERO_OUT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS} -lffi

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts
clean:
	rm -f $(ZERO_OUT_TARGET_LIB) 
