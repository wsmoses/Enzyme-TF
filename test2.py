import tensorflow as tf
from tf_enzyme import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dims', nargs='?', default=2048, type=int, help='dimension of differentiable variables')
parser.add_argument('enzyme', nargs='?', default=False, type=bool, help='dimension of differentiable variables')

args = parser.parse_args()

# The below code is a snippet adapted from the
# canonical HMLSTM implementation, specifically
# the code here:
#
# https://github.com/n-s-f/hierarchical-rnn/blob/f23a71cf1eff0bb8d0cbf9810f05c2eae8e10723/hmlstm/hmlstm_cell.py#L78-L83

print(args.dims)
print(args.enzyme)

def hmlstm_update_c(z, zb, c, f, i, g):
    if args.enzyme:
        return enzyme(z, zb, c, f, i, g, filename="test2.cpp", function="f")
    i = tf.sigmoid(i)
    g = tf.tanh(g)
    f = tf.sigmoid(f)

    return tf.where(
        tf.equal(z, tf.constant(1., dtype=tf.float32)),
        tf.multiply(i, g),
        tf.where(
            tf.equal(zb, tf.constant(0., dtype=tf.float32)),
            tf.identity(c),
            tf.add(tf.multiply(f, c), tf.multiply(i, g))
        )
    )


class Benchmark:
    def __init__(self, dims):
        # set up control variables
        tf.random.set_seed(0)
        self.z = tf.Variable(tf.cast(tf.less(tf.random.uniform([dims]), 0.5), dtype=tf.float32))
        self.zb = tf.Variable(tf.cast(tf.less(tf.random.uniform([dims]), 0.5), dtype=tf.float32))

        # set up differentiable variables
        self.c = tf.Variable(tf.random.uniform([dims, dims]))
        self.f = tf.Variable(tf.random.uniform([dims, dims]))
        self.i = tf.Variable(tf.random.uniform([dims, dims]))
        self.g = tf.Variable(tf.random.uniform([dims, dims]))
        print(self.g.shape)

    def warmup(self):
        #sess.run(tf.compat.v1.global_variables_initializer())
        #self.run(sess)
        with tf.GradientTape() as t:
          t.watch(self.c)
          t.watch(self.f)
          t.watch(self.i)
          t.watch(self.g)
          new_c = hmlstm_update_c(self.z, self.zb, self.c, self.f, self.i, self.g)
        # dy = 2x
        #print("new_c", new_c)
        dy_dx = t.gradient(new_c, [self.c, self.f, self.i, self.g])
        print([x.numpy() if x is not None else x for x in dy_dx ])
        pass

    def cmp(self):
        args.enzyme = False
        old_c = hmlstm_update_c(self.z, self.zb, self.c, self.f, self.i, self.g)
        print("old_c", old_c)
        args.enzyme = True
        new_c = hmlstm_update_c(self.z, self.zb, self.c, self.f, self.i, self.g)
        print(self.i)
        print("new_c", new_c)
        print("diff", old_c - new_c)

    def run(self):
        #sess.run([self.c_grad, self.f_grad, self.i_grad, self.g_grad], **kwargs)
                # y = x ^ 2
        with tf.GradientTape() as t:
          t.watch(self.c)
          t.watch(self.f)
          t.watch(self.i)
          t.watch(self.g)
          new_c = hmlstm_update_c(self.z, self.zb, self.c, self.f, self.i, self.g)
        # dy = 2x
        dy_dx = t.gradient(new_c, [self.c, self.f, self.i, self.g])
        pass



import timeit


def pretty_print_time(s):
    if s < 1e-6:
        unit = "n"
        factor = 1e9
    elif s < 1e-3:
        unit = "u"
        factor = 1e6
    elif s < 1:
        unit = "m"
        factor = 1e3
    else:
        unit = ""
        factor = 1
    print("{0:.2f} {1}s".format(s*factor, unit))

# turn on the XLA JIT compiler
#config = tf.ConfigProto()
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

if __name__ == '__main__':
    #tf.compat.v1.disable_eager_execution()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    b = Benchmark(args.dims)
    #b.cmp()
    b.warmup()
    t = timeit.Timer("b.run()", globals=globals())
    its, total_time = t.autorange()
    pretty_print_time(total_time / its)
