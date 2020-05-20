import tensorflow as tf
from tf_enzyme import *

out = enzyme([[1,2], [3,4.]], filename='test.cpp', function='f')
print(out)


x = tf.constant([1,2,3,4.])
# y = x ^ 2
with tf.GradientTape() as t:
  t.watch(x)
  y = enzyme(x, filename="test.cpp", function="f")
# dy = 2x
dy_dx = t.gradient(y, x)
print(dy_dx.numpy())
