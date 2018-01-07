# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def intro1():
    x1 = tf.Variable(2)
    x2 = tf.Variable(3)
    a = tf.Variable(10)
    x = tf.add(x1, x2)
    z = tf.add(x, a)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    result = sess.run([x, z])
    print(result)
    sess.close()


def intro2():
    x1 = tf.placeholder(dtype=np.float32)
    x2 = tf.placeholder(dtype=np.float32)
    x = tf.add(x1, x2)
    with tf.Session() as sess:
        result = sess.run(x, feed_dict={x1: 1, x2: 2})
        print(result)


if __name__ == '__main__':
    intro1()
    intro2()
