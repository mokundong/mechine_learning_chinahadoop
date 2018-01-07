# !/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib as mpl


def calc_accuracy(y, y_true):
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return sess.run(accuracy, feed_dict={x: x_test, y_true: y_test})


if __name__ == '__main__':
    path = 'C:\\test_mnist_tf'
    data = input_data.read_data_sets(train_dir=path, one_hot=True)
    print(data)
    x_train, y_train = data.train.images, data.train.labels
    x_test, y_test = data.test.images, data.test.labels
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    n = 28*28
    x = tf.placeholder(tf.float32, [None, n], name='x_input')
    w = tf.Variable(tf.zeros([n, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b
    y_true = tf.placeholder(tf.float32, [None, 10], name='y_true_output')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    acc = []
    for i in range(4000):
        xs, ys = data.train.next_batch(60)
        sess.run(train, feed_dict={x: xs, y_true: ys})
        if i % 100 == 0:
            acc.append(calc_accuracy(y, y_true))
    print('准确率：', acc[-1])
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(acc[1:], 'r-', acc[1:], 'go', lw=2, markersize=4)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('使用TensorFlow完成Softmax回归', fontsize=15)
    plt.grid(b=True, ls=':')
    plt.show()
