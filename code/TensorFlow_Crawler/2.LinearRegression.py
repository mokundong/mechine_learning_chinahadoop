# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    N = 10
    x_data = np.float32(np.random.rand(N, 100))
    print(x_data.shape)
    theta = np.linspace(0.1, 1, N)
    print(theta)
    y_data = np.dot(theta, x_data) + 100

    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, N], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
    train = optimizer.minimize(loss=loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    losses = []
    for step in range(0, 1001):
        sess.run(train)
        if step % 20 == 0:
            losses.append(sess.run(loss))
            print(step, sess.run(W), sess.run(b), sess.run(loss))
    print('最终结果：', sess.run(W), sess.run(b), sess.run(loss))
    sess.close()
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,8), facecolor='w')
plt.plot(losses, 'r-', losses, 'g*', lw=2, markersize=10)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.title('使用TensorFlow完成线性回归', fontsize=15)
plt.grid(b=True, ls=':')
plt.show()
