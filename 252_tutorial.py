# -*- coding: utf-8 -*-
# ４次関数の近似を行う
# ただし、５次以上の関数についても近似できるように
# 一般化してコードを書く
# あくまで「n次関数を近似する」としてn次関数専用のモデルを作成している
# ４次関数なら４次関数、５次関数なら５次関数を
# NFD = 4
# の値を変更することで近似する

import tensorflow as tf
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

start = time.time()
AC=100
# number of the function s dimention
NFD = 4
#1+number of the function s dimention
NN=NFD+1
# number of W
WN=NN

a = np.arange(1,WN+1,1)
a = 0.1 * a

x_data = np.random.rand(AC,1,1).astype("float32")
y_data = np.zeros((AC,1,1)).astype("float32")
npow = 1

for i in xrange(WN):
    y_data += a[i] * npow
    npow *= x_data

W = tf.Variable(tf.random_uniform([WN], -1.0, 1.0))
y = 0
npow = 1


for i in xrange(WN):
    y += W[i] * npow
    npow *= x_data

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(1000*pow(2,NN+2)):
    sess.run(train)
    if step % (10*pow(2,NN)) == 0:
        print "training...", sess.run(W)

plot_x = []
plot_y = []

for i in xrange(100):
    result = 0
    npow = 1
    xx = i-50
    for j in xrange(WN):
        result += W[j] * npow
        npow *= xx
    plot_x.append(xx)
    plot_y.append(sess.run(result))
    # print i, sess.run(result)

x_correct = []
y_correct = []

for i in xrange(100):
    result = 0
    npow = 1
    xx = i-50
    for j in xrange(NN):
        result += a[j] * npow
        npow *= xx
    x_correct.append(xx)
    y_correct.append(result)
    # print i, result

plt.plot(x_correct, y_correct)
plt.scatter(plot_x, plot_y)

plt.show()

sess.close()

timer = time.time() - start

print ("time:{0}".format(timer)) + "[sec]"
