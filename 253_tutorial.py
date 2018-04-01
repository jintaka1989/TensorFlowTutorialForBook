# - * - coding: utf-8 - * -
# 二次関数の近似を行いたい
# y_sample = 2 * x_sample * x_sample + 0.5
# シグモイド関数により、１次関数が曲がることを確認する

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

coefficient=2
intercept=0.5

start = time.time()

# data set
x_sample = np.random.rand(100,1).astype("float32")
x_sample = x_sample
y_sample = coefficient * x_sample * x_sample + intercept

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])

w_input = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b_input = tf.Variable(tf.zeros([1]))

w = tf.Variable(tf.random_uniform([4,4], -1.0, 1.0))
b = tf.Variable(tf.zeros([4,4]))

# 1st layer
h_linear1 = tf.sigmoid(tf.add(w_input * x_data, b_input))

# 2nd layer
h_linear2 = tf.reduce_sum(tf.matmul(h_linear1 * w, b))

# output
y = h_linear2

# Add summary ops to collect data
w_hist = tf.histogram_summary("weights", w)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

loss = tf.reduce_mean(tf.square(y_data - y))
loss_summary = tf.scalar_summary("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)
sess.run(init)

for step in xrange(1001):
    for i in xrange(100):
        if step % 100 == 0:
            result = sess.run([merged, loss],feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
            summary_str = result[0]
            acc = result[1]
            writer.add_summary(summary_str, step)
        else:
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
    if step % 100 == 0:
        print step
        print sess.run(w)
        print sess.run(b)

xx = np.arange(-1, 1, 0.1)
yy = coefficient  *  xx  *  xx + intercept

plt.plot(xx, yy)

x=[]
r = []

for i in xrange(20):
    prot_x = (i-10) * 0.1
    x.append(prot_x)
    r.append(sess.run(y, feed_dict={x_data:[prot_x]}))
    print(r[i])

plt.scatter(x, r)

plt.show()

sess.close()

timer = time.time() - start
print(("time:{0}".format(timer)) + "[sec]")
