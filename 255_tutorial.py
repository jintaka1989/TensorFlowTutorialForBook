# - * - coding: utf-8 - * -

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

intercept=-0.25
coefficient1=0.5
coefficient2=5
coefficient3=5

start = time.time()

# data set
data_num = 200
tensor_num = 50

# graph_para
graph_range = (-2, 2)
x_plot_sub = 0.01
x_range = int((graph_range[1] - graph_range[0])/x_plot_sub)

x_sample = np.random.rand(data_num,1).astype("float32")
x_sample = x_sample * 2.0 - 1.0
y_sample = coefficient3 * x_sample * x_sample * x_sample + coefficient2 *  x_sample * x_sample + coefficient1 * x_sample + intercept
# y_sample = np.sin(5 * x_sample) + intercept + np.random.rand(data_num,1).astype("float32")/2

x_data = tf.placeholder(tf.float32,[1])
y_data = tf.placeholder(tf.float32,[1])

w_input = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b_input = tf.Variable(tf.zeros([1]))

w = tf.constant(tensor_num,shape=[tensor_num,1],dtype=tf.float32)
b = tf.constant(2.0 * (np.arange(tensor_num).astype(float)-(tensor_num/2.0)),shape=[tensor_num,1],dtype=tf.float32)

w_output = tf.Variable(tf.random_uniform([2,tensor_num], -1.0, 1.0))
b_output = tf.Variable(tf.zeros([1]))

# 1st layer
# layer1 = x_data
layer1 = tf.add(w_input * x_data, b_input)

# 2nd layer
layer2 = tf.sigmoid(tf.add(layer1 * w, b))
layer2_2 = tf.reduce_sum(tf.matmul(w_output, layer2))

# output
y = tf.add(layer2_2,b_output)

# # Add summary ops to collect data
# w_hist = tf.histogram_summary("weights", w)
# b_hist = tf.histogram_summary("biases", b)
# y_hist = tf.histogram_summary("y", y)

loss = tf.reduce_mean(tf.square(y_data - y))
# loss_summary = tf.scalar_summary("loss", loss)

# optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
# merged = tf.merge_all_summaries()
# writer = tf.train.SummaryWriter("/tmp/tensorflow_log", sess.graph_def)
sess.run(init)

for step in xrange(1001):
    for i in xrange(data_num):
        if step % 100 == 0:
            sess.run(loss, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
            result = sess.run(loss,feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
            # summary_str = result[0]
            acc = result
            # writer.add_summary(summary_str, step)
        else:
            sess.run(train, feed_dict={x_data:x_sample[i], y_data:y_sample[i]})
    if step % 100 == 0:
        print step
        print sess.run(w)
        print sess.run(b)
        # print acc
        plt.scatter(x_sample, y_sample)

        x=[]
        r = []

        for i in xrange(x_range):
            prot_x = (i-x_range/2.0) * x_plot_sub
            x.append(prot_x)
            r.append(sess.run(y, feed_dict={x_data:[prot_x]}))
            print(r[i])

        plt.plot(x, r, color='orangered')
        plt.pause(2)
        plt.close()

timer = time.time() - start
print(("time:{0}".format(timer)) + "[sec]")

plt.scatter(x_sample, y_sample)

x=[]
r = []

for i in xrange(x_range):
    prot_x = (i-x_range/2.0) * x_plot_sub
    x.append(prot_x)
    r.append(sess.run(y, feed_dict={x_data:[prot_x]}))
    print(r[i])

plt.plot(x, r, color='orangered')

plt.show()

sess.close()
