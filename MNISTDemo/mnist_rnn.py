# -*- coding: utf-8 -*-
# Author: qwchen
# Date: 2018-05-01
# 使用RNN来做MNIST的分类
#
# 使用RNN来做MNIST图片分类，需要将每张图片看成是一个像素序列。
# 因为MNIST图片的大小是28*28像素，所以我们把每一个图片看成一行行的序列。
# 这样，一张图片对应的输入层节点是28个，时间步也是28。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def model(X, n_inputs, n_time_steps, n_hidden_units, n_classes):
    """循环神经网络模型"""
    # 隐含层
    w_hidden = tf.Variable(tf.truncated_normal([n_inputs, n_hidden_units], stddev=0.1))
    b_hidden = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
    # 把输入的X转换成shape（batch_size * n_time_steps, n_inputs）
    X_in = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X_in, w_hidden) + b_hidden  # shape(batch_size * n_time_steps, n_hidden_units)
    X_in = tf.reshape(X_in, [-1, n_time_steps, n_hidden_units])  # shape(batch_size, n_time_steps, n_hidden_units)
    # 使用基本的LSTM循环单元
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 初始化为零值，lstm单元有两部分组成：（c_state, h_state）.
    # 注意下面指定了batch_size为固定大小，所以训练数据和测试数据每次只能为一个batch_size进入网络
    init_state = lstm_cell.zero_state(batch_size=FLAGS.batch_size, dtype=tf.float32)
    # dynamic_rnn 接收张量（batch，steps，inputs）或者（steps，batch，inputs）作为X_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    # 输出层
    w_output = tf.Variable(tf.truncated_normal([n_hidden_units, n_classes]))
    b_output = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    results = tf.matmul(final_state[1], w_output) + b_output
    return results


def main(_):
    n_inputs = 28  # 输入层的神经元个数
    n_time_steps = 28  # 时间步
    n_hidden_units = 128  # 隐含层的神经元个数
    n_classes = 10  # 输出层的神经元个数

    # 1 加载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    X = tf.placeholder(tf.float32, [None, n_time_steps, n_inputs])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    # 2 定义网络结构
    Y_ = model(X, n_inputs, n_time_steps, n_hidden_units, n_classes)

    # 3 定义损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    # 4 模型预测
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 5 训练和评估模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(FLAGS.max_steps):
            # 将训练数据分为一个个batch送进网络进行训练
            len_train = len(mnist.train.images)
            train_step = 0
            while train_step * FLAGS.batch_size < len_train:
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                batch_xs = batch_xs.reshape([-1, n_time_steps, n_inputs])
                sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})
                train_step += 1
            # 将测试数据分为一个个batch送进网络进行训练
            len_test = len(mnist.test.images)
            test_step = 0
            print_accuracy = 0.0
            while test_step * FLAGS.batch_size < len_test:
                batch_xs, batch_ys = mnist.test.next_batch(FLAGS.batch_size)
                batch_xs = batch_xs.reshape([-1, n_time_steps, n_inputs])
                print_accuracy += sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys})
                test_step += 1
            print('第{0}次评估的准确率：{1}'.format(i, print_accuracy / test_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/mnist', help='Directory for storing input data')
    parser.add_argument('--max_steps', type=int, default=100, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Initial batch size')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
