# -*- coding: utf-8 -*-
# Author: qwchen
# Date: 2018-04-30
# 使用DNN来做MNIST的分类

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # 1 加载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # 2 构建网络
    # 2.1 定义模型
    x = tf.placeholder(tf.float32, [None, 784])
    # 将参数初始化的0
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 使用随机初始化
    # W = tf.Variable(tf.random_normal([784, 10], stddev=1, seed=1))
    # b = tf.Variable(tf.random_normal([10], stddev=1, seed=1))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.int16, [None, 10])

    # 2.2 定义损失函数和优化器
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 3 训练模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for _ in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # 4 模型评估
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/mnist', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

