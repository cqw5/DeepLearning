# -*- coding: utf-8 -*-
# Author: qwchen
# Date: 2018-05-01
# 使用CNN来做MNIST的分类

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def init_weight(shape):
    """初始化权重"""
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))  # 正太分布


def init_bias(shape):
    """初始化偏置"""
    return tf.Variable(tf.constant(0.1, shape=shape))


def model(X, p_keep_conv, p_keep_hidden):
    """构建模型"""
    # 初始化参数
    # 第一个卷积层
    w1 = init_weight([3, 3, 1, 32])  # 卷积核大小为3*3，输入维度为1，输出维度为32
    b1 = init_bias([32])  # 每一维输出对应一个bias
    # 第二个卷积层
    w2 = init_weight([3, 3, 32, 64])  # 卷积核大小为3*3，输入维度为32，输出维度为64
    b2 = init_bias([64])  # 每一维输出对应一个bias
    # 第三个卷积层
    w3 = init_weight([3, 3, 64, 128])  # 卷积核大小为3*3，输入维度为64， 输出维度为128
    b3 = init_bias([128])  # 每一维输出对应一个bias
    # 全连接层
    w4 = init_weight([4 * 4 * 128, 512])  # 4 * 4 是经过第三个卷积层在pool之后，图片的长*宽
    b4 = init_bias([512])
    # 输出层
    w5 = init_weight([512, 10])  # 10个类别
    b5 = init_bias([10])

    # 网络结构
    # 第一组卷积层及池化层，最后dropout一些神经元
    # conv2d函数
    # @param strides:为不同维度上的步长，一个4维的数组，第1维和第4维一定是1，因为卷积层的步长只对矩阵的长和宽有效；
    #        第2维和第3维表示在长和宽上的步长
    # @param padding:填充方法，SAME表示添加全0填充，VALID表示不添加
    conv1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')  # shape=(None, 28, 28, 32)
    bias1 = tf.nn.bias_add(conv1, b1)  # shape=(None, 28, 28, 32)
    actived1 = tf.nn.relu(bias1)  # shape=(None, 28, 28, 32)
    # max_pool函数
    # @param ksize:为pool过滤器的尺寸，一个4维的数组，第1维和第4维一定是1，意味着pool过滤器不能跨不同样本或者节点矩阵的深度
    # @param strides:类似于conv2d函数的该参数
    # @param padding:类似于conv2d函数的该参数
    pool1 = tf.nn.max_pool(actived1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')  # shape=(None, 14, 14, 32)
    drop1 = tf.nn.dropout(pool1, p_keep_conv)  # shape=(None, 14, 14, 32)

    # 第二组卷积层及池化层，最后dropout一些神经元
    conv2 = tf.nn.conv2d(drop1, w2, strides=[1, 1, 1, 1], padding='SAME')  # shape=(None, 14, 14, 64)
    bias2 = tf.nn.bias_add(conv2, b2)  # shape=(None, 14, 14, 64)
    actived2 = tf.nn.relu(bias2)  # shape=(None, 14, 14, 64)
    pool2 = tf.nn.max_pool(actived2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # shape=(None, 7, 7, 64)
    drop2 = tf.nn.dropout(pool2, p_keep_conv)  # shape=(None, 7, 7, 64)

    # 第三组卷积陈及池化层，遭遇过dropout一些神经元
    conv3 = tf.nn.conv2d(drop2, w3, strides=[1, 1, 1, 1], padding='SAME')  # shape=(None, 7, 7, 128)
    bias3 = tf.nn.bias_add(conv3, b3)  # shape=(None, 7, 7, 128)
    actived3 = tf.nn.relu(bias3)
    pool3 = tf.nn.max_pool(actived3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')  # shape=(None, 4, 4, 128)
    drop3 = tf.nn.dropout(pool3, p_keep_conv)  # shape=(None, 4, 4, 128)

    # 全连接层, pool的输出要连接全连接层时，需要先做一个reshape
    layer3 = tf.reshape(drop3, shape=[-1, 2048])  # shape=(None, 2048)
    layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)  # shape=(None, 512)
    layer4 = tf.nn.dropout(layer4, p_keep_hidden)  # shape=(None, 512)

    # 输出层
    output = tf.matmul(layer4, w5) + b5  # shape=(None, 10)

    return output


def main(_):
    # 1 加载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    # -1表示不考虑输入图片的数量，28*28是图片的长宽像素，1是通道数量（黑白图片）
    trainX = trainX.reshape(-1, 28, 28, 1)
    testX = testX.reshape(-1, 28, 28, 1)
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])
    p_keep_conv = tf.placeholder(tf.float32)
    p_keep_hidden = tf.placeholder(tf.float32)

    # 2 构建网络
    Y_ = model(X, p_keep_conv, p_keep_hidden)

    # 3 定义损失函数及优化器
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    # 4 训练和评估模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(FLAGS.max_steps):
            # 训练
            training_batch = zip(range(0, len(trainX), FLAGS.batch_size),
                                 range(FLAGS.batch_size, len(trainX) + 1, FLAGS.batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trainX[start: end], Y: trainY[start: end],
                                              p_keep_conv: FLAGS.dropout_conv,
                                              p_keep_hidden: FLAGS.dropout_hidden})
            # 测试
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print_accuracy = sess.run(accuracy, feed_dict={X: testX, Y: testY, p_keep_conv: 1, p_keep_hidden: 1})
            print('第{0}轮迭代的准确率: {1}'.format(i, print_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/mnist', help='Directory for storing input data')
    parser.add_argument('--max_steps', type=int, default=100, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Initial batch size')
    parser.add_argument('--dropout_conv', type=float, default=0.8, help='Keep probability for training conv dropout.')
    parser.add_argument('--dropout_hidden', type=float, default=0.5, help='Keep probability for training hidden dropout.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
