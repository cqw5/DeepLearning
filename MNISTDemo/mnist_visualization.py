# -*- coding: utf-8 -*-
# Author: qwchen
# Date: 2018-04-30
# 使用DNN来做MNIST的分类的可视化例子

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
    # 1 加载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    sess = tf.InteractiveSession()

    # 2 构建网络
    # 2.1 输入数据的Placeholder
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    # (可视化)为输入的数据添加摘要，用于Images版面
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    # 2.2 定义模型
    # 权重和偏置不能初始化为0
    def weight_variable(shape):
        """使用正太分布来初始化权重变量"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """将偏置遍历设置为固定值"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """（可视化）对一个张量添加多个摘要描述。用于Scalars中绘制每一层的均值、标准差、最大值和最小值"""
        with tf.name_scope('summaries'):
            # 均值
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            # 标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            # 最大值
            tf.summary.scalar('max', tf.reduce_max(var))
            # 最小值
            tf.summary.scalar('min', tf.reduce_min(var))
            # 直方图
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """定义网络的层，并给每一层加一个name_scope"""
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                # 初始化权重
                weights = weight_variable([input_dim, output_dim])
                # （可视化）为权重添加摘要，并在Scalar版面绘制摘要
                variable_summaries(weights)
            with tf.name_scope('biases'):
                # 初始化偏置
                biases = bias_variable([output_dim])
                # （可视化）为偏置添加摘要，并在Scalar版面绘制摘要
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                # 计算W * x + b
                preactivate = tf.matmul(input_tensor, weights) + biases
                # （可视化）在distributions版面为 W * x + b 绘制激活前直方图
                tf.summary.histogram('pre_activations', preactivate)
            # 激活函数，激活relu(W * x + b)
            activations = act(preactivate, name='activation')
            # （可视化）在distributions版面为 W * x + b 绘制激活后直方图
            tf.summary.histogram('activations', activations)
            return activations

    # 2.2.1 定义模型的第一层
    hidden1 = nn_layer(x, 784, 500, 'layer1')

    # 对第一层输出结果进行dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        # （可视化）
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    # 2.2.1 定义模型的第二层
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    # 2.3 定义损失函数
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    # （可视化）在Scalars版面绘制loss
    tf.summary.scalar('cross_entropy', cross_entropy)

    # 2.4 定义优化器
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # 4 模型评估
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # （可视化）在Scalars版面绘制准确率
    tf.summary.scalar('accuracy', accuracy)

    # 合并所有摘要，并将其写到logs中
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    # 3 模型训练
    tf.global_variables_initializer().run()

    def feed_dict(train):
        """构造Tensorflow feed_dict
        train为true时，输出训练数据，minibatch的size为100
        train为false时，输出测试数据"""
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = FLAGS.dropout  # dropout只有训练的时候需要
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0  # 测试时不进行dropout
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):
        # 每10步进行一次test
        if i % 10 == 0:
            # merged是用来记录可视化所需要的信息
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            # （可视化）写入测试信息
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            # 每100步记录一次执行状态
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                # （可视化）写入训练信息
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                # （可视化）写入训练信息
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='../data/mnist', help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='../logs/mnist_with_summaries', help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

