import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.set_random_seed(23450)


# 获取数据
train_data = np.load("data/train_image_data.npz")
test_data = np.load("data/test_image_data.npz")
# 获取训练集
train_image_list = train_data['train_image_list']
train_image_label = train_data['train_image_label']
# 获取测试集
test_image_list = test_data['test_image_list']
test_image_label = test_data['test_image_label']

# 定义train_next_bach 函数
g_b = 0

def train_next_batch(size):
    global g_b
    x = train_image_list[g_b:g_b+size]
    y = train_image_label[g_b:g_b+size]
    g_b += size
    return x, y


# 定义test_next_bach 函数
g_b = 0
def test_next_batch(size):
    global g_b
    x = test_image_list[g_b:g_b+size]
    y = test_image_label[g_b:g_b+size]
    g_b += size
    return x, y


# 定义权重函数
def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

# 定义偏置函数
def init_bit(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# 定义卷积操作
def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

#定义池化操作
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 用来计算权重和偏置的均值和方差
def variable_summaries(var):
    # 统计参数的均值,并记录
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
    # 计算参数的标准差
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
    # 统计参数的最大最小值
    tf.summary.scalar("max", tf.reduce_max(var))
    tf.summary.scalar("min", tf.reduce_min(var))
    # 用直方图统计参数的分布
    tf.summary.histogram("histogram", var)


if __name__ == "__main__":
    with tf.name_scope("input"):
        # 定义占位符
        x = tf.placeholder(tf.float32, [None, 100, 100, 3], name="input")
        y = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope("conv1"):
        with tf.name_scope("weithts"):
            w1 = init_weight([5, 5, 3, 32])
            variable_summaries(w1)

        with tf.name_scope("bias"):
            b1 = init_bit([32])
            variable_summaries(b1)

        with tf.name_scope("relu_output"):
            conv_1 = conv2d(x, w1) + b1
            h_conv1 = tf.nn.relu(conv_1)
            # 100*100*32
        with tf.name_scope("max_pool"):
            h_pool1 = max_pool(h_conv1)
            # 50*50*32
    with tf.name_scope("conv2"):
        with tf.name_scope("weights"):
            w2 = init_weight([5, 5, 32, 64])
            variable_summaries(w2)
        with tf.name_scope("bias"):
            b2 = init_bit([64])
            variable_summaries(b2)
        with tf.name_scope("relu_output"):
            conv_2 = conv2d(h_pool1, w2) + b2
            h_conv2 = tf.nn.relu(conv_2)
            # 50*50*64
        with tf.name_scope("max_pool"):
            h_pool2 = max_pool(h_conv2)
            # 25*25*64
    with tf.name_scope("conv3"):
        with tf.name_scope("weights"):
            w3 = init_weight([5, 5, 64, 128])
            variable_summaries(w3)
        with tf.name_scope("bias"):
            b3 = init_bit([128])
            variable_summaries(b3)
        with tf.name_scope("relu_output"):
            conv_3 = conv2d(h_pool2, w3) + b3
            h_conv3 = tf.nn.relu(conv_3)
            # 25*25*128
        with tf.name_scope("max_pool"):
            h_pool3 = max_pool(h_conv3)
            # 13*13*128
    with tf.name_scope("conv4"):
        with tf.name_scope("weights"):
            w4 = init_weight([5, 5, 128, 256])
            variable_summaries(w4)
        with tf.name_scope("bias"):
            b4 = init_bit([256])
            variable_summaries(b4)
        with tf.name_scope("relu_output"):
            conv_4 = conv2d(h_pool3, w4) + b4
            h_conv4 = tf.nn.relu(conv_4)
            # 13*13*256
        with tf.name_scope("max_pool"):
            h_pool4 = max_pool(h_conv4)
            # 7*7*256

    with tf.name_scope("reverse_conv1") as scope:
        # 第四层
        reverse_weight7 = init_weight([5, 5, 128, 256])
        reverse_conv7 = tf.nn.conv2d_transpose(conv_4, reverse_weight7, [50, 13, 13, 128], strides=[1, 1, 1, 1],
                                               padding="SAME")
        reverse_weight8 = init_weight([5, 5, 64, 128])
        reverse_conv8 = tf.nn.conv2d_transpose(reverse_conv7, reverse_weight8, [50, 25, 25, 64], strides=[1, 2, 2, 1],
                                               padding="SAME")
        reverse_weight9 = init_weight([5, 5, 32, 64])
        reverse_conv9 = tf.nn.conv2d_transpose(reverse_conv8, reverse_weight9, [50, 50, 50, 32], strides=[1, 2, 2, 1],
                                               padding="SAME")
        reverse_weight10 = init_weight([5, 5, 1, 32])
        reverse_conv10 = tf.nn.conv2d_transpose(reverse_conv9, reverse_weight10, [50, 100, 100, 1], strides=[1, 2, 2, 1],
                                               padding="SAME")
        # 第三层
        reverse_weight1 = init_weight([5, 5, 64, 128])
        reverse_conv1 = tf.nn.conv2d_transpose(conv_3, reverse_weight1, [50, 25, 25, 64], strides=[1, 1, 1, 1],
                                               padding="SAME")
        reverse_weight2 = init_weight([5, 5, 32, 64])
        reverse_conv2 = tf.nn.conv2d_transpose(reverse_conv1, reverse_weight2, [50, 50, 50, 32], strides=[1, 2, 2, 1],
                                               padding="SAME")
        reverse_weight3 = init_weight([5, 5, 1, 32])
        reverse_conv3 = tf.nn.conv2d_transpose(reverse_conv2, reverse_weight3, [50, 100, 100, 1], strides=[1, 2, 2, 1],
                                               padding="SAME")

        # 第二层
        reverse_weight4 = init_weight([5,5,32,64])
        reverse_conv4 = tf.nn.conv2d_transpose(conv_2, reverse_weight4, [50,50,50,32], strides=[1, 1, 1, 1],
                                               padding="SAME")
        reverse_weight5 = init_weight([5, 5, 1, 32])
        reverse_conv5 = tf.nn.conv2d_transpose(reverse_conv4, reverse_weight5,[50, 100, 100,1],strides=[1, 2, 2, 1],
                                               padding="SAME")

        # 第一层
        reverse_weight6 = init_weight([5, 5, 1, 32])
        reverse_conv6 = tf.nn.conv2d_transpose(conv_1, reverse_weight6,[50, 100, 100,1],strides=[1, 1, 1, 1],
                                               padding="SAME")
    tf.summary.image("reverse_conv4", reverse_conv10, 10)
    tf.summary.image("reverse_conv3", reverse_conv3, 10)
    tf.summary.image("reverse_conv2", reverse_conv5, 10)
    tf.summary.image("reverse_conv1", reverse_conv6, 10)

    with tf.name_scope("fc1"):
        with tf.name_scope("weights"):
            w5 = init_weight([7 * 7 * 256, 1024])
            variable_summaries(w5)
        with tf.name_scope("bias"):
            b5 = init_bit([1024])
            variable_summaries(b5)
        with tf.name_scope("relu_output"):
            h_pool3_flat = tf.reshape(h_pool4, [-1, 7*7*256])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w5) + b5)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder("float", name="keep_prob")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope("fc2"):
        with tf.name_scope("weights"):
            w6 = init_weight([1024, 512])
            variable_summaries(w6)
        with tf.name_scope("bias"):
            b6 = init_bit([512])
            variable_summaries(b6)
        with tf.name_scope("relu_output"):
            y1 = tf.nn.relu(tf.matmul(h_fc1_drop, w6) + b6)
    with tf.name_scope("fc3"):
        with tf.name_scope("weights"):
            w7 = init_weight([512, 256])
            variable_summaries(w7)
        with tf.name_scope("bias"):
            b7 = init_bit([256])
            variable_summaries(b7)
        with tf.name_scope("relu_output"):
            y2 = tf.nn.relu(tf.matmul(y1, w7) + b7)
    with tf.name_scope("fc4"):
        with tf.name_scope("weights"):
            w8 = init_weight([256, 10])
            variable_summaries(w8)
        with tf.name_scope("bias"):
            b8 = init_bit([10])
            variable_summaries(b8)
        with tf.name_scope("output"):
            y_con = tf.nn.softmax(tf.matmul(y2, w8) + b8, name="output")
    with tf.name_scope("loss"):
        loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_con,1e-10,1.0))))
    tf.summary.scalar("loss", loss_function)
    with tf.name_scope("train"):
        train = tf.train.GradientDescentOptimizer(0.001).minimize(loss_function)
    with tf.name_scope("accuracy"):
        with tf.name_scope("correction_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_con, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar("accurzcy", accuracy)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("logs/log1/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs/log1/test")
        batch_size = 50
        tf.global_variables_initializer().run()
        train_batch = train_next_batch(batch_size)
        test_batch = test_next_batch(batch_size)
        for i in range(2001):
            sess.run(train, feed_dict={x: train_batch[0], y: train_batch[1], keep_prob: 1})

            if (i%100 == 0):
                summary = sess.run(merged, feed_dict={x: train_batch[0], y: train_batch[1], keep_prob: 1})
                train_writer.add_summary(summary, i)

                summary = sess.run(merged, feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1})
                test_writer.add_summary(summary, i)

                train_accuracy = sess.run(accuracy, feed_dict={x:train_batch[0], y: train_batch[1], keep_prob: 1})
                print("after %d steps the trainnig accuracy is %g" % (i, train_accuracy))

                test_accuracy = sess.run(accuracy, feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1})
                print("after %d steps the testnig accuracy is %g" % (i, test_accuracy))
                print("")
