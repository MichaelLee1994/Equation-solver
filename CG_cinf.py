import time
import os
import shutil
import numpy as np
import tensorflow as tf

import NNUtil
import baseUtil
import util


class CGcinfSolution:
    def __init__(self, N, partitions, layers, equation):
        self.upper_bound = partitions[-1]
        self.lower_bound = partitions[0]
        self.x_bd = np.array([self.lower_bound, self.upper_bound]).reshape(2, 1)
        self.solution, self.source_term = equation
        self.u_bd = self.solution(self.x_bd)
        self.partitions = partitions
        self.num_elements = len(partitions) - 1

        # create intervals
        self.x_mid_bd = np.array([self.partitions[0:-1], self.partitions[1:]])

        self.layers = layers
        self.N = N

        self.xi, self.W, self.W_d, self.W_bd = baseUtil.compute_legendre_gauss(self.N)
        self.C = self.x_mid_bd[1:2, :] - self.x_mid_bd[0:1, :]

        # create mapped x, shape: [Q, parts]
        self.x = baseUtil.linear_multiple_map(self.xi, self.partitions)

        # initialize NN
        self.weights, self.biases = NNUtil.initialize_nn(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]], name='input')
        self.x_mid_bd_tf = tf.placeholder(tf.float32, shape=[None, self.x_mid_bd.shape[1]], name='mid_bd_input')
        self.x_bd_tf = tf.placeholder(tf.float32, shape=[None, self.x_bd.shape[1]], name='bd_input')
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')

        self.W_tf = tf.constant(self.W, dtype=tf.float32)
        self.W_d_tf = tf.constant(self.W_d, dtype=tf.float32)
        self.W_bd_tf = tf.constant(self.W_bd, dtype=tf.float32)
        self.u_bd_tf = tf.constant(self.u_bd, dtype=tf.float32, name='bd_truth')

        self.tf_dict = {self.x_tf: self.x, self.x_bd_tf: self.x_bd, self.x_mid_bd_tf: self.x_mid_bd}

        self.u_bd_pred = NNUtil.neural_net(self.x_bd_tf, self.weights, self.biases)

        self.res_bd_pred = self.net_bd(self.x_mid_bd_tf)
        self.res_int_pred = self.net_residue(self.x_tf)

        self.loss = tf.nn.l2_loss(self.res_bd_pred + self.res_int_pred) + \
            tf.nn.l2_loss(self.u_bd_pred - self.u_bd_tf)

        self.train_op_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def net_bd(self, x):
        x_flatten = tf.reshape(x, [-1, 1])
        u_flatten = NNUtil.neural_net(x_flatten, self.weights, self.biases)
        u_x_flatten = tf.gradients(u_flatten, x_flatten)
        u_x = tf.reshape(u_x_flatten, [-1, self.num_elements])
        residue_bd = tf.matmul(self.W_bd_tf, u_x)
        return residue_bd

    def net_residue(self, x):
        x_flatten = tf.reshape(x, [-1, 1])
        u_flatten = NNUtil.neural_net(x_flatten, self.weights, self.biases)
        u_x_flatten = tf.gradients(u_flatten, x)
        u = tf.reshape(u_flatten, [-1, self.num_elements])
        u_x = tf.reshape(u_x_flatten, [-1, self.num_elements])
        f = 2 * tf.math.cos(x) - 2 * x * tf.math.sin(x)
        integral = tf.matmul(self.W_d_tf, u_x) + 0.5*self.C*tf.matmul(self.W_tf, u + f)
        return integral

    # def train(self, max_iter):
    #     epochs = []
    #     loss_records = []
    #
    #     start_time = time.time()
    #     for it in range(max_iter):
    #         self.sess.run(self.train_op_Adam, self.tf_dict)
    #         # Print
    #         if it % 100 == 0:
    #             elapsed = time.time() - start_time
    #             epochs.append(it)
    #             loss_value = self.sess.run(self.loss, self.tf_dict)
    #             print('It: %d, Loss: %.3e,  Time: %.2f, Progress: %.2f%%' %
    #                   (it, loss_value, elapsed, it * 100.0 / max_iter))
    #             loss_records.append(np.log10(loss_value))
    #             start_time = time.time()
    #
    #             if loss_value < 1e-3:
    #                 return epochs, loss_records
    #     return epochs, loss_records

    # def predict(self, test):
    #     u_prediction = self.sess.run(self.u_bd_pred, {self.x_bd_tf: test[:, 0:1]})
    #     return u_prediction


if __name__ == "__main__":
    working_dir = "./Test_Collection/MNN_CG_test/"
    paras = util.load_parameter(working_dir + "paras.txt")
    N = int(paras["N"])
    width = int(paras["width"])
    depth = int(paras["depth"])
    max_epoch = int(paras["max_epoch"])
    num_tested_per_element = int(paras["num_tested_per_element"])
    pivot = util.to_list(paras["pivot"])

    upperBound = pivot[-1]
    lowerBound = pivot[0]
    num_element = len(pivot) - 1
    layers = [width] * depth
    layers.append(1)
    layers.insert(0, 1)
    new_dir = "c_inf/{}_{}_{}/N={}/{}_{} ({})/".format(lowerBound, upperBound, num_element, N, depth, width, max_epoch)
    dump_dir = working_dir + new_dir
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    shutil.copy(working_dir + "paras.txt", dump_dir)

    model = CGcinfSolution(N, pivot, layers)

    start_time = time.time()
    epoch, loss_history = NNUtil.train(model, max_epoch)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    x_test = np.linspace(lowerBound, upperBound, num_tested_per_element*num_element).reshape(-1, 1)

    u_truth = model.solution(x_test)
    u_pred = NNUtil.nn_predict(model, x_test)

    error_u = np.sqrt(np.square(u_pred - u_truth).mean())
    print('Error u: %e' % error_u)

    dump_data = np.hstack((u_truth, u_pred))

    np.savetxt(dump_dir + 'result.csv', X=dump_data, header="truth,prediction", delimiter=',')

    util.plot_result(epoch, loss_history, x_test, u_truth, u_pred, dump_dir, error_u)
