import time
import os
import shutil
import numpy as np
import tensorflow as tf

import NNUtil
import baseUtil
import util


class CBSolutionNN:
    def __init__(self, N, partitions, layers, equation):
        self.upper_bound = partitions[-1]
        self.lower_bound = partitions[0]
        self.x_bd = np.array([self.lower_bound, self.upper_bound]).reshape(2, 1)
        self.solution, self.source_term = equation
        self.u_bd = self.solution(self.x_bd)
        self.partitions = partitions
        self.num_elements = len(partitions) - 1

        self.hidden_layers = layers
        self.N = N

        # create intervals
        self.x_mid_bd = np.array([self.partitions[0:-1], self.partitions[1:]])

        # compute weights
        self.xi, self.W, self.W_d, self.sW = baseUtil.compute_continuous_basis(self.N)
        self.C = self.x_mid_bd[1:2, :] - self.x_mid_bd[0:1, :]

        # create mapped x, shape: [Q, K]
        self.x = baseUtil.linear_multiple_map(self.xi, self.partitions)

        # initialize NN
        self.weights, self.biases = NNUtil.initialize_mnn_tensor(self.hidden_layers, self.num_elements)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]], name='input')
        self.x_mid_bd_tf = tf.placeholder(tf.float32, shape=[None, self.x_mid_bd.shape[1]], name='mid_bd_input')
        self.x_bd_tf = tf.placeholder(tf.float32, shape=[None, self.x_bd.shape[1]], name='bd_input')
        self.u_bd_tf = tf.placeholder(tf.float32, shape=[None, self.u_bd.shape[1]], name='bd_truth')
        self.predict_input_tf = tf.placeholder(tf.float32, shape=[None, self.num_elements], name='pred_input')
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')

        self.W_tf = tf.constant(self.W, dtype=tf.float32)
        self.W_d_tf = tf.constant(self.W_d, dtype=tf.float32)
        self.sW_tf = tf.constant(self.sW, dtype=tf.float32)

        self.tf_dict = {self.x_tf: self.x, self.x_bd_tf: self.x_bd, self.x_mid_bd_tf: self.x_mid_bd,
                        self.u_bd_tf: self.u_bd}

        self.small_bd_pred, self.u_bd_pred, self.c0 = self.net_bd(self.x_mid_bd_tf)
        self.res_int_pred = self.net_residue(self.x_tf)
        self.res_int_small_pred = self.net_residue_small(self.x_tf)
        self.u_pred = self.net_pred(self.predict_input_tf)

        self.loss = tf.nn.l2_loss(self.res_int_pred) \
            + tf.nn.l2_loss(self.u_bd_pred - self.u_bd_tf) \
            + tf.nn.l2_loss(self.small_bd_pred + self.res_int_small_pred) + tf.nn.l2_loss(self.c0)

        self.train_op_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def net_bd(self, x):
        u = NNUtil.neural_net(x, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        first = u_x[0:1, 0:1] - (u[1:2, 0:1] - u[0:1, 0:1])/(x[1:2, 0:1] - x[0:1, 0:1])
        last = -u_x[1:2, -1:] + (u[1:2, -1:] - u[0:1, -1:])/(x[1:2, -1:] - x[0:1, -1:])
        left = (u[1:2, :-1] - u[0:1, :-1])/(x[1:2, :-1] - x[0:1, :-1])
        right = -(u[1:2, 1:] - u[0:1, 1:])/(x[1:2, 1:] - x[0:1, 1:])
        hats = left + right
        diff = u[0:1, 1:] - u[1:2, :-1]
        u_bd_pred = tf.stack([u[0:1, 0], u[1:2, -1]], axis=0)
        return tf.transpose(tf.concat([first, hats, last], axis=1)), u_bd_pred, diff

    def net_residue(self, x):
        u = NNUtil.neural_net(x, self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        f = 2 * tf.math.cos(x) - 2 * x * tf.math.sin(x)
        integral_residue = tf.matmul(self.W_d_tf, u_x) + 0.5*self.C*tf.matmul(self.W_tf, u + f)
        return tf.stack(integral_residue)

    def net_residue_small(self, x):
        u = NNUtil.neural_net(x, self.weights, self.biases)
        f = self.source_term(x)
        integrals = 0.5*self.C*tf.matmul(self.sW_tf, u + f)
        first = integrals[0:1, 0:1]
        hats = integrals[0:1, 1:] + integrals[1:2, 0:-1]
        last = integrals[1:2, -1:]
        return tf.transpose(tf.concat([first, hats, last], axis=1))

    def net_pred(self, x):
        u = NNUtil.neural_net(x, self.weights, self.biases)
        return u


if __name__ == "__main__":
    working_dir = "./Test_Collection/CB/"
    paras = util.load_parameter(working_dir + "paras.txt")
    N = int(paras["N"])
    width = int(paras["width"])
    depth = int(paras["depth"])
    max_epoch = int(paras["max_epoch"])
    num_tested_per_element = int(paras["num_tested_per_element"])
    pivot = util.to_list(paras["pivot"])

    upper_bound = pivot[-1]
    lower_bound = pivot[0]
    num_element = len(pivot) - 1
    hidden_layers = [width] * depth

    new_dir = "{}_{}_{}/N={}/{}_{}/".format(lower_bound, upper_bound, num_element, N, depth, width)
    dump_dir = working_dir + new_dir
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    shutil.copy(working_dir + "paras.txt", dump_dir)

    model = CBSolutionNN(N, pivot, hidden_layers)

    start_time = time.time()
    epoch, loss_history = NNUtil.train(model, max_epoch)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    x_test, u_pred, u_truth = NNUtil.mnn_predict(model, num_tested_per_element)

    error_u = np.sqrt(np.square(u_pred - u_truth).mean())
    print('Error u: %e' % error_u)

    prediction = np.hstack((u_truth, u_pred))

    np.savetxt(dump_dir + 'result.csv', X=prediction, header="truth,prediction", delimiter=',')

    util.plot_result(epoch, loss_history, x_test, u_truth, u_pred, dump_dir, error_u)
