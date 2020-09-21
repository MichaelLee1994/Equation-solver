import time
import os
import shutil
import numpy as np
import tensorflow as tf
import numpy.polynomial.legendre as leg

import NNUtil
import baseUtil
import util


def solution(x):
    return x*np.sin(x)


def source_term(x):
    return 2 * tf.math.cos(x) - 2 * x * tf.math.sin(x)


equation_info = (solution, source_term)


class DirectLocSolution:
    def __init__(self, num_quad, partitions, layers, equation):
        self.upper_bound = partitions[-1]
        self.lower_bound = partitions[0]
        self.solution, self.source_term = equation
        self.u_bd = self.solution(np.array([self.lower_bound, self.upper_bound]).reshape(2, 1))
        self.partitions = partitions
        self.num_elements = len(partitions) - 1  # P

        self.hidden_layers = layers
        self.num_quad = num_quad    # Q

        # create intervals, shape: (P, 2, 1)
        # self.x_mid_bd = np.array([self.partitions[0:-1], self.partitions[1:]]).T.reshape((-1, 2, 1))
        # self.x_mid_bd_diag = np.array([self.partitions[0:-1], self.partitions[1:]])
        self.x_mid_bd_list = np.hsplit(np.array([self.partitions[0:-1], self.partitions[1:]]), self.num_elements)

        self.xi, self.guass_weights = leg.leggauss(self.num_quad)
        self.W = np.ones((1, self.num_quad)) * self.guass_weights   # shape: (1, Q)
        # self.C = 0.5 * (self.x_mid_bd[:, 1:2, 0] - self.x_mid_bd[:, 0:1, 0]).T  # shape: (1, P)
        # self.C_diag = 0.5 * (self.x_mid_bd_diag[1:2, :] - self.x_mid_bd_diag[0:1, :])
        self.C = np.array([0.5*(elem[1, 0]-elem[0, 0]) for elem in self.x_mid_bd_list])

        # create mapped x, shape: (P, Q, 1)
        # self.x = baseUtil.linear_multiple_map(self.xi, self.partitions)
        # self.x_diag = baseUtil.linear_multiple_map_diag(self.xi, self.partitions)
        self.x_list = baseUtil.linear_multiple_map_list(self.xi, self.partitions)   # shape: (Q, 1)

        # initialize NN
        # self.weights, self.biases = NNUtil.initialize_mnn(self.hidden_layers, self.num_elements)
        # self.weights, self.biases = NNUtil.initialize_mnn_diag(self.hidden_layers, self.num_elements)
        self.weights_list, self.biases_list = NNUtil.initialize_mnn_list(self.hidden_layers, self.num_elements)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        # self.x_tf = tf.placeholder(tf.float32, shape=[self.num_elements, None, self.x.shape[2]], name='input')
        # self.x_mid_bd_tf = tf.placeholder(tf.float32, shape=[self.num_elements, None, self.x_mid_bd.shape[2]], name='mid_bd_input')
        # self.predict_input_tf = tf.placeholder(tf.float32, shape=[self.num_elements, None, 1], name='pred_input')

        # self.x_diag_tf = tf.placeholder(tf.float32, shape=[None, self.x_diag.shape[1]], name='input')
        # self.x_mid_bd_diag_tf = tf.placeholder(tf.float32, shape=[None, self.x_mid_bd_diag.shape[1]], name='mid_bd_input')
        # self.predict_input_tf = tf.placeholder(tf.float32, shape=[None, self.num_elements], name='pred_input')

        self.x_list_tf = [tf.placeholder(tf.float32, shape=[None, 1], name='input_{}'.format(elem))
                          for elem in range(self.num_elements)]
        self.x_mid_bd_list_tf = [tf.placeholder(tf.float32, shape=[None, 1], name='mid_bd_input_{}'.format(elem))
                                 for elem in range(self.num_elements)]
        self.predict_input_list_tf = [tf.placeholder(tf.float32, shape=[None, 1], name='pred_input_{}'.format(elem))
                                      for elem in range(self.num_elements)]

        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')

        self.W_tf = tf.constant(self.W, dtype=tf.float32)
        self.u_bd_tf = tf.constant(self.u_bd, dtype=tf.float32, name='bd_truth')

        self.tf_dict = self._create_feed_dict()
        # self.tf_dict = {self.x_diag_tf: self.x_diag, self.x_mid_bd_diag_tf: self.x_mid_bd_diag}

        # self.u_bd_pred, self.c0, self.c1 = self.net_bd(self.x_mid_bd_tf)
        # self.res_int_pred = self.net_residual(self.x_tf)

        self.u_bd_pred, self.c0, self.c1 = self.net_bd_list(self.x_mid_bd_list_tf)
        self.res_int_pred = self.net_residual_list(self.x_list_tf)
        self.u_pred = self.net_pred_list(self.predict_input_list_tf)

        self.b_op, self.w_op, self.new_wb = \
            NNUtil.compute_least_square_single_list(self, self.weights_list, self.biases_list)

        self.loss = tf.nn.l2_loss(self.res_int_pred) \
            + tf.nn.l2_loss(self.u_bd_pred - self.u_bd_tf) + tf.nn.l2_loss(self.c0) + tf.nn.l2_loss(self.c1)

        self.train_op_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _create_feed_dict(self):
        x_dict = {k: v for k, v in zip(self.x_list_tf, self.x_list)}
        x_mid_bd_dict = {k: v for k, v in zip(self.x_mid_bd_list_tf, self.x_mid_bd_list)}
        return {**x_dict, **x_mid_bd_dict}

    # def net_bd(self, mid_bd):
    #     u = NNUtil.neural_net(mid_bd, self.weights, self.biases)
    #     u_x = tf.gradients(u, mid_bd)[0]
    #     u_bd_pred = tf.stack([u[0, 0:1, 0], u[-1, 1:2, 0]], axis=0)
    #     diff = u[1:, 0:1, 0] - u[:-1, 1:2, 0]
    #     diff_x = u_x[1:, 0:1, 0] - u_x[-1:, 1:2, 0]
    #     return u_bd_pred, diff, diff_x
    #
    # def net_residual(self, x):
    #     u = NNUtil.neural_net(x, self.weights, self.biases)
    #     u_x = tf.gradients(u, x)[0]
    #     u_xx = tf.gradients(u_x, x)[0]
    #     f = self.source_term(x)
    #     equation_residual = tf.math.log(tf.math.cosh(u_xx - u - f))
    #     # equation_residual = tf.math.square(u_xx - u - f)
    #     equation_residual_sq = tf.transpose(tf.squeeze(equation_residual, axis=2))  # reshape to (Q, P)
    #     integral_residual = self.C * tf.matmul(self.W_tf, equation_residual_sq)
    #     return integral_residual

    # def net_bd_diag(self, mid_bd):
    #     u = NNUtil.neural_net(mid_bd, self.weights, self.biases)
    #     u_x = tf.gradients(u, mid_bd)[0]
    #     u_bd_pred = tf.stack([u[0:1, 0], u[1:2, -1]], axis=0)
    #     diff = u[0:1, 1:] - u[1:2, :-1]
    #     diff_x = u_x[0:1, 1:] - u_x[1:2, :-1]
    #     return u_bd_pred, diff, diff_x
    #
    # def net_residual_diag(self, x):
    #     u = NNUtil.neural_net(x, self.weights, self.biases)
    #     u_x = tf.gradients(u, x)[0]
    #     u_xx = tf.gradients(u_x, x)[0]
    #     f = self.source_term(x)
    #     equation_residual = tf.math.log(tf.math.cosh(u_xx - u - f))
    #     integral_residual = self.C_diag * tf.matmul(self.W_tf, equation_residual)
    #     return integral_residual
    #
    # def net_pred(self, x):
    #     u = NNUtil.neural_net(x, self.weights, self.biases)
    #     return u

    def net_bd_list(self, mid_bd: list):
        u = NNUtil.neural_net_list(mid_bd, self.weights_list, self.biases_list)
        u_x = tf.gradients(u, mid_bd)
        u_bd_pred = tf.stack([u[0][0:1, 0], u[-1][1:2, 0]], axis=0)
        u_concat = tf.concat(u, axis=1)
        u_x_concat = tf.concat(u_x, axis=1)
        diff = u_concat[0:1, 1:] - u_concat[1:2, :-1]
        diff_x = u_x_concat[0:1, 1:] - u_x_concat[1:2, :-1]
        return u_bd_pred, diff, diff_x

    def net_residual_list(self, x: list) -> tf.Tensor:
        u = NNUtil.neural_net_list(x, self.weights_list, self.biases_list)

        u_x = tf.gradients(u, x, unconnected_gradients='zero', name="GRAD_x")
        u_xx = tf.gradients(u_x, x, name="GRAD_xx")

        u_concat = tf.concat(u, axis=1)
        x_concat = tf.concat(x, axis=1)
        u_xx_concat = tf.concat(u_xx, axis=1)

        f_concat = self.source_term(x_concat)
        # equation_residual = tf.math.log(tf.math.cosh(u_xx_concat - u_concat - f_concat))
        equation_residual = tf.math.square(u_xx_concat - u_concat - f_concat)
        integral_residual = self.C * tf.matmul(self.W_tf, equation_residual)
        return integral_residual

    def net_pred_list(self, x: list) -> list:
        u = NNUtil.neural_net_list(x, self.weights_list, self.biases_list)
        return u


if __name__ == "__main__":
    working_dir = "./Test_Collection/direct_loc/"
    paras = util.load_parameter(working_dir + "paras.txt")
    Q = int(paras["Q"])
    width = int(paras["width"])
    depth = int(paras["depth"])
    max_epoch = int(paras["max_epoch"])
    num_tested_per_element = int(paras["num_tested_per_element"])
    pivot = util.to_list(paras["pivot"])

    upperBound = pivot[-1]
    lowerBound = pivot[0]
    num_element = len(pivot) - 1
    hidden_layers = [width] * depth

    new_dir = "LS/{}_{}_{}_sig=1e-4/Q={}/{}_{}({})/".format(lowerBound, upperBound, num_element, Q, depth, width, max_epoch)
    dump_dir = working_dir + new_dir
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    shutil.copy(working_dir + "paras.txt", dump_dir)

    model = DirectLocSolution(Q, pivot, hidden_layers, equation_info)

    # writer = tf.summary.FileWriter(dump_dir, tf.get_default_graph())
    # writer.close()

    start_time = time.time()
    epoch, loss_history = NNUtil.train(model, max_epoch, dump_dir, ls_op=True)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    # x_test, u_pred, u_truth = NNUtil.mnn_predict_list(model, num_tested_per_element)
    #
    # error_u = np.sqrt(np.square(u_pred - u_truth).mean())
    # print('Error u: %e' % error_u)
    #
    # dump_data = np.hstack((u_truth, u_pred))
    #
    # np.savetxt(dump_dir + 'result.csv', X=dump_data, header="truth,prediction", delimiter=',')
    #
    # util.plot_result(epoch, loss_history, x_test, u_truth, u_pred, dump_dir, error_u)
