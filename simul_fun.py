import tensorflow as tf
import numpy as np
import os
import time
import numpy.polynomial.legendre as leg

import util


def xsin(a_=1, w_=1, if_tf=True):
    def function(x):
        return a_ * x * np.sin(w_ * x)

    def function_tf(x_tf):
        return a_ * x_tf * tf.math.sin(w_ * x_tf)
    if if_tf:
        return function_tf
    else:
        return function


class SimFunction:
    def __init__(self, lower, upper, num_quad, fun_tf, layers_info: list, trainable_last):
        self.upper_bound = upper
        self.lower_bound = lower
        self.function_tf = fun_tf
        self.layers = layers_info
        self.num_quad = num_quad

        xi, self.gauss_weights = leg.leggauss(self.num_quad)
        self.x_training = self.linear_map(xi)

        # initialize NN
        self.weights, self.biases = self.initialize_nn(trainable_last)

        # tf placeholders and graph
        self.input_tf = tf.placeholder(tf.float32, shape=[None, 1], name='input')
        self.output_tf = self.neural_net()
        self.truth_tf = self.function_tf(self.input_tf)
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')
        self.lam_tf = tf.placeholder(tf.float32, name='lam')
        self.loss = tf.reduce_sum(((self.output_tf - self.truth_tf) ** 2) * self.gauss_weights)
        self.train_op_Adam = tf.train.AdamOptimizer(learning_rate=self.lr_tf).minimize(self.loss)
        self.ls_op, self.new_wb = self.compute_ls()

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def linear_map(self, xi):
        x = (self.upper_bound + self.lower_bound) / 2.0 + (self.upper_bound - self.lower_bound) * xi / 2.0
        return x.reshape((-1, 1))

    def compute_ls(self):
        num_layers = len(self.layers)
        result = self.input_tf
        for layer in range(0, num_layers - 2):
            weight = self.weights[layer]
            bias = self.biases[layer]
            result = tf.tanh(tf.add(tf.matmul(result, weight), bias))

        samples = tf.shape(self.input_tf)[-2]
        xw = result
        extra_col = tf.ones([samples, 1], dtype=tf.float32)
        extra_row = tf.ones([1, samples], dtype=tf.float32)
        xt = tf.concat([tf.transpose(xw), extra_row], axis=0)
        x = tf.concat([xw, extra_col], axis=1) * (self.gauss_weights.reshape((-1, 1)))
        left_side = tf.matmul(xt, x) + 2 * self.lam_tf * tf.eye(tf.shape(x)[-1])

        source_terms = tf.cast(self.function_tf(self.input_tf) * (self.gauss_weights.reshape((-1, 1))), dtype=tf.float32)
        right_side = tf.matmul(xt, source_terms)

        new_wb = tf.linalg.solve(left_side, right_side)

        assign_bias_op = self.biases[-1].assign(new_wb[-1:, 0:1])
        assign_weights_op = self.weights[-1].assign(new_wb[0:-1, 0:1])
        ls_op = (assign_weights_op, assign_bias_op)

        return ls_op, new_wb

    def initialize_nn(self, trainable_last=True) -> tuple:
        """
        Initialize neural network (dense layer)
        :return: initialized weights and biases for each layer
        :param trainable_last: whether the last layer is trainable
        """
        def _xavier_init(size: list, trainable=True):
            in_dim = size[-2]
            out_dim = size[-1]
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            return tf.Variable(tf.truncated_normal(size, stddev=xavier_stddev), dtype=tf.float32, trainable=trainable)

        weights = []
        biases = []
        num_layers = len(self.layers)

        for layer in range(0, num_layers - 2):
            weight = _xavier_init(size=[self.layers[layer], self.layers[layer + 1]])
            bias = tf.Variable(tf.zeros([1, self.layers[layer + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(weight)
            biases.append(bias)
        # the last layer
        weight = _xavier_init(size=[self.layers[-2], self.layers[-1]], trainable=trainable_last)
        bias = tf.Variable(tf.zeros([1, self.layers[-1]], dtype=tf.float32), dtype=tf.float32, trainable=trainable_last)
        weights.append(weight)
        biases.append(bias)
        return weights, biases

    def neural_net(self):
        num_layers = len(self.weights) + 1

        result = self.input_tf
        for layer in range(0, num_layers - 2):
            weight = self.weights[layer]
            bias = self.biases[layer]
            result = tf.tanh(tf.add(tf.matmul(result, weight), bias))
        weight = self.weights[-1]  # last layer
        bias = self.biases[-1]
        result = tf.add(tf.matmul(result, weight), bias)
        return result

    def train(self, max_iter, log_dir, adjust_epoch, reg=0.0,
              learning_rate=1e-3, threshold=1e-7, if_ls=False):
        if self.x_training.shape[-1] != 1:
            raise ValueError("The last dimension of x has to be one!")
        input_dict = {self.input_tf: self.x_training, self.lam_tf: reg}
        lr = learning_rate
        lr_dict = {self.lr_tf: lr}
        feed_dict = {**input_dict, **lr_dict}
        adjusting = False
        epochs = []
        loss_records = []

        with open(log_dir + "Training_history.txt", 'w') as writer:
            writer.writelines("--- Optimizer: lr = %.2e ---\n" % lr)
            start = time.time()

            for it in range(max_iter):
                if it >= adjust_epoch:
                    adjusting = True
                    loss_base = int(np.log10(self.sess.run(self.loss, feed_dict)))
                # train the model
                if it != 0:
                    self.sess.run(self.train_op_Adam, feed_dict)
                check_point = max(1, int(max_iter / 100))

                # checkpoints
                if it % check_point == 0:
                    elapsed = time.time() - start
                    epochs.append(it)
                    loss_value, weights, biases = self.sess.run([self.loss, self.weights, self.biases], input_dict)
                    cp_result = "It: %d, Loss: %.3e,  Time: %.2f, Progress: %.2f%%" % \
                                (it, loss_value, elapsed, it * 100.0 / max_iter)
                    print(cp_result)
                    writer.writelines("\n".join([cp_result,
                                                 "=== current wb ===", str(weights[-1]), str(biases[-1]), "=======\n"]))

                    if if_ls and it % (2 * check_point) == 0:
                        _, new_wb = self.sess.run([self.ls_op, self.new_wb], feed_dict)
                        new_loss_value = self.sess.run(self.loss, feed_dict)
                        optimized_result = "--- LS OPTIMIZING ---\nLoss: %.3e, Loss Optimized: %.2f%%" % \
                                           (new_loss_value, (loss_value - new_loss_value) * 100.0 / loss_value)
                        print(optimized_result)
                        writer.writelines("\n".join([optimized_result,
                                                     "=== old ====", str(weights[-1]), str(biases[-1]),
                                                     "=== new ====", str(new_wb), "=======\n"]))
                        loss_value = new_loss_value

                    log_loss = np.log10(loss_value)
                    loss_records.append(log_loss)

                    #  check threshold
                    if loss_value < threshold:
                        return epochs, loss_records
                    # if loss becomes too small, learning rate should be adjusted
                    if adjusting and loss_base > int(log_loss):
                        loss_base = int(log_loss)
                        lr /= 10
                        lr_dict = {self.lr_tf: lr}
                        feed_dict = {**input_dict, **lr_dict}
                        writer.writelines("\n>>> ADJUSTING <<<\n>>> Optimizer: lr = %.2e <<<\n" % lr)
                    start = time.time()
        return epochs, loss_records

    def test(self, x_input):
        feed_dict = {self.input_tf: x_input}
        pred, y_truth = self.sess.run([self.output_tf, self.truth_tf], feed_dict)
        return pred, y_truth


if __name__ == "__main__":
    working_dir = "./Test_Collection/sim_function/"
    a, w = 1, 1
    lower_bound, upper_bound = 0, 5
    quad_order = 28

    layers = [50, 50]
    layers.insert(0, 1)
    layers.append(1)

    init_lr = 1e-5
    max_epoch = 100
    adjusting_epoch = 101
    min_loss = 1e-7
    num_testing = 1000
    least_squared = True
    if least_squared:
        lam = 1e-6
    else:
        lam = 0.0

    new_dir = "/{}_{}_{}_{}/Q={}, init_lr={}_{}_{}/{}({})/"\
        .format(a, w, lower_bound, upper_bound, quad_order, init_lr, least_squared, lam, layers, max_epoch)
    dump_dir = working_dir + new_dir
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model = SimFunction(lower_bound, upper_bound, quad_order, xsin(a, w), layers, trainable_last=not least_squared)

    start_time = time.time()
    epoch_list, loss_history = model.train(max_epoch, dump_dir, reg=lam,
                                           adjust_epoch=adjusting_epoch, learning_rate=init_lr,
                                           threshold=min_loss, if_ls=least_squared)
    training_time = time.time() - start_time
    print('Training time: %.4f' % training_time)

    x_testing = np.linspace(lower_bound, upper_bound, num_testing).reshape((-1, 1))
    prediction, truth = model.test(x_testing)

    error_u = np.sqrt(np.square(prediction - truth).mean())
    print('Error u: %e' % error_u)

    with open(dump_dir + "info.txt", 'w') as log:
        writing_items = [">>> Function <<<",
                         "Interval = [%.2f, %.2f]" % (lower_bound, upper_bound), "Q = %d" % quad_order,
                         "(a, w) = (%.2f, %.2f)" % (a, w),
                         "\n>>> Network <<<",
                         "Layers = %s" % str(layers), "Max_epochs = %d" % max_epoch, "Num_testing = %d" % num_testing,
                         "Init_lr = %.2e" % init_lr, "Least_squared = %s" % least_squared, "Lambda = %.2f" % lam,
                         "Loss_threshold = %.1e" % min_loss, "Adjusting_period = %d" % adjusting_epoch,
                         "\n>>> Result <<<",
                         "Training time: %.4fs" % training_time, "L2_error: %e" % error_u]
        log.writelines("\n".join(writing_items))

    util.plot_result(epoch_list, loss_history, x_testing, truth, prediction, dump_dir, error_u)
