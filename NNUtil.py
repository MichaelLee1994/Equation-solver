import pickle
import time

import tensorflow as tf
import numpy as np


def _xavier_init(size: list, trainable=True):
    in_dim = size[-2]
    out_dim = size[-1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal(size, stddev=xavier_stddev), dtype=tf.float32, trainable=trainable)


def initialize_nn(layers: list, trainable_last=True) -> tuple:
    """
    Initialize neural network (dense layer)
    :param layers: a list containing number of neurons for each layer
    :return: initialized weights and biases for each layer
    :param trainable_last: whether the last layer is trainable
    """
    weights = []
    biases = []
    num_layers = len(layers)

    for layer in range(0, num_layers - 2):
        weight = _xavier_init(size=[layers[layer], layers[layer + 1]])
        bias = tf.Variable(tf.zeros([1, layers[layer + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(weight)
        biases.append(bias)
    # the last layer
    weight = _xavier_init(size=[layers[-2], layers[-1]], trainable=trainable_last)
    bias = tf.Variable(tf.zeros([1, layers[-1]], dtype=tf.float32), dtype=tf.float32, trainable=trainable_last)
    weights.append(weight)
    biases.append(bias)
    return weights, biases


def initialize_mnn_diag(layers_: list, num_element):
    # layers: hidden layers for each element
    layers = layers_.copy()
    layers.append(1)
    layers.insert(0, 1)
    if num_element < 2:
        return initialize_nn(layers)
    weights = []
    biases = []
    num_layer = len(layers)
    for layer in range(0, num_layer - 1):
        zeros = tf.Variable(tf.zeros([layers[layer], layers[layer + 1]], dtype=tf.float32), dtype=tf.float32,
                            trainable=False)
        trainable_weight = _xavier_init(size=[layers[layer], layers[layer + 1]])
        diag_weight = tf.concat([trainable_weight, tf.tile(zeros, [1, num_element - 1])], axis=-1)
        bias = tf.Variable(tf.zeros([1, layers[layer + 1] * num_element], dtype=tf.float32), dtype=tf.float32)
        for i in range(1, num_element - 1):
            trainable_weight = _xavier_init(size=[layers[layer], layers[layer + 1]])
            row = tf.concat([tf.tile(zeros, [1, i]), trainable_weight, tf.tile(zeros, [1, num_element-1-i])], axis=-1)
            diag_weight = tf.concat([diag_weight, row], axis=0)
        trainable_weight = _xavier_init(size=[layers[layer], layers[layer + 1]])
        last_row = tf.concat([tf.tile(zeros, [1, num_element - 1]), trainable_weight], axis=-1)
        diag_weight = tf.concat([diag_weight, last_row], axis=0)
        weights.append(diag_weight)
        biases.append(bias)
    return weights, biases


def initialize_mnn_tensor(layers_: list, num_elem):
    # layers: hidden layers for each element
    layers = layers_.copy()
    layers.append(1)
    layers.insert(0, 1)
    if num_elem < 2:
        return initialize_nn(layers)
    weights = []
    biases = []
    num_layer = len(layers)
    for layer in range(0, num_layer - 1):
        weight = _xavier_init(size=[num_elem, layers[layer], layers[layer + 1]])
        bias = tf.Variable(tf.zeros([num_elem, 1, layers[layer + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def initialize_mnn_list(layers_: list, num_elem, trainable_last=True):
    # layers: hidden layers for each element
    layers = layers_.copy()
    layers.append(1)
    layers.insert(0, 1)
    # if num_elem < 2:
    #     return initialize_nn(layers)
    weights_list = []
    biases_list = []
    num_layer = len(layers)
    for layer in range(0, num_layer - 2):
        layer_weights = []
        layer_biases = []
        for elem in range(num_elem):
            weight = _xavier_init(size=[layers[layer], layers[layer + 1]])
            bias = tf.Variable(tf.zeros([1, layers[layer + 1]], dtype=tf.float32), dtype=tf.float32)
            layer_weights.append(weight)
            layer_biases.append(bias)
        weights_list.append(layer_weights)
        biases_list.append(layer_biases)
    # last layer
    layer_weights = []
    layer_biases = []
    for elem in range(num_elem):
        weight = _xavier_init(size=[layers[-2], layers[-1]], trainable=trainable_last)
        bias = tf.Variable(tf.zeros([1, layers[-1]], dtype=tf.float32), dtype=tf.float32, trainable=trainable_last)
        layer_weights.append(weight)
        layer_biases.append(bias)
    weights_list.append(layer_weights)
    biases_list.append(layer_biases)
    return weights_list, biases_list


# def compute_least_square_diag(model, weights, biases):
#     def _compute_sliced_grad(y, x_cols, last_dim, num_elem):
#         if num_elem < 2:
#             y_sliced = tf.split(y, last_dim, axis=1)
#             grad_list = [tf.gradients(yl, x_cols[0])[0] for yl in y_sliced]
#         else:
#             # raise NotImplementedError
#             # y_sliced = tf.split(y, last_dim, axis=1)
#             grad_list = []
#             for i in range(last_dim*num_elem):
#                 idx = i % num_elem
#                 grad = tf.gradients(y[:, i:i+1], x_cols[idx:idx + 1])[0]
#                 grad_list.append(grad)
#         return tf.concat(grad_list, axis=1)
#
#     def _build_last_bias(quad_order, num_elem):
#         zeros = tf.zeros([quad_order, 1], dtype=tf.float32)
#         diag = -tf.ones([quad_order, 1], dtype=tf.float32)
#         bias = tf.concat([diag, tf.tile(zeros, [1, num_elem - 1])], axis=-1)
#         for i in range(1, num_elem - 1):
#             row = tf.concat([tf.tile(zeros, [1, i]), diag, tf.tile(zeros, [1, num_elem - 1 - i])], axis=-1)
#             bias = tf.concat([bias, row], axis=0)
#         last_row = tf.concat([tf.tile(zeros, [1, num_elem - 1]), diag], axis=-1)
#         bias = tf.concat([bias, last_row], axis=0)
#         return bias
#
#     def _build_last_weight(ls_weight, layer_dim, num_elem):
#         zeros = tf.zeros([layer_dim, 1], dtype=tf.float32)
#         diag_weight = tf.concat([ls_weight[0:layer_dim, 0:1], tf.tile(zeros, [1, num_elem - 1])], axis=-1)
#         for i in range(1, num_elem - 1):
#             row = tf.concat([tf.tile(zeros, [1, i]), ls_weight[i * layer_dim:(i + 1) * layer_dim, 0:1],
#                              tf.tile(zeros, [1, num_elem - 1 - i])], axis=-1)
#             diag_weight = tf.concat([diag_weight, row], axis=0)
#         last_row = tf.concat([tf.tile(zeros, [1, num_elem - 1]), ls_weight[-layer_dim:, 0:1]], axis=-1)
#         diag_weight = tf.concat([diag_weight, last_row], axis=0)
#         return diag_weight
#
#     bias_coeff = _build_last_bias(model.num_quad, model.num_elements)
#     source_terms = model.source_term(tf.reshape(tf.transpose(model.x_diag_tf), [-1, 1]))
#     num_layers = len(weights) + 1
#     last_layer_dim = model.hidden_layers[-1]
#
#     x_sliced = tf.split(model.x_diag_tf, model.num_elements, axis=1)
#     result = tf.concat(x_sliced, axis=1)
#     for layer in range(0, num_layers - 2):
#         weight = weights[layer]
#         bias = biases[layer]
#         result = tf.tanh(tf.add(tf.matmul(result, weight), bias))
#
#     result_x = _compute_sliced_grad(result, x_sliced, last_layer_dim, model.num_elements)   # shape: (Q, PL)
#     result_xx = _compute_sliced_grad(result_x, x_sliced, last_layer_dim, model.num_elements)
#     x_temp = result_xx - result
#     x_coeff = tf.matmul(tf.transpose(x_temp), x_temp*model.guass_weights)
#     mat_coeff = tf.concat([x_coeff, bias_coeff], axis=-1)
#     new_wb = tf.linalg.lstsq(mat_coeff, source_terms)
#
#     last_weight = _build_last_weight(new_wb[0:-model.num_elements, 0:1], last_layer_dim, model.num_elements)
#
#     weights[-1].assign(last_weight)
#     biases[-1].assign(tf.transpose(new_wb[-model.num_elements:, 0:1]))
#
#     return weights, biases


def compute_least_square_single_list(model, weights: list, biases: list):
    def _compute_sliced_grad(y, x, last_dim):
        y_sliced = tf.split(y, last_dim, axis=1)
        grad_list = [tf.gradients(yl, x)[0] for yl in y_sliced]
        return tf.concat(grad_list, axis=1)

    num_layers = len(weights) + 1
    last_layer_dim = model.hidden_layers[-1]
    x_col = model.x_list_tf[0]
    result = model.x_list_tf[0]
    for layer in range(0, num_layers - 2):
        weight = weights[layer][0]
        bias = biases[layer][0]
        result = tf.tanh(tf.add(tf.matmul(result, weight), bias))

    quad_order = model.num_train
    result_x = _compute_sliced_grad(result, x_col, last_layer_dim)   # shape: (Q, L)
    result_xx = _compute_sliced_grad(result_x, x_col, last_layer_dim)
    x_temp = result_xx - result
    extra_col = -tf.ones([quad_order, 1], dtype=tf.float32)
    extra_row = tf.ones([1, quad_order], dtype=tf.float32)
    x0 = tf.concat([tf.transpose(x_temp), extra_row], axis=0)
    x1 = tf.concat([x_temp, extra_col], axis=1) * (model.guass_weights.reshape((-1, 1)))
    left_side = tf.matmul(x0, x1) + 2 * 1e-4 * tf.eye(last_layer_dim + 1)

    source_terms = tf.cast(model.loss_term(model.x_list[0]) * (model.guass_weights.reshape((-1, 1))), dtype=tf.float32)
    right_side = tf.matmul(x0, source_terms)

    # new_wb = tf.linalg.lstsq(left_side, right_side)
    new_wb = tf.linalg.solve(left_side, right_side)

    assign_bias_op = biases[-1][0].assign(new_wb[-1:, 0:1])
    assign_weights_op = weights[-1][0].assign(new_wb[0:-1, 0:1])

    return assign_bias_op, assign_weights_op, new_wb


def neural_net(x, weights, biases):
    num_layers = len(weights) + 1

    result = x
    for layer in range(0, num_layers - 2):
        weight = weights[layer]
        bias = biases[layer]
        result = tf.tanh(tf.add(tf.matmul(result, weight), bias))
    weight = weights[-1]    # last layer
    bias = biases[-1]
    result = tf.add(tf.matmul(result, weight), bias)
    return result


def neural_net_list(x: list, weights: list, biases: list) -> list:
    num_layers = len(weights) + 1
    num_elem = len(x)

    results = x.copy()
    for layer in range(0, num_layers - 2):
        for elem in range(num_elem):
            weight = weights[layer][elem]
            bias = biases[layer][elem]
            results[elem] = tf.tanh(tf.add(tf.matmul(results[elem], weight), bias))
    # last layer
    for elem in range(num_elem):
        weight = weights[-1][elem]
        bias = biases[-1][elem]
        results[elem] = tf.add(tf.matmul(results[elem], weight), bias)
    return results


def train(model, max_iter, logdir, steps=None, threshold=1e-7, ls_op=False):
    with open(logdir + "history.txt", 'w') as writer:
        if steps is None:
            steps = [4000, 10000, max_iter]
        else:
            steps.append(max_iter)
        learning_rates = [1e-3, 1e-3, 1e-4]
        epochs = []
        loss_records = []
        current_mode = 0
        learning_rate = learning_rates[current_mode]
        learning_rate_dict = {model.lr_tf: learning_rate}
        feed_dict = {**model.tf_dict, **learning_rate_dict}
        writer.writelines("\n--- Optimizer: lr = %.2e ---\n" % learning_rates[current_mode])
        loss_base = threshold
        adjusting = False
        start_time = time.time()
        for it in range(max_iter):
            # switch learning rate
            if it >= steps[current_mode]:
                current_mode += 1
                # check whether to switch to adjusting phase
                if current_mode == len(learning_rates) - 1:
                    adjusting = True
                    loss_base = int(np.log10(model.sess.run(model.loss, model.tf_dict)))
                learning_rate = learning_rates[current_mode]
                learning_rate_dict = {model.lr_tf: learning_rate}
                feed_dict = {**model.tf_dict, **learning_rate_dict}
                writer.writelines("\n--- SWITCHING ---\n--- Optimizer: lr = %.2e ---\n" % learning_rate)

            # train the model
            model.sess.run(model.train_op_Adam, feed_dict)

            # Checkpoints
            if it % 1 == 0:
                elapsed = time.time() - start_time
                epochs.append(it)
                loss_value, weights, biases = model.sess.run([model.loss, model.weights_list, model.biases_list], model.tf_dict)
                writer.writelines('It: %d, Loss: %.3e,  Time: %.2f, Progress: %.2f%%\n' %
                      (it, loss_value, elapsed, it * 100.0 / max_iter))
                writer.writelines("=== current wb ===\n")
                writer.writelines(str(weights[-1][0]) + '\n')
                writer.writelines(str(biases[-1][0]) + '\n')
                writer.writelines("======\n")
                # LS Optimization
                if ls_op:  # and it != 0 and it % 10 == 0:
                    writer.writelines("\n--- LS OPTIMIZING ---\n")
                    _, _, new_wb = model.sess.run([model.b_op, model.w_op, model.new_wb], model.tf_dict)
                    new_loss_value = model.sess.run(model.loss, model.tf_dict)
                    writer.writelines("Loss: %.3e, Loss Optimized: %.2f%%\n" %
                          (new_loss_value, (loss_value - new_loss_value) * 100.0 / loss_value))
                    loss_value = new_loss_value
                    writer.writelines("=== old ====\n")
                    writer.writelines(str(weights[-1][0]) + '\n')
                    writer.writelines(str(biases[-1][0]) + '\n')
                    writer.writelines("=== new ====\n")
                    writer.writelines(str(new_wb) + '\n')
                    writer.writelines("=======\n")
                log_loss = np.log10(loss_value)
                loss_records.append(log_loss)
                # check target loss
                if loss_value < threshold:
                    return epochs, loss_records
                # if loss becomes too small, learning rate should be adjusted
                if loss_base > int(log_loss) and adjusting:
                    loss_base = int(log_loss)
                    learning_rate /= 10
                    learning_rate_dict = {model.lr_tf: learning_rate}
                    feed_dict = {**model.tf_dict, **learning_rate_dict}
                    writer.writelines("\n>>> ADJUSTING <<<\n>>> Optimizer: lr = %.2e <<<\n" % learning_rate)
                start_time = time.time()
    return epochs, loss_records


def nn_predict(model, test):
    u_prediction = model.sess.run(model.u_bd_pred, {model.x_bd_tf: test[:, 0:1]})
    return u_prediction


def mnn_predict(model, num_per_element):
    # test = np.linspace(model.lower_bound, model.upper_bound, num_per_element*model.num_elements)\
    #     .reshape((model.num_elements, num_per_element, 1))
    test = np.linspace(model.lower_bound, model.upper_bound, num_per_element*model.num_elements)\
        .reshape((-1, num_per_element)).T
    prediction = model.sess.run(model.u_pred, {model.predict_input_tf: test})
    truth = model.solution(test)
    # input_flatten = test.reshape((-1, 1))
    # pred_flatten = prediction.reshape((-1, 1))
    # truth_flatten = truth.reshape((-1, 1))
    input_flatten = test.T.reshape((-1, 1))
    pred_flatten = prediction.T.reshape((-1, 1))
    truth_flatten = truth.T.reshape((-1, 1))
    return input_flatten, pred_flatten, truth_flatten


def mnn_predict_list(model, num_per_element):
    test = np.linspace(model.lower_bound, model.upper_bound, num_per_element*model.num_elements)\
        .reshape((-1, num_per_element)).T
    x_list = np.hsplit(test, model.num_elements)
    feed_dict = {k: v for k, v in zip(model.predict_input_list_tf, x_list)}
    prediction = model.sess.run(model.u_pred, feed_dict)
    truth = model.solution(test)
    input_flatten = test.T.reshape((-1, 1))
    truth_flatten = truth.T.reshape((-1, 1))
    pred_flatten = np.concatenate(prediction, axis=1).T.reshape((-1, 1))
    return input_flatten, pred_flatten, truth_flatten


def save_weights(model, path="./"):
    weights_ = model.sess.run(model.weights)
    biases_ = model.sess.run(model.biases)
    with open(path + "weights.txt", 'wb') as fp:
        pickle.dump(weights_, fp)
    with open(path + "biases.txt", 'wb') as fp:
        pickle.dump(biases_, fp)
