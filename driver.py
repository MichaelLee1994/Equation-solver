import os
import time
import sys
import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf

import util
import NNUtil
from DG_c0 import DGC0Solution
from CG_ck import CGckSolution
from DG import DGSolution
from CG_cinf import CGcinfSolution
from CB import CBSolutionNN
from Direct_loc import DirectLocSolution


def solution(x):
    return x*np.sin(x)


def source_term(x):
    return 2 * tf.math.cos(x) - 2 * x * tf.math.sin(x)


equation_info = (solution, source_term)
methods = ["CB", "CG_cinf", "CG_ck", "DG", "DG_c0", "Direct"]

if __name__ == "__main__":
    method = sys.argv[1]
    root_dir = "./Test_Collection/"
    result_dir = root_dir + "{}/".format(method)
    paras = util.load_parameter(root_dir + "paras.txt")
    N = int(sys.argv[2])
    depth = int(sys.argv[3])
    width = int(sys.argv[4])
    max_epoch = int(sys.argv[5])
    message = sys.argv[6]
    num_tested_per_element = int(paras["num_tested_per_element"])
    pivot = util.to_list(paras["pivot"])
    c = 0
    if method == "CG_ck":
        c = int(sys.argv[7])

    upper_bound = pivot[-1]
    lower_bound = pivot[0]
    num_element = len(pivot) - 1
    hidden_layers = [width] * depth

    if method == "Direct":
        new_dir = "{}_{}_{}/Q={}/{}_{}({})/".format(lower_bound, upper_bound, num_element, N, depth, width, max_epoch)
    else:
        new_dir = "{}_{}_{}/N={}/{}_{}({})/".format(lower_bound, upper_bound, num_element, N, depth, width, max_epoch)
    if method == "CG_ck":
        dump_dir = result_dir + "c{}/".format(c) + new_dir
    else:
        dump_dir = result_dir + new_dir
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    with open(dump_dir + "paras.txt", 'w') as writer:
        if method == "Direct":
            writer.writelines("Q={}".format(N))
        else:
            writer.writelines("N={}".format(N))
        writer.writelines("\ndepth={}\nwidth={}\nc={}\n".format(depth, width, c))
        writer.writelines("max_epoch={}\nnum_tested_per_element={}\n".format(max_epoch, num_tested_per_element))
        writer.writelines("%.2f\t" % point for point in pivot)

    if method == "CB":
        model = CBSolutionNN(N, pivot, hidden_layers, equation_info)
    elif method == "CG_cinf":
        layers = hidden_layers
        layers.append(1)
        layers.insert(0, 1)
        model = CGcinfSolution(N, pivot, layers, equation_info)
    elif method == "CG_ck":
        model = CGckSolution(N, c, pivot, hidden_layers, equation_info)
    elif method == "DG":
        model = DGSolution(N, pivot, hidden_layers, equation_info)
    elif method == "Direct":
        model = DirectLocSolution(N, pivot, hidden_layers, equation_info)
    else:
        model = DGC0Solution(N, pivot, hidden_layers, equation_info)

    start_time = time.time()
    epoch, loss_history = NNUtil.train(model, max_epoch)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    if method == "CG_cinf":
        x_test = np.linspace(lower_bound, upper_bound, num_tested_per_element * num_element).reshape(-1, 1)
        u_truth = solution(x_test)
        u_pred = NNUtil.nn_predict(model, x_test)
    else:
        x_test, u_pred, u_truth = NNUtil.mnn_predict(model, num_tested_per_element)

    error_u = np.sqrt(np.square(u_pred - u_truth).mean())
    print('Error u: %e' % error_u)

    dump_data = np.hstack((u_truth, u_pred))
    np.savetxt(dump_dir + 'result.csv', X=dump_data, header="truth,prediction", delimiter=',')
    dump_history = np.hstack((np.asarray(epoch).reshape((-1, 1)), np.asarray(loss_history).reshape((-1, 1))))
    np.savetxt(dump_dir + 'history.csv', X=dump_history, header="epochs,log10(loss)", delimiter=',')
    NNUtil.save_weights(model, dump_dir)

    util.plot_result(epoch, loss_history, x_test, u_truth, u_pred, dump_dir, error_u, True)
