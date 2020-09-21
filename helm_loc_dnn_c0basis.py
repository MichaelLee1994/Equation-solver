"""
This implementation:
(1) basis function global C^0 continuous, modes on common element boundaries
    across neighvoring elements corresponds to one equation
(2) field function represented by local DNN within each element
    C^k continuity enforced across element boundaries for field function
        where k=-1, 0, 1, 2, 3, 4

solve Helmholtz equation in 1D by
spectral element methods with DNN, on domain [a,b]
multiple elements, on domain [a,b]
Spectral element Petrov-Galerkin method DNN
Using Sherin-Karniadakis basis functions

function represented by multiple local DNN, or a DNN with multiple inputs and multiple outputs
enforce weak form of equations with SK global C^0 basis, but function represented by local DNN
so the boundary mode on common boundary of two neighboring elements corresponds to one equation
enforce C^k continuity across element boundaries, where k=0, 1, 2, 3, 4, or -1
  when k=-1, do not enforce any continuity across element boundary for field function
local DNN representation for each element, enforce continuity across element boundary

need to extract data for element from global tensor for u and du/dx
use keras.backend.slice() or tensorflow.split() to do this

'adam' seems better than 'nadam' for this problem

another implementation:
  basis matrix dimension: [N_quad, N_modes]
  Weight matrix dimension: [N_elem, N_quad], etc
"""
import os
import sys
import numpy as np
import keras
# import keras.models as kmod
import keras.layers as klay
import keras.backend as K
import keras.callbacks as kclbk

import util
import polylib
import skbasis as skb

_epsilon = 1.0e-14

## ========================================= ##
root_dir = "./Test_Collection/"
result_dir = root_dir + "CB_keras/"
paras = util.load_parameter(root_dir + "paras.txt")
# domain contains coordinates of element boundaries
domain = util.to_list(paras["pivot"])

depth = int(sys.argv[2])
width = int(sys.argv[3])
max_epoch = int(sys.argv[4])

# domain parameters
N_elem = len(domain) - 1  # number of elements
upper_bound = domain[-1]
lower_bound = domain[0]

# C_k continuity
CK = 0  # C^k continuity

LAMBDA_COEFF = 1.0  # lambda constant
aa = 3.0

## ========================================= ##
# NN parameters
layers = [width] * depth
activations = ["tanh"] * depth
layers.append(1)
layers.insert(0, 1)
activations.append("linear")
activations.insert(0, "None")

# spectral methods parameters
N_modes = int(sys.argv[1])  # number of modes
N_quad = round(1.5 * N_modes)  # number of quadrature points

## ========================================= ##
# default, set to double precision
K.set_floatx('float64')
K.set_epsilon(_epsilon)


def the_anal_soln(x):
    # return x * np.cos(aa * x)
    return x * np.sin(x)
    # return np.cos(x)

## ============================ ##
# def the_anal_soln_deriv2(x):
#     # second derivative analytic solution
#     return -2.0 * aa * np.sin(aa * x) - aa * aa * x * np.cos(aa * x)
#     # return -2.0*np.sin(x)-x*np.cos(x) # deriv2 = -2sin(x)-x*cos(x)
#     # return -np.cos(x)


def the_source_term(lambda_coeff, x):
    # value = the_anal_soln_deriv2(x) - lambda_coeff * the_anal_soln(x)
    value = 2 * np.cos(x) - 2 * x * np.sin(x)
    return value


## ========================= ##
def calc_basis(id, z):
    """
    compute id-th basis function on given points z
    :param id: index of basis function
    :param z: points to evaluate on
    :return: values on points
    """
    # use Sherwin-Karniadakis basis
    return skb.calc_basis(id, z)


def calc_basis_deriv(id, z):
    """
    compute derivative of id-th basis fuction on given points
    :param id: index of basis function
    :param z: points to evaluate on
    :return: values on points
    """
    # use Sherwin-Karniadakis basis
    return skb.calc_basis_deriv(id, z)


def calc_zw(n_quads):
    """
    compute quadrature points and weights
    :param n_quads: number of qaudrature points
    :return:
    """
    z, w = polylib.zwgll(n_quads)  # zeros and weights of Gauss-Labatto-Legendre quadrature
    return (z, w)


def calc_jacobian(this_domain):
    """
    compute the Jacobian for each element
    :param this_domain: vector containing element boundary coefficients
    :return:
    """
    L = len(this_domain)
    jacob = np.zeros(L - 1)
    for i in range(L - 1):
        jacob[i] = (this_domain[i + 1] - this_domain[i]) * 0.5

    return jacob


def get_basis_mat(n_modes, n_quads):
    """
    compute matrix of basis functions and weight vector
    on Gauss-Lobatto-Legendre quadrature points
    :param n_modes: number of modes
    :param n_quads: number of quadrature points
    :return: tuple (B, w), where B[n_modes][n_quads] is the basis matrix on quadrature points
                           w[n_quads] is the vector of weights on the quadrature points
    """
    B = np.zeros((n_modes, n_quads))
    z, w = calc_zw(n_quads)  # polylib.zwgll(n_quads)
    # (z,w) contains zeros and weights of Lobatto-Legendre

    # compute basis matrix
    for i in range(n_modes):
        B[i, :] = calc_basis(i, z)  # polylib.legendref(z, i)
    # now B[n_modes,n_quads] contains matrix of basis functions

    return (B, w)


def get_basis_deriv_mat(n_modes, n_quads):
    """
    compute derivative matrix of basis functions on Gauss-Lobatto-Legendres quadrature points
    :param n_modes: number of modes
    :param n_quads: number of quadrature points
    :return: basis derivative matrix Bd[n_modes][n_quads] on quadrature points
    """
    Bd = np.zeros((n_modes, n_quads))
    z, _ = calc_zw(n_quads)  # polylib.zwgll(n_quads)

    for i in range(n_modes):
        Bd[i, :] = calc_basis_deriv(i, z)  # polylib.legendred(z, i)

    return Bd


def get_basis_boundary_mat(n_modes):
    """
    compute matrix of basis functions on boundaries
    :param n_modes: number of modes
    :return: values on boundaries 1, and -1
    """
    z = np.zeros(2)
    z[0] = -1.0
    z[1] = 1.0

    B_bound = np.zeros((n_modes, 2))
    for i in range(n_modes):
        B_bound[i, :] = calc_basis(i, z)

    return B_bound


def get_basis_info(n_modes, n_quads):
    """
    compute expansion basis data for each element
    :param n_modes: number of modes per element
    :param n_quads: number of quadrature points per element
    :return: tuple of basis data, (B, Bd, W, B_bound)
             where B : basis matrix on quadrature points, shape: (n_quads, n_modes)
                   Bd : basis derivative matrix on quadrature points, shape: (n_quads, n_modes)
                   W : weight matrix on quadrature points, shape: (1, n_quads)
                   B_bound : basis values on boundary x=-1, 1, shape: (2, n_modes)
                             B_bound[0, :] contains basis values on x=-1
                             B_bound[1, :] contains basis values on x=1
    """
    B, W = get_basis_mat(n_modes, n_quads)
    B_trans = np.transpose(B)  # B_trans shape: (n_quads, n_modes)
    # Now B contains matrix of bases on quadrature points, dimension [n_modes][n_quads]
    #     W contains vector of weights on quadrature points, dimension [n_quads]

    # basis function derivative matrix
    Bd = get_basis_deriv_mat(n_modes, n_quads)
    Bd_trans = np.transpose(Bd)  # shape: (n_quads, n_modes)
    # now Bd contains matrix of derivatives of bases on quadrature points, in standard element
    #     dimension [n_modes][n_quads]

    W_tmp = np.expand_dims(W, 0)  # shape: (1, n_quads)

    B_bound = get_basis_boundary_mat(n_modes)
    # now B_bound[n_modes][0:2] contains basis values on -1 and 1, in standard element
    B_bound_trans = np.transpose(B_bound)
    # now B_bound_trans has shape: (2, n_modes)

    return (B_trans, Bd_trans, W_tmp, B_bound_trans)


def one_elem_loss_generator(id_elem, elem_param, basis_info, model_info):
    """
    build loss function for one element
    :param id_elem: id of current element and sub-model
    :param elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : numpy array, shape: (2,2); bc[:, 0] contains coordinates of 2 boundaries
                                                             bc[:, 1] contains Dirivhlet BC values of 2 boundaries
                             jacobian: vector containing jacobians of each element, shape: (n_elem,)
                             domain: vector contains element boundary coordinates, shape: (n_elem+1,)
                             ck_k : k value in C_k continuity
    :param basis_info: tuple, (B, Bd, W, B_bound)
                       where B : basis matrix, shape: (n_quads, n_modes)
                             Bd : basis derivative matrix, shape: (n_quads, n_modes)
                             W : weight matrix, shape: (1, n_quads)
                             B_bound : matrix of basis values on x=-1, 1; shape: (2, n_modes)
    :param model_info: tuple, (model, sub_model_set)
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             sub_model_set: list of sub-models for each element, shape: (n_elem,)
    :return: loss function for the sub-model of this element

    if this is the first element, will compute equation residual of this element, and also of element-boundary
        C^k continuity conditions, as well as global boundary condition loss
    if this is not the first element, will compute only the equation residual of this element
    """
    n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k = elem_param
    n_elem = jacob.shape[0]  # number of elements

    global_model, sub_model_list = model_info  # global_model contains the overall model
    this_sub_model = sub_model_list[id_elem]  # sub_model for this element
    first_sub_model = sub_model_list[0]  # sub_model for first element
    last_sub_model = sub_model_list[n_elem - 1]  # sub_model for last element

    B_trans, Bd_trans, W_mat_A, B_bound = basis_info
    # B_trans contains basis function matrix, shape: (n_quads, n_modes)
    # Bd_trans contains basis derivative matrix, shape: (n_quads, n_modes)
    # W_mat_A contains weight matrix, shape: (1, n_quads)
    # B_bound contains boundary basis matrix, shape: (2, n_modes)

    W_mat = np.zeros((n_elem, n_quads))
    W_A = np.zeros((n_elem, n_quads))
    for i in range(n_elem):
        W_mat[i, :] = W_mat_A[0, :] * jacob[i]
        W_A[i, :] = W_mat_A[0, :]
    # now W_mat contains jacobian * W matrix, shape: (n_elem, n_quads)
    #     W_A contains W matrix, shape: (n_elem, n_quads)

    B_bound_trans = np.copy(B_bound)  # shape: (2, n_modes)
    B_bound_trans[0, :] = -B_bound_trans[0, :]
    # negate value at x=-1, because in weak form we have -phi(-1)*du/dx|_{x=-1}

    B_bound_left = np.zeros((n_elem, n_modes, 1))  # shape: (n_elem,n_quads,1)
    B_bound_right = np.zeros((n_elem, n_modes, 1))  # shape: (n_elem,n_quads,1)

    B_bound_left[0, :, 0] = B_bound_trans[0, :]  # modes of first element, left boundary x=a
    B_bound_right[-1, :, 0] = B_bound_trans[1, :]  # modes of right element, right boundary x=b

    ## ============================================##
    # generate Keras tensors from these matrices or vectors
    B_tensor = K.constant(B_trans)  # in standard element, shape: (n_quads, n_modes)
    W_mat_tensor = K.constant(W_mat)  # shape of W_mat_tensor: (n_elem, n_quads)
    W_mat_A_tensor = K.constant(W_A)  # shape : (n_elem, n_quads)
    Bd_tensor = K.constant(Bd_trans)  # in standard element, shape; (n_quads, n_modes)

    # B_bound_tensor = K.constant(B_bound_trans) # shape: (2, n_modes)
    B_bound_left_tensor = K.constant(B_bound_left)  # shape: (n_elem, n_modes, 1)
    B_bound_right_tensor = K.constant(B_bound_right)  # shape: (n_elem, n_modes, 1)

    # ======================================= #
    # element boundary du/dx
    # Zb = np.zeros((2, 1))
    # Zb[0, 0] = this_domain[id_elem]
    # Zb[1, 0] = this_domain[id_elem+1]
    # now Zb_elem contains coordinates of end points for this element

    # Zb_tensor = K.constant(Zb)
    # Zb_input_tensor = keras.Input(tensor = Zb_tensor)
    # now Zb_input_tensor contains input tensor for this sub-model

    # ======================================= #
    # boundary conditions for domain
    ZL = np.zeros((1, 1))
    ZL[0, 0] = bc[0, 0]  # left boundary coordinate
    ZL_tensor = K.constant(ZL)
    # convert ZL_tensor into Keras input tensor
    ZL_input_tensor = keras.Input(tensor=ZL_tensor)

    ZR = np.zeros((1, 1))
    ZR[0, 0] = bc[1, 0]  # right boundary coordinate
    ZR_tensor = K.constant(ZR)
    ZR_input_tensor = keras.Input(tensor=ZR_tensor)

    BCL = np.zeros((1, 1))
    BCL[0, 0] = bc[0, 1]  # left boundary condition
    BCL_tensor = K.constant(BCL)  # shape: (1,1)

    BCR = np.zeros((1, 1))
    BCR[0, 0] = bc[1, 1]  # right boundary condition
    BCR_tensor = K.constant(BCR)  # shape: (1,1)

    # ====================================== #
    # continuity across element boundaries
    Zb_elem = np.zeros((n_elem, 2, 1))
    for i in range(n_elem):
        Zb_elem[i, 0, 0] = this_domain[i]
        Zb_elem[i, 1, 0] = this_domain[i + 1]

    Zb_elem_tensor = []
    for i in range(n_elem):
        temp_tensor = K.constant(Zb_elem[i])  # shape (2,1)
        temp_input_tensor = keras.Input(tensor=temp_tensor)
        Zb_elem_tensor.append(temp_input_tensor)
    # now Zb_elem_tensor contains list of element-boundary coordinate tensors

    # element-boundary value difference
    bound_assembly = np.zeros((n_elem - 1, n_elem * 2))
    for i in range(n_elem - 1):
        bound_assembly[i, 2 * i + 1] = 1.0  # right boundary of i-th element, the one on the left
        bound_assembly[i, 2 * (i + 1)] = -1.0  # left boundary of (i+1)-th element, the one on the right

    bound_assembly_tensor = K.constant(bound_assembly)  # shape: (n_elem-1, n_elem*2)
    # to be used for computing different between values across element boundaries

    # ============================= #
    # element boundary mode assembly matrix and tensor
    bmode_assembler = np.zeros((n_elem + 1, 2 * n_elem))
    bmode_assembler[0, 0] = 1.0
    bmode_assembler[-1, -1] = 1.0
    for i in range(n_elem - 1):
        bmode_assembler[i + 1, 2 * i + 1] = 1.0  # i-th element, right boundary
        bmode_assembler[i + 1, 2 * (i + 1)] = 1.0  # (i+1)-th element, left boundary
    # now bmode_assembler contains assembler matrix
    bmode_assembler_tensor = K.constant(bmode_assembler)  # Keras tensor, shape: (n_elem+1, 2*n_elem)

    # ============================= #
    def loss_generator_Ck(ck):
        """
        Ck loss function generator
        :param ck: integer, >= 0, k value in C_k continuity
        :return: loss function for C_k
        """

        def loss_func_c0():
            """
            compute loss across element boundary for C^0 continuity across element boundaries
            :return: loss value associated with C0 continuity condition
            """
            # now Zb_elem_tensor contains a list of (2,1) element boundary coordinate tensors
            output_tensor = global_model(Zb_elem_tensor)
            # now output_tensor contains list of output tensors, shape: (n_elem, 2, 1)

            temp_out_tensor = K.concatenate(output_tensor, 0)  # concatenate list tensors
            # now temp_out_tensor has shape (n_elem*2, 1)

            C0_residual_tensor = K.dot(bound_assembly_tensor, temp_out_tensor)
            # C0_residual_tensor has shape: (n_elem-1, 1)

            this_loss = K.sum(K.square(C0_residual_tensor))
            return this_loss

        def loss_func_c1():
            """
            compute loss across element boundary for C^1 continuity across element boundaries
            :return: loss value
            """
            output_tensor = global_model(Zb_elem_tensor)
            # now output_tensor contains list of output tensors, shape: (n_elem, 2, 1)

            grad_dudx = K.gradients(output_tensor, Zb_elem_tensor)
            # now grad_dudx contains gradients,
            #     shape is (n_elem, 2, 1)
            #     grad_dudx is a list of (2,1) arrays

            C0_concat = K.concatenate(output_tensor, 0)  # shape (n_elem*2, 1)
            C1_concat = K.concatenate(grad_dudx, 0)  # shape (n_elem*2, 1)

            C0_residual = K.dot(bound_assembly_tensor, C0_concat)  # shape: (n_elem-1, 1)
            C1_residual = K.dot(bound_assembly_tensor, C1_concat)  # shape: (n_elem-1, 1)

            this_loss = K.sum(K.square(C0_residual)) + K.sum(K.square(C1_residual))
            return this_loss

        def loss_func_c2():
            """
            loss for C^2 continuity across element boundaries
            :return: loss value
            """
            ub = global_model(Zb_elem_tensor)  # ub constains element boundary value tensor, shape: (n_elem, 2, 1)
            dub_dx = K.gradients(ub, Zb_elem_tensor)  # dub_dx contains du/dx tensor on boundary, shape: (n_elem,2,1)
            dub_dx_2 = K.gradients(dub_dx, Zb_elem_tensor)
            # dud_dx_2 contains d^2u/dx^2 tensor on boundary, shape: (n_elem,2,1)
            # Note: tensorflow.gradients sums up all w.r.t. to list of output tensors
            #       but since only du1/dx1, du2/dx2, du3/dx3 ... is non-zero, while all du_i/dx_j=0 if i!=j
            #       dudx effectively contains only du_i/dx_i, for i=0, ..., n_elem-1
            #       similarly, for second derivatives, du_dx_2 contains only d^2 u_i/dx_i^2, for i=0,...,n_elem-1 only

            C0_concat = K.concatenate(ub, 0)  # shape: (n_elem*2, 1)
            C1_concat = K.concatenate(dub_dx, 0)  # shape: (n_elem*2, 1)
            C2_concat = K.concatenate(dub_dx_2, 0)  # shape: (n_elem*2, 1)

            C0_residual = K.dot(bound_assembly_tensor, C0_concat)  # shape: (n_elem-1, 1)
            C1_residual = K.dot(bound_assembly_tensor, C1_concat)  # shape: (n_elem-1, 1)
            C2_residual = K.dot(bound_assembly_tensor, C2_concat)  # shape: (n_elem-1, 1)

            this_loss = K.sum(K.square(C0_residual)) + K.sum(K.square(C1_residual)) + K.sum(K.square(C2_residual))
            return this_loss

        def loss_func_c3():
            """
            compute loss associated with C^3 continuity
            :return: loss value
            """
            ub = global_model(Zb_elem_tensor)  # ub constains element boundary value tensor, shape: (n_elem, 2, 1)
            dub_dx = K.gradients(ub, Zb_elem_tensor)  # dub_dx contains du/dx tensor on boundary, shape: (n_elem,2,1)
            dub_dx_2 = K.gradients(dub_dx, Zb_elem_tensor)  # second derivative tensor, shape: (n_elem,2,1)
            dub_dx_3 = K.gradients(dub_dx_2, Zb_elem_tensor)  # third derivative tensor, shape: (n_elem,2,1)

            C0_concat = K.concatenate(ub, 0)  # shape: (n_elem*2, 1)
            C1_concat = K.concatenate(dub_dx, 0)  # shape: (n_elem*2, 1)
            C2_concat = K.concatenate(dub_dx_2, 0)  # shape: (n_elem*2, 1)
            C3_concat = K.concatenate(dub_dx_3, 0)  # shape: (n_elem*2,1)

            C0_residual = K.dot(bound_assembly_tensor, C0_concat)  # shape: (n_elem-1, 1)
            C1_residual = K.dot(bound_assembly_tensor, C1_concat)  # shape: (n_elem-1, 1)
            C2_residual = K.dot(bound_assembly_tensor, C2_concat)  # shape: (n_elem-1, 1)
            C3_residual = K.dot(bound_assembly_tensor, C3_concat)  # shape: (n_elem-1, 1)

            this_loss = K.sum(K.square(C0_residual)) + K.sum(K.square(C1_residual)) \
                        + K.sum(K.square(C2_residual)) + K.sum(K.square(C3_residual))
            return this_loss

        def loss_func_c4():
            """
            compute loss associated with C^4 continuity
            :return: loss value
            """
            ub = global_model(Zb_elem_tensor)  # ub constains element boundary value tensor, shape: (n_elem, 2, 1)
            dub_dx = K.gradients(ub, Zb_elem_tensor)  # dub_dx contains du/dx tensor on boundary, shape: (n_elem,2,1)
            dub_dx_2 = K.gradients(dub_dx, Zb_elem_tensor)  # second derivative tensor, shape: (n_elem,2,1)
            dub_dx_3 = K.gradients(dub_dx_2, Zb_elem_tensor)  # third derivative tensor, shape: (n_elem,2,1)
            dub_dx_4 = K.gradients(dub_dx_3, Zb_elem_tensor)  # fourth derivative tensor, shape: (n_elem,2,1)

            C0_concat = K.concatenate(ub, 0)  # shape: (n_elem*2, 1)
            C1_concat = K.concatenate(dub_dx, 0)  # shape: (n_elem*2, 1)
            C2_concat = K.concatenate(dub_dx_2, 0)  # shape: (n_elem*2, 1)
            C3_concat = K.concatenate(dub_dx_3, 0)  # shape: (n_elem*2,1)
            C4_concat = K.concatenate(dub_dx_4, 0)  # shape: (n_elem*2, 1)

            C0_residual = K.dot(bound_assembly_tensor, C0_concat)  # shape: (n_elem-1, 1)
            C1_residual = K.dot(bound_assembly_tensor, C1_concat)  # shape: (n_elem-1, 1)
            C2_residual = K.dot(bound_assembly_tensor, C2_concat)  # shape: (n_elem-1, 1)
            C3_residual = K.dot(bound_assembly_tensor, C3_concat)  # shape: (n_elem-1, 1)
            C4_residual = K.dot(bound_assembly_tensor, C4_concat)  # shape: (n_elem-1, 1)

            this_loss = K.sum(K.square(C0_residual)) + K.sum(K.square(C1_residual)) \
                        + K.sum(K.square(C2_residual)) + K.sum(K.square(C3_residual)) \
                        + K.sum(K.square(C4_residual))
            return this_loss

        def loss_func_ck_default():
            return K.constant(0.0)

        # ++++++++++++ #
        if ck == 0:
            return loss_func_c0
        elif ck == 1:
            return loss_func_c1
        elif ck == 2:
            return loss_func_c2
        elif ck == 3:
            return loss_func_c3
        elif ck == 4:
            return loss_func_c4
        else:
            print("ERROR: loss_generator_ck() -- C^k continuity with (infinity > k > 4) is not implemented!\n")
            return loss_func_ck_default
        # +++++++++++++ #

    The_Loss_Func_Ck = loss_generator_Ck(ck_k)

    # now The_Loss_Func_Ck is the loss function for computing Ck continuity loss
    #     across element boundaries

    # ======================================= #
    def Equation_Residual(left_tensor_pair, right_tensor_pair):
        """
        actual computation of residual tensor for equation for all elements
        :param: left_tensor_pair: tuple, (BCL_in_tensor, BCL_out_tensor)
                                  where BCL_in_tensor is the input tensor for left domain boundary, shape: (1,1)
                                        BCL_out_tensor is the output tensor for left domain boundary, shape: (1,1)
        :param: right_tensor_pair: tuple, (BCR_in_tensor, BCR_out_tensor)
                                  where BCR_in_tensor is the input tensor for right domain boundary, shape: (1,1)
                                        BCR_out_tensor is the output tensor for right domain boundary, shape: (1,1)
        :return: equation residual tensor for all elements

        Note: when using tensorflow.gradients to compute derivative of a list of output tensors (u_i) with respect to
              a list of input tensors (x_j), the resultant gradient is a list with \sum_{i=0}^{N-1} du_i/dx_j, for
              j = 0, 1, ..., M-1. Note that it is summed over i. So the length of resultant list is equal to
              the length of the list of input tensors.
              In this code, since different sub-models are not connected, i.e. du_i/dx_j = 0 if i != j, the resultant
              sum will have no effect. So the resultant gradient is effectively du_i/dx_i, i=0, ..., M-1.
        """
        # now global_model contains entire model
        #     sub_model_list contains the list of sub-models
        #     first_sub_model contains the DNN for the first element
        #     last_sub_model contains the DNN for the last element

        # \sum du/dx * d_phi/dx*jacobian * w = \sum du/dx * d_phi/dxi * w
        #    where x is physical coordinate, xi is coordinate in standard element
        In_tensors = global_model.inputs  # list of input tensors, shape: (n_elem, n_quads, 1)
        Out_tensors = global_model.outputs  # list of output tensors, shape: (n_elem, n_quads, 1)

        dudx = K.gradients(Out_tensors, In_tensors)  # list of du/dx tensors, shape: (n_elem, n_quads, 1)
        dudx_con = K.concatenate(dudx, 0)  # shape: (n_elem*n_quads, 1)
        dudx_1 = K.reshape(dudx_con, (n_elem, n_quads))  # shape: (n_elem, n_quads)
        dudx_w = dudx_1 * W_mat_A_tensor  # element wise multiply, shape: (n_elem, n_quads)
        T1 = K.dot(dudx_w, Bd_tensor)  # shape: (n_elem, n_modes)
        # Note that Bd_tensor contains dphi/dxi in standard element, shape: (n_quads,n_modes)
        #           dphi/dx in physical space is dphi/dxi / jacobian
        # now T1[n_modes, n_elem] contains first term in weak form

        # lambda_coeff* \sum u*basis*jacobian * w
        # now Out_tensors contains u on all quadrature points of all elements, list of (n_quads,1) tensors
        # first concatenate them into a single tensor
        u_con = K.concatenate(Out_tensors, 0)  # shape: (n_elem*n_quads, 1)
        u_1 = K.reshape(u_con, (n_elem, n_quads))  # shape: (n_elem, n_quads)
        temp_v1 = lambda_coeff * u_1 * W_mat_tensor  # note: W_mat_tensor includes jacobian, shape: (n_elem,n_quads)
        T2 = K.dot(temp_v1, B_tensor)  # shape: (n_elem, n_modes)
        # now T2 contains second term, shape: (n_elem, n_modes)

        # source term \sum f(x) * basis * jacobian * w
        # Note: list of label tensors is contained in global_model.targets
        target_tensors = global_model.targets  # list of label tensors, shape: (n_elem, n_quads, ?)
        label_tensors = K.concatenate(target_tensors, 0)  # shape: (n_elem*n_quads, ?)
        y_true_0 = label_tensors[:, 0]  # shape: (n_elem*n_quads,)
        y_true_1 = K.reshape(y_true_0, (n_elem, n_quads))  # shape: (n_elem, n_quads)
        temp_v2 = y_true_1 * W_mat_tensor  # temp_v2 has shape (n_elem, n_quads)
        T3 = K.dot(temp_v2, B_tensor)  # shape: (n_elem, n_modes)
        # now T3 contains source term, shape (n_elem, n_modes)

        # boundary terms
        left_in_tensor, left_out_tensor = left_tensor_pair
        right_in_tensor, right_out_tensor = right_tensor_pair

        dudx_a = K.gradients(left_out_tensor, left_in_tensor)[0]  # gradient on left boundary, shape: (1,1)
        dudx_b = K.gradients(right_out_tensor, right_in_tensor)[0]  # gradient on right boundary, shape: (1,1)
        # Note: returned data from tensorflow.gradients is a list of tensors, even though the length
        #       of the list may be 1. So we need the "...[0]" in the above to get the actual tensor

        T_bound_left_1 = K.dot(B_bound_left_tensor, dudx_a)  # shape: (n_elem,n_modes,1)
        T_bound_left = K.squeeze(T_bound_left_1, -1)  # shape: (n_elem, n_modes)

        T_bound_right_1 = K.dot(B_bound_right_tensor, dudx_b)  # shape: (n_elem, n_modes, 1)
        T_bound_right = K.squeeze(T_bound_right_1, -1)  # shape: (n_elem, n_modes)

        T_bound_elem = T_bound_left + T_bound_right  # shape: (n_elem, n_modes)
        # now T_bound_elem contains contribution of boundary terms
        #                  -dudx(a)*phi(a) + dudx(b)*phi(b)

        # loss
        T_tot = T1 + T2 + T3 - T_bound_elem  # now T_tot has shape: (n_elem, n_modes)
        return T_tot

    def The_Loss_Func(y_true, y_pred):
        """
        actual computation of loss function for ordinary element, not the first element
        only compute the residual loss of equation for current element
        :param y_true: label data
        :param y_pred: preduction data
        :return: loss value
        """
        # DNN corresponding to other elements will simply return 0 for loss
        # DNN corresponding to first element will compute the loss for all elements
        this_loss = K.constant(0.0)
        return this_loss

    def The_First_Loss_Func(y_true, y_pred):
        """
        actual computation of loss for all elements
        compute residual loss of equation for all elements, and loss in continuity across element
           boundary, and loss in boundary condition of domain
        :param y_true:
        :param y_pred:
        :return:
        """
        # ========================== #
        # domain boundary condition
        BCL_out = first_sub_model(ZL_input_tensor)  # shape: (1,1)
        T_bc_L = BCL_out - BCL_tensor  # shape: (1,1)

        BCR_out = last_sub_model(ZR_input_tensor)  # shape: (1,1)
        T_bc_R = BCR_out - BCR_tensor  # shape: (1,1)

        # ========================= #
        # equation residual
        bcl_pair = (ZL_input_tensor, BCL_out)
        bcr_pair = (ZR_input_tensor, BCR_out)

        T_tot = Equation_Residual(bcl_pair, bcr_pair)
        # now T_tot contains residual tensor for equation of all elements, shape: (n_elem, n_modes)

        # ========================== #
        # extract element boundary and interior modes
        # assemble boundary modes
        bmode = K.slice(T_tot, (0, 0), (n_elem, 2))
        # bmode contains boundary mode contributions, shape: (n_elem, 2)

        interior_mode = K.slice(T_tot, (0, 2), (n_elem, n_modes - 2))
        # interior_mode contains interior mode contributions, shape: (n_elem, n_modes-2)

        bmode_2 = K.reshape(bmode, (n_elem * 2, 1))  # shape (n_elem*2,1)
        bmode_glob = K.dot(bmode_assembler_tensor, bmode_2)
        # now bmode_glob contains global mode contributions, shape: (n_elem+1,1)
        #     interior_mode contains interior mode contributions

        value_1 = K.flatten(bmode_glob)
        value_2 = K.flatten(interior_mode)
        value = K.concatenate((value_1, value_2))
        # now value contains all global modes, shape: (n_elem+1+n_elem*(n_modes-2),), 1D array

        # ========================= #
        # C^k continuity across element boundary
        Ck_loss = The_Loss_Func_Ck()
        # now Ck_loss contains loss value for C_k continuity across element boundaries

        # ========================= #
        this_loss = K.mean(K.square(value)) \
                    + (K.sum(K.square(T_bc_L)) + K.sum(K.square(T_bc_R)) + Ck_loss)
        return this_loss

    # ====================================== #
    if id_elem == 0:
        return The_First_Loss_Func
    else:
        return The_Loss_Func
    # ====================================== #


def multi_elem_loss_generator(elem_param, basis_info, model_info):
    """
    build list of loss functions
    :param elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : numpy array, shape: (2,2); bc[:, 0] contains coordinates of 2 boundaries
                                                             bc[:, 1] contains Dirivhlet BC values of 2 boundaries
                             jacobian: vector containing jacobians of each element, shape: (n_elem,)
                             domain: vector contains element boundary coordinates, shape: (n_elem+1,)
                             ck_k : k value in C_k continuity
    :param basis_info: tuple, (B, Bd, W, B_bound)
                       where B : basis matrix, shape: (n_quads, n_modes)
                             Bd : basis derivative matrix, shape: (n_quads, n_modes)
                             W : weight matrix, shape: (1, n_quads)
                             B_bound : matrix of basis values on x=-1, 1; shape: (2, n_modes)
    :param model_info: tuple, (model, sub_model_set)
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             sub_model_set: list of sub-models for each element, shape: (n_elem,)
    :return: list of loss functions for the multi-element model
    """
    _, _, _, _, jacob, _, _ = elem_param
    n_elem = jacob.shape[0]

    loss_func_set = []
    loss_weights = []
    for ix_elem in range(n_elem):
        loss_one = one_elem_loss_generator(ix_elem, elem_param, basis_info, model_info)
        loss_func_set.append(loss_one)
        loss_weights.append(0.0)
    loss_weights[0] = 1.0  # set loss weight for first element to 1.0, other elements to 0.0
    # now loss_func_set contains the list of loss functions
    #     loss_weights contains the list of loss weights

    return (loss_func_set, loss_weights)


def build_one_element_model(layers, activ):
    """
    build model for 1 element
    :param layers: vectors containing number of nodes for each layer
    :param activ: vectors containing the activation functions for
    :return: model for one element
    """
    this_model = keras.models.Sequential()

    N_layer = len(layers) - 1
    if N_layer < 1:
        print("ERROR: number of layers is zero or less!\n")

    # first layer
    ix_lay = 0
    this_model.add(klay.Dense(layers[ix_lay + 1], activation=activ[ix_lay + 1],
                              input_shape=(layers[ix_lay],)))
    ix_lay = ix_lay + 1

    for i in range(N_layer - 1):
        this_model.add(klay.Dense(layers[ix_lay + i + 1], activation=activations[ix_lay + i + 1]))

    return this_model


def build_multi_element_model(N_elem, layers, activ):
    """
    build model for multiple elements
    :param N_elem: number of elements, >=1
    :param layers: layers vector in each one-element model
    :param activ: activation vector in each one-element model
    :return: multi-element model
    """
    model_set = []
    input_set = []
    output_set = []
    for ix_elem in range(N_elem):
        elem_model = build_one_element_model(layers, activ)
        model_set.append(elem_model)
        input_set.append(elem_model.input)
        output_set.append(elem_model.output)
    # now model_set contains the list of element models
    #     input_set contains the list of input tensors
    #     output_set contains the list output tensors

    this_model = keras.Model(inputs=input_set, outputs=output_set)
    # now this_model contains the multi-input multi-output model

    return (this_model, model_set)


def build_model(layers, activ, n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k):
    n_elem = jacob.shape[0]
    my_model, sub_model_set = build_multi_element_model(n_elem, layers, activ)
    # now my_model contains the multi-element DNN model

    basis_info = get_basis_info(n_modes, n_quads)  # basis data
    elem_param = (n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k)
    model_info = (my_model, sub_model_set)
    loss_func_list, loss_weight_list = multi_elem_loss_generator(elem_param, basis_info, model_info)
    # now loss_func_list contains list of loss functions
    #     loss_weight_list contains list of loss weights

    # compile model
    my_model.compile(optimizer='adam',
                     loss=loss_func_list, loss_weights=loss_weight_list,
                     metrics=['accuracy'])
    return my_model


def get_learning_rate(model):
    lr = K.get_value(model.optimizer.lr)
    return float(lr)


def set_learning_rate(model, LR):
    K.set_value(model.optimizer.lr, LR)
    # return get_learning_rate(model)


def output_data(file, z_coord, v_pred, v_true):
    dim = len(z_coord)
    with open(file, 'w') as fileout:
        fileout.write("x,value-predict,value-true,error\n")
        for line in range(dim):
            fileout.write("%.14e,%.14e,%.14e,%.14e\n" % (z_coord[line], v_pred[line], v_true[line],
                                                         np.abs(v_pred[line] - v_true[line])))


if __name__ == '__main__':

    max_epochs = 100000
    new_dir = "{}_{}_{}/N={}/{}_{}({})/".format(lower_bound, upper_bound, N_elem, N_modes, depth, width, max_epoch)
    dump_dir = result_dir + new_dir
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    # files
    solution_file = dump_dir + 'spectral_dnn_soln.csv'
    model_weight_file = dump_dir + 'spectral_dnn_weights.hd5'
    model_init_weight_file = dump_dir + 'spectral_dnn_init_weights.hd5'

    Ck_cont = CK
    Lambda_Coeff = LAMBDA_COEFF  # np.float(2.0)

    # number of entries in input data: n_quads*n_elem
    N_input = N_quad * N_elem
    batch_size = N_input

    early_stop_patience = 1000

    bc_data = np.zeros((2, 2))
    bc_data[0, 0] = domain[0]  # coordinate of left boundary
    bc_data[0, 1] = the_anal_soln(bc_data[0, 0])  # boundary value of left boundary
    bc_data[1, 0] = domain[-1]  # coordinate of right boundary
    bc_data[1, 1] = the_anal_soln(bc_data[1, 0])

    jacobian = calc_jacobian(domain)
    # now jacobian contains the vector of jacobians

    the_DNN = build_model(layers, activations,
                          N_modes, N_quad, Lambda_Coeff, bc_data, jacobian, domain, Ck_cont)

    default_lr = get_learning_rate(the_DNN)
    print("Default learning rate: %f" % default_lr)

    In_data = []
    Label_data = []
    zz, _ = calc_zw(N_quad)  # polylib.zwgll(N_input)

    for i in range(N_elem):
        A = jacobian[i]
        B = (domain[i] + domain[i + 1]) * 0.5

        tmp_in_data = np.zeros((N_quad, 1))
        tmp_in_data[:, 0] = zz * A + B
        In_data.append(tmp_in_data)

        tmp_label = the_source_term(Lambda_Coeff, tmp_in_data)
        Label_data.append(tmp_label)

    from timeit import default_timer as timer

    begin_time = timer()

    early_stop = kclbk.EarlyStopping(monitor='loss', mode='min',
                                     verbose=1,
                                     patience=early_stop_patience,
                                     restore_best_weights=True)
    history = the_DNN.fit(In_data, Label_data,
                          epochs=max_epochs, batch_size=batch_size, shuffle=False,
                          callbacks=[early_stop])

    end_time = timer()
    print("Training time elapsed (seconds): %.14e" % (end_time - begin_time))

    # save weights to file
    the_DNN.save_weights(model_weight_file)
    # my_util.backup_file(model_init_weight_file)
    # my_util.copy_file(model_weight_file, model_init_weight_file)

    # prediction
    N_pred = 100
    data_in = []
    for i in range(N_elem):
        tmp_data = np.zeros((N_pred, 1))
        tmp_data[:, 0] = np.linspace(domain[i], domain[i + 1], N_pred)
        data_in.append(tmp_data)

    Soln = the_DNN.predict(data_in)

    A_in_data = np.concatenate(data_in, 0)  # shape: (N_elem*N_pred, 1)
    pred_in_data = np.reshape(A_in_data, (N_elem * N_pred,))
    A_soln = np.concatenate(Soln, 0)  # shape: (N_elem*N_pred, 1)
    Soln_flat = np.reshape(A_soln, (N_elem * N_pred,))

    Exact_soln = the_anal_soln(A_in_data)
    Exact_soln_flat = np.reshape(Exact_soln, (N_elem * N_pred,))

    error_u = np.sqrt(np.square(Soln_flat - Exact_soln_flat).mean())

    loss_history = np.log10(history.history['loss'])
    epoch = range(len(loss_history))

    with open(dump_dir + "paras.txt", 'w') as writer:
        writer.writelines("N={}\ndepth={}\nwidth={}\nc={}\n".format(N_elem, depth, width, Ck_cont))
        writer.writelines("max_epoch={}\nnum_tested_per_element={}\n".format(max_epoch, N_pred))
        writer.writelines("%.2f\t" % point for point in domain)

    output_data(solution_file, pred_in_data, Soln_flat, Exact_soln_flat)
    util.plot_result(epoch, loss_history, pred_in_data, Exact_soln_flat, Soln_flat, dump_dir, error_u)
