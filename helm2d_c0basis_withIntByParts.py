"""
05/07/2020, to implement local-DNN/SEM solver for 2D Helmhiltz equation,
            with integration by parts and global C^0 basis for test functions.
            local representation for unknown field function, enforce C^k
            continuity across element boundaries.

            this has been implemented. results seem good

05/04/2020, to implement local-DNN/SEM to solve 2D Helmholtz equation,
            with another formulation, i.e. with integration by parts
            to remove second-order derivative. but still local basis to
            each element. enforce C^1 continuity for field function.

            this has been implemented. results are good

05/01/2020, to implement local-DNN/SEM methods for solving 2D Helmholtz equation
            rectangular domain, regular rectangular elements. each element
            represented by a DNN; formulation with no integration by parts.
            test function basis are local basis to each element; enforce C^1
            continuity for unknown function across element boundaries.
            assume same element order and same number of quadrature points
            in x and y directions

            this has been implemented. result seems correct. not fully tested yet

04/29/2020, to implement another formulation for solving Helmholtz equation,
            with no integration by parts, i.e.
                \int u_xx * phi_p dx - lambda* \int u*phi_p - \int f*phi_p = 0
            and enforcing C^k continuity acorss element boundaries
            enforce equations in local elements, local basis

            this is implemented. need to add a lambda layer right after input layer
            to scale the input data to [-1,1]. without the lambda layer,
            result is bad, and training does not converge. with the lambda layer,
            result is very good. so this formulation also works. need C^1 continuity
            across element boundaries.

In this implementation:
(1) Sherwin-Karniadakis basis within each element.
    enforce equation within each element. different modes in different elements
        correspond to different equations
(2) Field function represented by local DNN within each element
    enforce C^k continuity across element boundaries for field function, where k=0,1,2,3,4, or -1

solve Helmholtz equation in 1D by
spectral element methods with DNN, on domain [a,b]
multiple elements, on domain [a,b]
Spectral element Petrov-Galerkin method DNN
Using Sherin-Karniadakis basis functions, within each element

function represented by multiple local DNN, or a DNN with multiple inputs and multiple outputs
enforce weak form of equations within each element in terms of polynomials (SK-basis)
only enforce equation within each element,
different modes in different elements correspond to different equations
enforce C^k continuity across element boundaries for field function, where k=0, 1, 2, 3, 4, or -1
    when k=-1, do not enforce any continuity for field function across element boundaries
local DNN representation for each element, enforce certain continuity across element boundary

need to extract data for element from global tensor for u and du/dx
use keras.backend.slice() or tensorflow.split() to do this

'adam' seems better than 'nadam' for this problem

another implementation:
  basis matrix dimension: [N_quad, N_modes]
  Weight matrix dimension: [N_elem, N_quad], etc
"""

import numpy as np
import keras
import keras.layers as klay
import keras.backend as K
import keras.callbacks as kclbk
import tensorflow as tf
import pandas as pd

import polylib
import skbasis as skb
import my_util

_epsilon = 1.0e-14

## ========================================= ##
# domain contains coordinates of element boundaries
domain_x = [ 0.0, 1.5, 3.0, 4.0 ]
domain_y = [ 0.0, 1.5, 3.0, 4.0 ]

# domain parameters
Nx_elem = len(domain_x) - 1  # number of elements in x direction
Ny_elem = len(domain_y) - 1  # number of elements in y direction
N_elem = Nx_elem * Ny_elem   # total number of elements in domain

# C_k continuity
CK = 0  # C^k continuity

## ========================================= ##
# NN parameters
layers = [ 2, 10, 10, 10, 1 ]
activations = [ 'None',  'tanh', 'tanh', 'tanh', 'linear']

# spectral methods parameters
N_modes = 20 # number of modes
N_quad = N_modes + 3 # number of quadrature points

## ==========================================##
LAMBDA_COEFF = 10.0   # lambda constant
aa = 3.0

to_read_init_weight = True # if True, will read in initial weight
MAX_EPOCHS = 20000
LR_factor = 1.0 # learning rate factor, LR = LR_factor*default_lr

## ========================================= ##
# default, set to double precision
K.set_floatx('float64')
K.set_epsilon(_epsilon)

## =========================== ##
def anal_g(x):
    return x * np.cos(aa * x)

def anal_g_deriv2(x):
    return -2.0 * aa * np.sin(aa * x) - aa * aa * x * np.cos(aa * x)

def the_anal_soln(coord):
    """
    anal_soln = g(x)*g(y)
    :param coord: shape: (?,2). [:,0] -- x; [:,1] -- y
    :return: analytic solution
    """
    x = coord[:,0:1]
    y = coord[:,1:]
    gx = anal_g(x)
    gy = anal_g(y)
    return gx*gy

def the_anal_soln_laplace(coord):
    """
    anal_soln = g(x)*g(y)
    laplace = g''(x)*g(y) + g(x)*g''(y)
    :param coord: shape: (?,2). [:,0] -- x; [:,1] -- y
    :return: laplace of analitic solution
    """
    x = coord[:, 0:1]
    y = coord[:, 1:]
    gx = anal_g(x)
    gy = anal_g(y)
    g2x = anal_g_deriv2(x)
    g2y = anal_g_deriv2(y)
    return gx*g2y+g2x*gy

def the_source_term(lambda_coeff, coord):
    lap = the_anal_soln_laplace(coord)
    soln = the_anal_soln(coord)
    value = lap - lambda_coeff * soln
    return value

## ===================================== ##
def calc_basis (id, z):
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

def calc_zw (n_quads):
    """
    compute quadrature points and weights
    :param n_quads: number of qaudrature points
    :return:
    """
    z, w = polylib.zwgll(n_quads) # zeros and weights of Gauss-Labatto-Legendre quadrature
    return (z,w)

def calc_jacobian(this_domain):
    """
    compute the Jacobian for each element
    :param this_domain: tuple, (domain_x,domain_y), containing element boundary coordinates
    :return: tuple, (jacobian_x, jacobian_y), jacobian in x and y directions,
                    each is a numpy array of shape (nx_elem,2) and (ny_elem,2)
    """
    x_domain, y_domain = this_domain
    nx_elem = len(x_domain) -1
    ny_elem = len(y_domain) - 1

    jacob_x = np.zeros((nx_elem,2))
    for i in range(nx_elem):
        jacob_x[i,0] = (x_domain[i+1]+x_domain[i])*0.5
        jacob_x[i,1] = (x_domain[i+1]-x_domain[i])*0.5

    jacob_y = np.zeros((ny_elem,2))
    for i in range(ny_elem):
        jacob_y[i,0] = (y_domain[i+1] + y_domain[i])*0.5
        jacob_y[i,1] = (y_domain[i+1] - y_domain[i])*0.5

    return (jacob_x, jacob_y)

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
    z, w = calc_zw(n_quads) #polylib.zwgll(n_quads)
    # (z,w) contains zeros and weights of Lobatto-Legendre

    # compute basis matrix
    for i in range(n_modes):
        B[i, :] = calc_basis(i, z) #polylib.legendref(z, i)
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
    z, _ = calc_zw (n_quads) #polylib.zwgll(n_quads)

    for i in range(n_modes):
        Bd[i, :] = calc_basis_deriv(i, z) # polylib.legendred(z, i)

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

    W_tmp = np.expand_dims(W, 0) # shape: (1, n_quads)

    B_bound = get_basis_boundary_mat(n_modes)
    # now B_bound[n_modes][0:2] contains basis values on -1 and 1, in standard element
    B_bound_trans = np.transpose(B_bound)
    # now B_bound_trans has shape: (2, n_modes)
    B_bound_left = B_bound_trans[0:1,:] # shape: (1,n_modes)
    B_bound_right = B_bound_trans[1:,:] # shape: (1,n_modes)

    # create tensors
    B_tensor = K.constant(B_trans) # basis tensor, shape: (n_quads,n_modes)
    Bd_tensor = K.constant(Bd_trans) # derivative basis tensor, shape: (n_quads,n_modes)
    W_tensor = K.constant(W_tmp) # weight tensor, shape: (1,n_quads)
    B_bound_left_tensor = K.constant(B_bound_left) # shape: (1,n_modes)
    B_bound_right_tensor = K.constant(B_bound_right) # shape: (1,n_modes)

    return (B_tensor, Bd_tensor, W_tensor, (B_bound_left_tensor, B_bound_right_tensor))

def gen_elem_boundary_input(jacob, n_quads):
    """
    generate input data for element boundaries
    :param jacob: tuple, (jacob_x,jacob_y),
                    jacob_x: shape (nx_elem,2), [:,0] -- Jx_0e, [:,1] -- Jx_1e
                    jacob_y: shape (ny_elem,2), [:,0] -- Jy_0e, [:,1] -- Jy_1e
    :param n_quads: number of quadrature points
    :return: list of (nx_elem*ny_elem) input tensors of shape (4*n_quads,2)
    """
    jacob_x, jacob_y = jacob
    nx_elem = jacob_x.shape[0]
    ny_elem = jacob_y.shape[0]
    zz, _ = calc_zw(n_quads)

    input_list = []
    for ix_elem in range(nx_elem):
        for iy_elem in range(ny_elem):
            temp_coord = np.zeros((4*n_quads,2))

            # left boundary of this element
            temp_coord[0:n_quads, 0] = jacob_x[ix_elem,0] - jacob_x[ix_elem,1] # x coordinate
            temp_coord[0:n_quads, 1] = jacob_y[iy_elem,0] + jacob_y[iy_elem,1]*zz # y coordinate

            # right boundary of this element
            temp_coord[n_quads:2*n_quads, 0] = jacob_x[ix_elem,0] + jacob_x[ix_elem,1] # x coordinate
            temp_coord[n_quads:2*n_quads, 1] = temp_coord[0:n_quads, 1] # y coordinate

            # bottom boundary of this element
            temp_coord[2*n_quads:3*n_quads,0] = jacob_x[ix_elem,0] + jacob_x[ix_elem,1]*zz # x coordinates
            temp_coord[2*n_quads:3*n_quads,1] = jacob_y[iy_elem,0] - jacob_y[iy_elem,1] # y coordinate

            # top boundary of this element
            temp_coord[3*n_quads:4*n_quads,0] = temp_coord[2*n_quads:3*n_quads,0] # x coordinate
            temp_coord[3*n_quads:4*n_quads,1] = jacob_y[iy_elem,0] + jacob_y[iy_elem,1] # y coordinate

            temp_tensor = K.constant(temp_coord)
            input_list.append(temp_tensor)

    return input_list

def gen_one_elem_boundary_input(id_elem, jacob, n_quads):
    """
    generate element boundary input data for one element
    :param id_elem: tuple, (ix_elem,iy_elem), id of element
    :param jacob: tuple, (jacob_x,jacob_y)
    :param n_quads: number of quadrature points in one direction
    :return: tensor containing boundary input data
    """
    ix_elem, iy_elem = id_elem
    jacob_x, jacob_y = jacob
    zz, _ = calc_zw(n_quads)

    temp_coord = np.zeros((4 * n_quads, 2))

    # left boundary of this element
    temp_coord[0:n_quads, 0] = jacob_x[ix_elem, 0] - jacob_x[ix_elem, 1]  # x coordinate
    temp_coord[0:n_quads, 1] = jacob_y[iy_elem, 0] + jacob_y[iy_elem, 1] * zz  # y coordinate

    # right boundary of this element
    temp_coord[n_quads:2 * n_quads, 0] = jacob_x[ix_elem, 0] + jacob_x[ix_elem, 1]  # x coordinate
    temp_coord[n_quads:2 * n_quads, 1] = temp_coord[0:n_quads, 1]  # y coordinate

    # bottom boundary of this element
    temp_coord[2 * n_quads:3 * n_quads, 0] = jacob_x[ix_elem, 0] + jacob_x[ix_elem, 1] * zz  # x coordinates
    temp_coord[2 * n_quads:3 * n_quads, 1] = jacob_y[iy_elem, 0] - jacob_y[iy_elem, 1]  # y coordinate

    # top boundary of this element
    temp_coord[3 * n_quads:4 * n_quads, 0] = temp_coord[2 * n_quads:3 * n_quads, 0]  # x coordinate
    temp_coord[3 * n_quads:4 * n_quads, 1] = jacob_y[iy_elem, 0] + jacob_y[iy_elem, 1]  # y coordinate

    temp_tensor = K.constant(temp_coord) # shape: (4*n_quads,2)
    return temp_tensor


def gen_elem_boundary_jump(jacob):
    """
    generate matrix/tensor for computing differences in function value on common element boundaries
    :param jacob: tuple, (jacob_x,jacob_y)
    :return: tuple, (bjump_x, bjump_y), tensors for computing jumps in x and y directions
                    bjump_x : shape (nx_elem-1,2*nx_elem)
                    bjump_y : shape (ny_elem-1,2*ny_elem)
    """
    jacob_x, jacob_y = jacob
    nx_elem = jacob_x.shape[0]
    ny_elem = jacob_y.shape[0]

    bjump_x = np.zeros((nx_elem-1, 2*nx_elem))
    for i in range(nx_elem-1):
        bjump_x[i, 2*i+1] = 1.0 # right boundary of element on the left
        bjump_x[i, 2*(i+1)] = -1.0 # left boundary of element on the right
    bjump_x_tensor = K.constant(bjump_x)

    bjump_y = np.zeros((ny_elem-1, 2*ny_elem))
    for i in range(ny_elem-1):
        bjump_y[i, 2*i+1] = 1.0 # top boundary of element below
        bjump_y[i, 2*(i+1)] = -1.0 # bottom boundary of element above
    bjump_y_tensor = K.constant(bjump_y)

    return (bjump_x_tensor, bjump_y_tensor)

def gen_elem_bound_assembler(jacob):
    jacob_x, jacob_y = jacob
    nx_elem = jacob_x.shape[0]
    ny_elem = jacob_y.shape[0]

    bassemb_x = np.zeros((nx_elem+1, 2*nx_elem))
    bassemb_x[0,0] = 1.0
    bassemb_x[-1,-1] = 1.0
    for i in range(nx_elem-1):
        bassemb_x[i+1, 2*i+1] = 1.0 # i-th element, right boundary
        bassemb_x[i+1, 2*(i+1)] = 1.0 # (i+1)-th element, left boundary
    bassemb_x_tensor = K.constant(bassemb_x)

    bassemb_y = np.zeros((ny_elem+1, 2*ny_elem))
    bassemb_y[0,0] = 1.0
    bassemb_y[-1,-1] = 1.0
    for i in range(ny_elem-1):
        bassemb_y[i+1, 2*i+1] = 1.0 # i-th element, right boundary
        bassemb_y[i+1, 2*(i+1)] = 1.0 # (i+1)-th element, left boundary
    bassemb_y_tensor = K.constant(bassemb_y)

    return (bassemb_x_tensor, bassemb_y_tensor)

def gen_elem_bound_info(jacob, n_quads):
    """
    generate element boundary input data, and boundary jump matrix data
    :param jacob: tuple, (jacob_x, jacob_y)
                    where jacob_x : shape: (nx_elem, 2); [:,0] -- Jx_0e; [:,1] -- Jx_1e
                          jacob_y : shape: (ny_elem, 2); [:,0] -- Jy_0e; [:,1] -- Jy_1e
    :param n_quads: number of quadrature points
    :return: tuple, (elem_binputs, bjump)
                    elem_binputs: list of (nx_elem*ny_elem) tensors of shape (4*n_quads,2) for element boundary input data
                    bjump : tuple, (bjump_x, bjump_y)
                            bjump_x : x-jump tensor, shape: (nx_elem-1, 2*nx_elem)
                            bjump_y : y-jump tensor, shape: (ny_elem-1, 2*ny_elem)
    """
    elem_binputs = gen_elem_boundary_input(jacob, n_quads) # element boundary input tensor
    # elem_binputs contains a list of (nx_elem*ny_elem) tensors of shape (4*n_quads,1)

    bjump = gen_elem_boundary_jump(jacob) # element boundary jump tensor
    # now bjump = (bjump_x, bjump_y), boundary jump tensor

    bassemb = gen_elem_bound_assembler(jacob) # element boundary assembler tensor
    # now bassemb = (bassemb_x,bassemb_y), boundary assembler tensor

    return (elem_binputs, bjump, bassemb)

def one_elem_loss_generator(id_elem, comb_param, basis_info, model_info):
    """
    build loss function for one element
    :param id_elem: tuple, (ix_elem,iy_elem), id of current element and sub-model
    :param comb_param: tuple, (elem_param, elem_binfo)
            elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : tuple, (bc_L,bc_R,bc_B,bc_T)
                                    bc_L = (coord_L,bc_data_L)
                                    bc_R = (coord_R,bc_data_R)
                                    bc_B = (coord_B,bc_data_B)
                                    bc_T = (coord_T,bc_data_T)
                             jacobian: tuple, (jacob_x,jacob_y)
                                        jacob_x: (nx_elem,2), [:,0] -- Jx_0e; [:,1] -- Jx_1e
                                        jacob_y: (ny_elem,2), [:,0] -- Jy_0e; [:,1] -- Jy_1e
                             domain: tuple, (domain_x,domain_y)
                                        domain_x : list, coordinates of element boundaries in x
                                        domain_y : list, coordinates of element boundaries in y
                             ck_k : k value in C_k continuity
            elem_binfo: tuple, (elem_binputs, bjump, bassemb)
                                elem_binputs: list of n_elem tensors of shape (4*n_quads,2)
                                                        with element boundary quadrature points
                                bjump: tuple, (bjump_x, bjump_y)
                                        bjump_x: tensor, shape: (nx_elem-1, 2*nx_elem)
                                        bjump_y: tensor, shape: (ny_elem-1, 2*ny_elem)
                                bassemb: tuple, (bassemb_x, bassemb_y)
                                        bassemb_x: tensor, shape: (nx_elem+1, 2*nx_elem)
                                        bassemb_y: tensor, shape: (ny_elem+1, 2*ny_elem)
    :param basis_info: tuple, (B, Bd, W, B_bound)
                       where B : basis matrix tensor, shape: (n_quads, n_modes)
                             Bd : basis derivative matrix tensor, shape: (n_quads, n_modes)
                             W : weight matrix tensor, shape: (1, n_quads)
                             B_bound : tuple, (bound_left,bound_right)
                                        bound_left : tensor, shape: (1,n_modes)
                                        bound_right : tensor, shape: (1,n_modes)
    :param model_info: keras model
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             model.sub_models: list of sub-models for each element
    :return: loss function for the sub-model of this element

    if this is the first element, will compute equation residual of this element, and also of element-boundary
        C^k continuity conditions, as well as global boundary condition loss
    if this is not the first element, will compute only the equation residual of this element
    """
    # ++++++++++++++++++++++++++++++++++++++++++++#
    def The_Loss_Func(y_true, y_pred):
        """
        actual computation of loss function for ordinary element, not the first element
        only compute the residual loss of equation for current element
        :param y_true: label data
        :param y_pred: preduction data
        :return: loss value
        """
        return K.constant(0.0)

    # ++++++++++++++++++++++++++++++++++++++++++ #
    # quick return for other elements
    if id_elem != (0,0):
        return The_Loss_Func

    # +++++++++++++++++++++++++++++++++++++++++++ #
    elem_param, elem_binfo = comb_param

    n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k = elem_param
    jacob_x, jacob_y = jacob
    nx_elem = jacob_x.shape[0] # number of elements in x
    ny_elem = jacob_y.shape[0] # number of elements in y

    ix_elem, iy_elem = id_elem # element index in x and y directions
    this_elem_id = ix_elem*ny_elem + iy_elem # linear index of this element

    global_model = model_info # global_model contains the overall model
    sub_model_list = global_model.sub_models # list of sub-models
    this_sub_model = sub_model_list[this_elem_id] # sub_model for this element

    B_tensor, Bd_tensor, W_tensor, (B_bound_left, B_bound_right) = basis_info
    B_trans_tensor = K.transpose(B_tensor) # shape: (n_modes,n_quads)
    Bd_trans_tensor = K.transpose(Bd_tensor) # shape: (n_modes,n_quads)
    W_trans_tensor = K.transpose(W_tensor) # shape: (n_quads,1)
    B_bound_left_trans = K.transpose(B_bound_left) # shape: (n_modes,1)
    B_bound_right_trans = K.transpose(B_bound_right) # shape: (n_modes,1)
    # B_tensor contains basis function matrix tensor, shape: (n_quads, n_modes)
    # B_trans_tensor contains transpose of B_tensor, shape: (n_modes, n_quads)
    # Bd_tensor contains derivative basis tensor, shape: (n_quads, n_modes)
    # Bd_trans_tensor contains transpose of Bd_tensor, shape: (n_modes,n_quads)
    # W_tensor contains weight matrix tensor, shape: (1, n_quads)
    # W_trans_tensor contains transpose of W_tensor, shape: (n_quads,1)
    # B_bound_left: tensor for basis values at xi=-1, shape: (1, n_modes)
    # B_bound_left_trans: tranpose of B_bound_left, shape: (n_modes, 1)
    # B_bound_right: tensor for basis values at xi=1, shape: (1, n_modes)
    # B_bound_right_trans: transpose of B_bound_right, shape: (n_modes, 1)

    elem_binputs, bjump, bassemb = elem_binfo
    # now elem_binputs is a list of (nx_elem*ny_elem) tensors with shape (4*n_quads,2)
    #     tensor[0:n_quads,:] contains (x,y) coordinates on left element boundary
    #           [n_quads:2*n_quads,:] contains (x,y) coordinates on right element boundary
    #           [2*n_quads:3*n_quads,:] contains (x,y) coordinates on bottom element boundary
    #           [3*n_quads:4*n_quads,:] contains (x,y) coordinates on top element boundary
    # to be used for computing the function data on all element boundaries
    #     bjump = (bjump_x,bjump_y)
    #             bjump_x: tensor, shape: (nx_elem-1,2*nx_elem)
    #             bjump_y: tensor, shape: (ny_elem-1,2*ny_elem)
    #     bassemb = (bassemb_x, bassemb_y)
    #             bassemb_x: shape: (nx_elem+1, 2*nx_elem)
    #             bassemb_y: shape: (ny_elem+1, 2*ny_elem)

    curr_elem_binput = elem_binputs[this_elem_id]
    # now curr_elem_binput contains boundary input data for current element, shape: (4*n_quads,2)

    # ======================================= #
    # boundary conditions for domain
    bc_L, bc_R, bc_B, bc_T = bc

    # ====================================== #
    # continuity across element boundaries
    # element boundary jump matrix/tensor,
    # i.e. difference between neighboring elements on common boundary
    bjump_x_tensor, bjump_y_tensor = bjump # gen_elem_boundary_jump(jacob)
    # bjump_x_tensor : shape (nx_elem-1, 2*nx_elem)
    # bjump_y_tensor : shape (ny_elem-1, 2*ny_elem)

    # +++++++++++++++++++++++++++++++++ #
    bassemb_x, bassemb_y = bassemb
    bassemb_x_trans = K.transpose(bassemb_x) # shape: (2*nx_elem, nx_elem+1)
    bassemb_y_trans = K.transpose(bassemb_y) # shape: (2*ny_elem, ny_elem+1)
    # bassemb_x : shape: (nx_elem+1, 2*nx_elem)
    # bassemb_y : shape: (ny_elem+1, 2*ny_elem)
    # bassemb_x_trans: transpose of bassemb_x, shape: (2*nx_elem, nx_elem+1)
    # bassemb_y_trans: transpose of bassemb_y, shape: (2*ny_elem, ny_elem+1)

    # ++++++++++++++++++++++++++++++#
    # jacobian tensor
    jacob_x_tensor = K.constant(jacob_x[:,1:]) # shape: (nx_elem, 1)
    jacob_x_tensor_2 = K.reshape(jacob_x_tensor, (nx_elem, 1, 1, 1)) # shape: (nx_elem, 1, 1, 1)

    jacob_y_tensor = K.constant(jacob_y[:,1:]) # shape: (ny_elem, 1)
    jacob_y_tensor_2 = K.reshape(jacob_y_tensor, (1, ny_elem, 1, 1)) # shape: (1, ny_elem, 1, 1)

    # +++++++++++++++++++++++++++++++ #
    # weight tensors
    w_y_tensor = K.reshape(W_tensor, (1,1,1,n_quads)) # shape: (1,1,1,n_quads)
    w_x_tensor = K.reshape(W_tensor, (1,1,n_quads,1)) # shape: (1,1,n_quads,1)

    # ======================================= #
    def Equation_Residual(bio_tensors):
        """
        computation of residual tensor for equation of all elements, total loss
        :param bio_tensors: tuple, (bio_l, bio_r, bio_b, bio_t)
                            bio_l = (bcoord, bpred)
                                    bcoord: list of ny_elem input tensors of shape (n_quads,1) on left boundary
                                    bpred: list of ny_elem output tensors of shape (n_quads,1) on left boundary
                                            prediction of u on left boundary
                            bio_r = (bcoord, bpred)
                                    bcoord, bpred: similar to bio_l, for right boundary
                            bio_b = (bcoord,bpred); for bottom boundary
                            bio_t = (bcoord, bpred); for top boundary
        :return: equation residual tensor
        """
        # now global_model.inputs contains list of input tensors
        #                 .outputs contains list of output tensors
        #                 .targets contains list of label tensors
        xy_list = global_model.inputs # list of nx_elem*ny_elem input tensors of shape (n_quads*n_quads,2)
        u_list = global_model.outputs # list of nx_elem*ny_elem output tensors of shape (n_quads*n_quads,1)
        f_source_list = global_model.targets # list of nx_elem*ny_elem label tensors of shape (n_quads*n_quads,1)

        deriv_list = K.gradients(u_list, xy_list) # list of nx_elem*ny_elem tensors of shape (n_quads*n_quads,2)
        deriv = K.stack(deriv_list, 0) # shape: (nx_elem*ny_elem, n_quads*n_quads, 2)
        dudx = deriv[:,:,0] # shape: (nx_elem*ny_elem, n_quads*n_quads)
        dudy = deriv[:,:,1] # shape: (nx_elem*ny_elem, n_quads*n_quads)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # volume integral terms

        # Fp34 term: \int (lambda*u + f) * phi_p
        u_1 = K.stack(u_list, 0) # shape: (nx_elem*ny_elem, n_quads*n_quads, 1)
        u_2 = K.reshape(u_1, (nx_elem*ny_elem, n_quads, n_quads)) # shape: (nx_elem*ny_elem, n_quads,n _quads)
        f_1 = K.stack(f_source_list, 0) # shape: (nx_elem*ny_elem, n_quads*n_quads, 1), or (nx_elem*ny_elem,?,?)
        f_2 = f_1[:,:,0] # shape: (nx_elem*ny_elem,n_quads*n_quads) or (nx_elem*ny_elem,?)
        f_3 = K.reshape(f_2, (nx_elem*ny_elem, n_quads, n_quads)) # shape: (nx_elem*ny_elem, n_quads, n_quads)

        temp_v1 = lambda_coeff*u_2 + f_3 # shape: (nx_elem*ny_elem, n_quads, n_quads)
        temp_v2 = K.reshape(temp_v1, (nx_elem, ny_elem, n_quads, n_quads)) # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_v3 = w_x_tensor * temp_v2 * w_y_tensor # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_v4 = jacob_x_tensor_2 * temp_v3 * jacob_y_tensor_2 # shape: (nx_elem,ny_elem,n_quads,n_quads)

        temp_v5 = K.dot(temp_v4, B_tensor) # shape: (nx_elem,ny_elem,n_quads,n_modes)
        temp_v6 = K.dot(B_trans_tensor, temp_v5) # shape: (n_modes, nx_elem, ny_elem, n_modes)
        Fp34 = tf.transpose(temp_v6, [1, 2, 0, 3]) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp34 contains the term \int_{domain} (lambda*u + f) * phi_p,
        #          shape: (nx_elem,ny_elem,n_modes,n_modes)

        # Fp1 term: \int_{domain} du/dx * dphi_p/dx
        dudx_1 = K.reshape(dudx, (nx_elem, ny_elem, n_quads, n_quads)) # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_a1 = w_x_tensor * dudx_1 * w_y_tensor # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_a2 = temp_a1 * jacob_y_tensor_2 # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_a3 = K.dot(temp_a2, B_tensor) # shape: (nx_elem,ny_elem,n_quads,n_modes)
        temp_a4 = K.dot(Bd_trans_tensor, temp_a3) # shape: (n_modes,nx_elem,ny_elem,n_modes)
        Fp1 = tf.transpose(temp_a4, [1, 2, 0, 3]) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp1 contains \int_{domain} du/dx * dphi_p/dx term,
        #         shape: (nx_elem, ny_elem, n_modes, n_modes)

        # Fp2 term: \int_{domain} du/dy * dphi_p/dy
        dudy_1 = K.reshape(dudy, (nx_elem, ny_elem, n_quads, n_quads)) # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_b1 = w_x_tensor * dudy_1 * w_y_tensor # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_b2 = jacob_x_tensor_2 * temp_b1 # shape: (nx_elem,ny_elem,n_quads,n_quads)
        temp_b3 = K.dot(temp_b2, Bd_tensor) # shape: (nx_elem,ny_elem,n_quads,n_modes)
        temp_b4 = K.dot(B_trans_tensor, temp_b3) # shape: (n_modes,nx_elem,ny_elem,n_modes)
        Fp2 = tf.transpose(temp_b4, [1, 2, 0, 3]) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp2 contains \int_{domain} du/dy * dphi_p/dy, shape: (nx_elem,ny_elem,n_modes,n_modes)

        Fp_vol = Fp1 + Fp2 + Fp34 # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp_vol contains volume integral terms, shape: (nx_elem,ny_elem,n_modes,n_modes)

        # ++++++++++++++++++++++++++++++++++++++++++ #
        # boundary integral terms
        bio_l, bio_r, bio_b, bio_t = bio_tensors
        bcoord_l, bpred_l = bio_l # xy and u on left boundary
        bcoord_r, bpred_r = bio_r # xy and u on right boundary
        bcoord_b, bpred_b = bio_b # xy and u on bottom boundary
        bcoord_t, bpred_t = bio_t # xy and u on top boundary

        bderiv_l = K.gradients(bpred_l, bcoord_l) # list of ny_elem du/dx,du/dy tensors of shape (n_quads,2) on left boundary
        bderiv_r = K.gradients(bpred_r, bcoord_r) # list of ny_elem deivative tensors of shape (n_quads,2) on right boundary
        bderiv_b = K.gradients(bpred_b, bcoord_b) # list of nx_elem tensors of shape (n_quads,2) on bottom boundary
        bderiv_t = K.gradients(bpred_t, bcoord_t) # list of nx_elem tensors of shape (n_quads,2) on top boundary

        # left boundary
        bderiv_l_1 = K.stack(bderiv_l, 0) # shape: (ny_elem, n_quads, 2), on left boundary
        bdudx_l = bderiv_l_1[:,:,0] # shape: (ny_elem, n_quads)
        temp_c1 = jacob_y_tensor * bdudx_l * W_tensor # shape: (ny_elem, n_quads)
        temp_c2 = K.dot(temp_c1, B_tensor) # shape: (ny_elem, n_modes)
        temp_c3 = K.reshape(temp_c2, (ny_elem, 1, n_modes))
        temp_c4 = K.dot(B_bound_left_trans, temp_c3) # shape: (n_modes, ny_elem, n_modes)
        Gp_l_1 = tf.transpose(temp_c4, [1, 0, 2]) # shape: (ny_elem, n_modes, n_modes)
        Gp_l = -Gp_l_1 # negation
        # now Gp_l contains left boundary term, shape: (ny_elem, n_modes, n_modes)

        # right boundary
        bderiv_r_1 = K.stack(bderiv_r, 0) # shape: (ny_elem, n_quads, 2)
        bdudx_r = bderiv_r_1[:,:,0] # shape: (ny_elem, n_quads)
        temp_d1 = jacob_y_tensor * bdudx_r * W_tensor # shape: (ny_elem, n_quads)
        temp_d2 = K.dot(temp_d1, B_tensor) # shape: (ny_elem, n_modes)
        temp_d3 = K.reshape(temp_d2, (ny_elem, 1, n_modes)) # shape: (ny_elem, 1, n_modes)
        temp_d4 = K.dot(B_bound_right_trans, temp_d3) # shape: (n_modes, ny_elem, n_modes)
        Gp_r = tf.transpose(temp_d4, [1, 0, 2]) # shape: (ny_elem, n_modes, n_modes)
        # now Gp_r contains right boundary term, shape: (ny_elem, n_modes, n_modes)

        # bottom boundary
        bderiv_b_1 = K.stack(bderiv_b, 0) # shape: (nx_elem, n_quads, 2)
        bdudy_b = bderiv_b_1[:,:,1] # shape: (nx_elem, n_quads)
        temp_e1 = jacob_x_tensor * bdudy_b * W_tensor # shape: (nx_elem, n_quads)
        temp_e2 = K.dot(temp_e1, B_tensor) # shape: (nx_elem, n_modes)
        temp_e3 = K.reshape(temp_e2, (nx_elem, n_modes, 1)) # shape: (nx_elem, n_modes, 1)
        Gp_b_1 = K.dot(temp_e3, B_bound_left) # shape: (nx_elem, n_modes, n_modes)
        Gp_b = -Gp_b_1 # negation
        # now Gp_b contains bottom boundary term, shape: (nx_elem, n_modes, n_modes)

        # top boundary
        bderiv_t_1 = K.stack(bderiv_t, 0) # shape: (nx_elem, n_quads, 2)
        bdudy_t = bderiv_t_1[:,:,1] # shape: (nx_elem, n_quads)
        temp_f1 = jacob_x_tensor * bdudy_t * W_tensor # shape: (nx_elem, n_quads)
        temp_f2 = K.dot(temp_f1, B_tensor) # shape: (nx_elem, n_modes)
        temp_f3 = K.reshape(temp_f2, (nx_elem, n_modes, 1)) # shape: (nx_elem, n_modes, 1)
        Gp_t = K.dot(temp_f3, B_bound_right) # shape: (nx_elem, n_modes, n_modes)
        # now Gp_t contains top boundary term, shape: (nx_elem, n_modes, n_modes)

        # +++++++++++++++++++++++++++++++++++++++++ #
        # add boundary terms to volume terms
        # now Fp_vol contains volume integral terms, shape: (nx_elem, ny_elem, n_modes, n_modes)
        #     Gp_l contains left boundary term, shape: (ny_elem, n_modes, n_modes)
        #     Gp_r contains right boundary term, shape: (ny_elem, n_modes, n_modes)
        #     Gp_b contains bottom boundary term, shape: (nx_elem, n_modes, n_modes)
        #     Gp_t contains top boundary term, shape: (nx_elem, n_modes, n_modes)

        Gp_l_a = K.reshape(Gp_l, (1, ny_elem, n_modes, n_modes)) # shape: (1,ny_elem,n_modes,n_modes)
        Gp_r_a = K.reshape(Gp_r, (1, ny_elem, n_modes, n_modes)) # shape: (1,ny_elem,n_modes,n_modes)
        Gp_b_a = K.reshape(Gp_b, (nx_elem, 1, n_modes, n_modes)) # shape: (nx_elem,1,n_modes,n_modes)
        Gp_t_a = K.reshape(Gp_t, (nx_elem, 1, n_modes, n_modes)) # shape: (nx_elem,1,n_modes,n_modes)

        # how to update only a slice of the tensor?
        # will split it into two, update one slice, then concatenate the two back together
        temp_g1 = Fp_vol[0:1, :, :, :] # shape: (1,ny_elem,n_modes,n_modes), slice for left boundary
        temp_g2 = Fp_vol[1:, :, :, :] # shape: (nx_elem-1, ny_elem, n_modes, n_modes), slice for the rest
        temp_z1 = temp_g1 - Gp_l_a # shape: (1,ny_elem,n_modes,n_modes)
        Fp_a1 = K.concatenate([temp_z1, temp_g2], 0) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp_a1 contains Fp_vol - Gp_l, shape: (nx_elem,ny_elem,n_modes,n_modes)

        temp_h1 = Fp_a1[0:nx_elem-1, :, :, :] # shape: (nx_elem-1, ny_elem, n_modes, n_modes), slice for rest
        temp_h2 = Fp_a1[nx_elem-1:nx_elem, :, :, :] # shape: (1, ny_elem, n_modes, n_modes), slice for right boundary
        temp_z1 = temp_h2 - Gp_r_a # shape: (1,ny_elem,n_modes,n_modes)
        Fp_a2 = K.concatenate([temp_h1, temp_z1], 0) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp_a2 contains Fp_vol - (Gp_l + Gp_r), shape: (nx_elem,ny_elem,n_modes,n_modes)

        temp_I1 = Fp_a2[:,0:1,:,:] # shape: (nx_elem,1,n_modes,n_modes), slice for bottom boundary
        temp_I2 = Fp_a2[:, 1:, :, :] # shape: (nx_elem,ny_elem-1,n_modes,n_modes), slice for rest of tensor
        temp_z1 = temp_I1 - Gp_b_a # shape: (nx_elem,1,n_modes,n_modes)
        Fp_a3 = K.concatenate([temp_z1, temp_I2], 1) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp_a3 contains Fp_vol - (Gp_l + Gp_r + Gp_b), shape: (nx_elem,ny_elem,n_modes,n_modes)

        temp_J1 = Fp_a3[:,0:ny_elem-1,:,:] # shape: (nx_elem,ny_elem-1,n_modes,n_modes), slice for rest of tensor
        temp_J2 = Fp_a3[:,ny_elem-1:ny_elem,:,:] # shape: (nx_elem,1,n_modes,n_modes), slice for top boundary
        temp_z1 = temp_J2 - Gp_t_a # shape: (nx_elem,1,n_modes,n_modes)
        Fp = K.concatenate([temp_J1, temp_z1], 1) # shape: (nx_elem,ny_elem,n_modes,n_modes)
        # now Fp contains Fp_vol - (Gp_l + Gp_r + Gp_b + Gp_t), shape: (nx_elem,ny_elem,n_modes,n_modes)
        #        all equation residual terms, in terms of local modes

        # now Fp contains all terms in equation residual
        return Fp

    def process_global_modes(local_modes):
        """
        process global modes based on local modes
        :param local_modes: tensor for local modes, shape: (nx_elem,ny_elem,n_modes,n_modes)
                            [:,:,r,s] ---> ... + r*n_modes + s in terms of ordering
                            r, s = 0, 1 --> corresponds to boundary modes in 1D
        :return: tensors for global vertex, edge and interior modes
        """
        # +++++++++++++++++++++++++++++++++ #
        # vertex modes
        v_modes = local_modes[:,:,0:2,0:2] # local vertex modes, shape: (nx_elem,ny_elem,2,2)
        vm_1 = tf.transpose(v_modes, [0, 2, 1, 3]) # shape: (nx_elem, 2, ny_elem, 2)
        vm_2 = K.reshape(vm_1, (nx_elem*2, ny_elem*2)) # shape: (nx_elem*2, ny_elem*2)
        vm_3 = K.dot(bassemb_x, vm_2) # shape: (nx_elem+1, ny_elem*2)
        g_vmodes = K.dot(vm_3, bassemb_y_trans) # shape: (nx_elem+1, ny_elem+1)
        # now g_vmodes contains global vertex modes, shape: (nx_elem+1, ny_elem+1)

        # ++++++++++++++++++++++++++++++++ #
        # edge modes
        e_modes_x = local_modes[:,:,0:2,2:] # vertical edge modes, shape: (nx_elem,ny_elem,2,n_modes-2)
        emx_1 = tf.transpose(e_modes_x, [0, 2, 1, 3]) # shape: (nx_elem,2,ny_elem,n_modes-2)
        emx_2 = K.reshape(emx_1, (nx_elem*2, ny_elem*(n_modes-2))) # shape: (nx_elem*2,ny_elem*(n_modes-2))
        g_emodes_x = K.dot(bassemb_x, emx_2) # shape: (nx_elem+1, ny_elem*(n_modes-2))
        # now g_emodes_x contains vertical global edge modes, shape: (nx_elem+1,ny_elem*(n_modes-2))

        e_modes_y = local_modes[:,:,2:,0:2] # horizontal edge modes, shape: (nx_elem,ny_elem,n_modes-2,2)
        emy_1 = tf.transpose(e_modes_y, [0, 2, 1, 3]) # shape: (nx_elem,n_modes-2,ny_elem,2)
        emy_2 = K.reshape(emy_1, (nx_elem*(n_modes-2), ny_elem*2))
        g_emodes_y = K.dot(emy_2, bassemb_y_trans) # shape: (nx_elem*(n_modes-2), ny_elem+1)
        # now g_emodes_y contains horizontal global edge modes, shape: (nx_elem*(n_modes-2),ny_elem+1)

        # ++++++++++++++++++++++++++++++++ #
        # interior modes
        i_modes = local_modes[:,:,2:,2:] # interior modes, shape: (nx_elem,ny_elem,n_modes-2,n_modes-2)
        # all interior modes are global modes

        return (g_vmodes, g_emodes_x, g_emodes_y, i_modes)

    # ============================= #
    def loss_generator_Ck(ck):
        """
        Ck loss function generator
        :param ck: integer, >= 0, k value in C_k continuity
        :return: loss function for C_k
        """

        def calc_elem_boundary_jump(elem_bdata):
            """
            compute boundary jump integral residual tensor
            :param elem_bdata: tensor of shape (nx_elem*ny_elem,4*n_quads,1)
            :return: tuple, (lr_bjump, bt_jump)
                        lr_bjump : tensor, shape: (nx_elem-1, ny_elem, n_modes), on left/right boundaries
                        bt_jump : tensor, shape: (ny_elem-1,nx_elem,n_modes), residual on bottom/top boundaries
            """
            bdata_1 = K.reshape(elem_bdata, (nx_elem*ny_elem, 2, 2*n_quads))
            # now bdata_1[:,0,:] contains (L,R) element boundary data, left/right boundaries
            #            [:,1,:] contains (B,T) element boundary data, bottom/top boundaries
            bdata_lr = bdata_1[:,0,:] # shape: (nx_elem*ny_elem,2*n_quads), left/right boundary data
            bdata_bt = bdata_1[:,1,:] # shape: (nx_elem*ny_elem,2*n_quads), top/bottom boundary data
            # reshape data
            bdata_lr_2 = K.reshape(bdata_lr, (nx_elem, ny_elem, 2*n_quads))
            bdata_bt_2 = K.reshape(bdata_bt, (nx_elem, ny_elem, 2*n_quads))

            # +++++++++++++++++++++++++++++++++++++++ #
            # left/right boundary residual
            # need to transpose bdata_lr_2
            bdata_lr_trans = tf.transpose(bdata_lr_2, perm=[1, 0, 2])
            # now bdata_lr_trans has shape: (ny_elem,nx_elem,2*n_quads)
            bdata_lr_trans_1 = K.reshape(bdata_lr_trans, (ny_elem, 2*nx_elem, n_quads))
            # new shape: (ny_elem, 2*nx_elem, n_quads)
            bjump_lr = K.dot(bjump_x_tensor, bdata_lr_trans_1) # shape: (nx_elem-1,ny_elem,n_quads)
            # now bjump_lr contains boundary jump of function on left/right boundaries
            #       shape: (nx_elem-1, ny_elem, n_quads)

            # compute integral \int [u_left - u_right] * phi_s(y) dy
            resh_w_tensor = K.reshape(W_tensor, (1,1,n_quads)) # shape: (1,1,n_quads)
            temp_v1 = bjump_lr * resh_w_tensor # shape: (nx_elem-1,ny_elem,n_quads)
            resh_jacob_y = K.reshape(jacob_y_tensor, (1,ny_elem,1)) # shape: (1,ny_elem,1)
            temp_v2 = temp_v1 * resh_jacob_y # shape: (nx_elem-1,ny_elem,n_quads)

            lr_jump_residual = K.dot(temp_v2, B_tensor) # shape: (nx_elem-1,ny_elem,n_modes)
            # now lr_jump_residual contains the residual tensor on left/right boundaries
            #       shape: (nx_elem-1, ny_elem, n_modes)

            # ++++++++++++++++++++++++++++++++++++++ #
            # bottom/top boundary residual
            bdata_bt_resh = K.reshape(bdata_bt_2, (nx_elem, 2*ny_elem, n_quads))
            # new shape: (nx_elem,2*ny_elem,n_quads)
            bjump_bt = K.dot(bjump_y_tensor, bdata_bt_resh)
            # now bjump_bt contains jump of function on bottom/top boundaries
            #       shape: (ny_elem-1, nx_elem, n_quads)

            # compute integral \int [u_bottom - u_top] * phi_r(x) dx
            temp_w1 = bjump_bt * resh_w_tensor # shape: (ny_elem-1, nx_elem, n_quads)
            resh_jacob_x = K.reshape(jacob_x_tensor, (1,nx_elem,1))
            temp_w2 = temp_w1 * resh_jacob_x # shape: (ny_elem-1,nx_elem,n_quads)

            bt_jump_residual = K.dot(temp_w2, B_tensor) # shape: (ny_elem-1,nx_elem,n_modes)
            # now bt_jump_residual contains the residual tensor on bottom/top boundaries
            #       shape: (ny_elem-1,nx_elem,n_modes)

            return (lr_jump_residual, bt_jump_residual)

        def loss_func_c0():
            """
            compute loss across element boundary for C^0 continuity across element boundaries
            :return: loss value
            """
            # elem_binputs contains boundary coordinate tensor for each element
            #   a list of nx_elem*ny_elem tensors of shape (4*n_quads,2)
            elem_bpred = global_model(elem_binputs)
            # now elem_bpred is prediction of element boundary data
            #         a list of nx_elem*ny_elem tensors of shape (4*n_quads,1)
            elem_bdata = K.stack(elem_bpred, 0) # shape: (nx_elem*ny_elem, 4*n_quads, 1)
            # element boundary data

            lr_jump_residual, bt_jump_residual = calc_elem_boundary_jump(elem_bdata)
            # jump residuals on left/right and bottom/top boundaries
            # lr_jump_residual, shape: (nx_elem-1,ny_elem,n_modes)
            # bt_jump_residual, shape: (ny_elem-1,nx_elem,n_modes)

            if nx_elem > 1:
                loss_lr = K.mean(K.square(lr_jump_residual))
            else:
                loss_lr = K.constant(0.0)
            if ny_elem > 1:
                loss_bt = K.mean(K.square(bt_jump_residual))
            else:
                loss_bt = K.constant(0.0)
            this_loss = loss_lr + loss_bt
            #this_loss = K.sum(K.square(lr_jump_residual)) + K.sum(K.square(bt_jump_residual))
            return this_loss

        def loss_func_c1():
            """
            compute loss across element boundary for C^1 continuity across element boundaries
            :return: loss value
            """
            # elem_binputs contains boundary coordinate tensor for each element
            #   a list of nx_elem*ny_elem tensors of shape (4*n_quads,2)
            elem_bpred = global_model(elem_binputs)
            # now elem_bpred is prediction of element boundary data
            #         a list of nx_elem*ny_elem tensors of shape (4*n_quads,1)
            elem_bdata = K.stack(elem_bpred, 0) # shape: (nx_elem*ny_elem, 4*n_quads, 1)
            # element boundary data

            # compute du/dx and du/dy
            bdata_deriv = K.gradients(elem_bpred,elem_binputs)
            # list of (nx_elem*ny_elem) tensors of shape (4*n_quads,2), containing du/dx and du/dy
            deriv_comb = K.stack(bdata_deriv, 0) # shape: (nx_elem*ny_elem, 4*n_quads, 2)
            bdata_dudx = deriv_comb[:,:,0:1] # shape: (nx_elem*ny_elem, 4*n_quads, 1)
            bdata_dudy = deriv_comb[:,:,1:2] # shape: (nx_elem*ny_elem, 4*n_quads, 1)

            # element boundary jump residuals
            u_jump_lr, u_jump_bt = calc_elem_boundary_jump(elem_bdata) # for u
            dudx_jump_lr, dudx_jump_bt = calc_elem_boundary_jump(bdata_dudx) # for du/dx
            dudy_jump_lr, dudy_jump_bt = calc_elem_boundary_jump(bdata_dudy) # for du/dy
            # u_jump_lr, dudx_jump_lr, dudy_jump_lr : shape: (nx_elem-1,ny_elem,n_modes)
            # u_jump_bt, dudx_jump_bt, dudy_jump_bt : shape: (ny_elem-1,nx_elem,n_modes)

            #this_loss = (K.sum(K.square(u_jump_lr)) + K.sum(K.square(u_jump_bt))
            #             + K.sum(K.square(dudx_jump_lr)) + K.sum(K.square(dudx_jump_bt))
            #             + K.sum(K.square(dudy_jump_lr)) + K.sum(K.square(dudy_jump_bt)) )
            loss_lr = K.constant(0.0)
            if nx_elem > 1:
                loss_lr = K.mean(K.square(u_jump_lr)) + K.mean(K.square(dudx_jump_lr)) + K.mean(K.square(dudy_jump_lr))
            loss_bt = K.constant(0.0)
            if ny_elem > 1:
                loss_bt = K.mean(K.square(u_jump_bt)) + K.mean(K.square(dudx_jump_bt)) + K.mean(K.square(dudy_jump_bt))
            this_loss = loss_lr + loss_bt
            #this_loss = (K.mean(K.square(u_jump_lr)) + K.mean(K.square(u_jump_bt))
            #             + K.mean(K.square(dudx_jump_lr)) + K.mean(K.square(dudx_jump_bt))
            #             + K.mean(K.square(dudy_jump_lr)) + K.mean(K.square(dudy_jump_bt)))
            return this_loss

        def loss_func_c2():
            """
            loss for C^2 continuity across element boundaries
            :return: loss value
            """
            return K.constant(0.0)

        def loss_func_cm1():
            return K.constant(0.0)

        def loss_func_ck_default():
            return K.constant(0.0)

        # ++++++++++++ #
        if ck == 0:
            return loss_func_c0
        elif ck == 1:
            return loss_func_c1
        elif ck == 2:
            return loss_func_c2
        elif ck == -1:
            return loss_func_cm1
        else:
            print("ERROR: loss_generator_ck() -- C^k continuity with (infinity > k > 4) is not implemented!\n")
            return loss_func_ck_default
        # +++++++++++++ #

    The_Loss_Func_Ck = loss_generator_Ck(ck_k)
    # now The_Loss_Func_Ck is the loss function for computing Ck continuity loss
    #     across element boundaries

    def calc_boundary_residual(bound_model, bc_struct, jacob_tensor):
        """
        compute boundary residual tensor
        :param bound_model: keras model for this boundary
        :param bc_struct: tuple, (coord, data)
                            coord : list of this_jacob.shape[0] tensors of shape (n_quads,2)
                            data : list of this_jacob.shape[0] tensors of shape (n_quads,1)
        :param jacob_tensor: shape: (num_elem, 1)
        :return: boundary residual tensor, shape: (num_elem,n_modes)
        """
        jacob_shape = K.int_shape(jacob_tensor)
        num_elem = jacob_shape[0]
        bcoord, bdata = bc_struct
        bdata_1 = K.stack(bdata, 0) # shape: (num_elem,n_quads,1)

        bpred = bound_model(bcoord) # list of num_elem tensors of shape (n_quads,1), boundary prediction tensor
        bpred_1 = K.stack(bpred, 0) # shape: (num_elem,n_quads,1)

        bres = bpred_1 - bdata_1 # difference in field data, shape: (num_elem, n_quads, 1)
        bres_1 = K.reshape(bres, (num_elem,n_quads)) # shape: (num_elem,n_quads)

        # compute \int_{boundary} (u_pred - u_bc) * phi_p
        temp_v1 = bres_1 * W_tensor # shape: (num_elem, n_quads)
        temp_v2 = temp_v1 * jacob_tensor # shape: (num_elem, n_quads)
        result = K.dot(temp_v2, B_tensor) # shape: (num_elem, n_modes)
        # now result contains boundary residual tensor, shape: (num_elem, n_modes)

        return result, (bcoord, bpred)

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
        # compute boundary residual tensor
        # left boundary
        Tb_l, bio_tensors_l = calc_boundary_residual(global_model.submodel_boundary_L,
                                       bc_L, jacob_y_tensor) # left boundary
        # Tb_l contains BC residual tensor on left boundary, shape: (ny_elem, n_modes)
        # bio_tensors_l: (bcoord, bpred)
        #                bcoord contains list of ny_elem input tensors of shape (n_quads,2) on current boundary
        #                bpred contains list of ny_elem output tensors of shape (n_quads,1) on current boundary
        #                      the prediction of u on current domain boundary

        # right boundary
        Tb_r, bio_tensors_r = calc_boundary_residual(global_model.submodel_boundary_R,
                                      bc_R, jacob_y_tensor)
        # now Tb_r contains BC residual tensor on right boundary, shape: (ny_elem, n_modes)
        # bio_tensors_r: (bcoord, bpred)
        #                bcoord contains list of ny_elem input tensors of shape (n_quads,2) on current boundary
        #                bpred contains list of ny_elem output tensors of shape (n_quads,1) on current boundary
        #                      the prediction of u on current domain boundary

        # bottom boundary
        Tb_b, bio_tensors_b = calc_boundary_residual(global_model.submodel_boundary_B,
                                      bc_B, jacob_x_tensor)
        # now Tb_b contains BC residual tensor on bottom boundary, shape: (nx_elem, n_modes)
        # bio_tensors_b: (bcoord, bpred)
        #                bcoord contains list of ny_elem input tensors of shape (n_quads,2) on current boundary
        #                bpred contains list of ny_elem output tensors of shape (n_quads,1) on current boundary
        #                      the prediction of u on current domain boundary

        # top boundary
        Tb_t, bio_tensors_t = calc_boundary_residual(global_model.submodel_boundary_T,
                                      bc_T, jacob_x_tensor)
        # now Tb_t contains BC residual tensor on top boundary, shape: (nx_elem, n_modes)
        # bio_tensors_t: (bcoord, bpred)
        #                bcoord contains list of ny_elem input tensors of shape (n_quads,2) on current boundary
        #                bpred contains list of ny_elem output tensors of shape (n_quads,1) on current boundary
        #                      the prediction of u on current domain boundary

        # +++++++++++++++++++++++++++++++++++++++++++++++++#
        # equation residual
        bio_tensors = (bio_tensors_l, bio_tensors_r, bio_tensors_b, bio_tensors_t)
        T_tot = Equation_Residual(bio_tensors)
        # now T_tot contains residual tensor for equation of this element,
        #           shape: (nx_elem,ny_elem,n_modes, n_modes)

        # process equation residuals, global modes
        gv_modes, ge_modes_x, ge_modes_y, gi_modes = process_global_modes(T_tot)
        # now gv_modes contains global vertex modes, shape: (nx_elem+1, ny_elem+1)
        #     ge_modes_x contains vertical global edge modes, shape: (nx_elem+1,ny_elem*(n_modes-2))
        #     ge_modes_y contains horizontal global edge modes, shape: (nx_elem*(n_modes-2),ny_elem+1)
        #     gi_modes contains global interior modes, shape: (nx_elem,ny_elem,n_modes-2,n_modes-2)

        gv_flat = K.flatten(gv_modes)
        ge_flat_x = K.flatten(ge_modes_x)
        ge_flat_y = K.flatten(ge_modes_y)
        gi_flat = K.flatten(gi_modes)
        value = K.concatenate((gv_flat, ge_flat_x, ge_flat_y, gi_flat))

        # ========================= #
        # C^k continuity across element boundary
        Ck_loss = The_Loss_Func_Ck()
        # now Ck_loss contains loss value for C_k continuity across element boundaries

        # ========================= #
        #this_loss = K.mean(K.square(T_tot)) \
        #            + (K.sum(K.square(Tb_l)) + K.sum(K.square(Tb_r)) + K.sum(K.square(Tb_b))
        #                + K.sum(K.square(Tb_t)) + Ck_loss)*(nx_elem*ny_elem)
        this_loss = K.mean(K.square(value)) \
                    + (K.mean(K.square(Tb_l)) + K.mean(K.square(Tb_r)) + K.mean(K.square(Tb_b))
                       + K.mean(K.square(Tb_t)) + Ck_loss)
        return this_loss

    # ====================================== #
    return The_First_Loss_Func

    # ====================================== #

def multi_elem_loss_generator(comb_param, basis_info, model_info):
    """
    build list of loss functions
    :param comb_param: tuple, (elem_param, elem_binfo)
           elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacobian, domain, ck_k)
                       where n_modes is the number of modes in each element
                             n_quads is the number of quadrature points in each element
                             lambda_coeff is the lambda coefficient
                             bc : boundary condition data
                             jacobian: jacobian data
                             domain: domain data
                             ck_k : k value in C_k continuity
           elem_binfo : element boundary information data
    :param basis_info: tuple, (B, Bd, W, B_bound) tensors
                       where B : basis matrix tensor, shape: (n_quads, n_modes)
                             Bd : basis derivative matrix tensor, shape: (n_quads, n_modes)
                             W : weight matrix tensor, shape: (1, n_quads)
                             B_bound : tuple, (bound_left,boud_right),
                                        matrix of basis values on x=-1, 1; each has shape: (1, n_modes)
    :param model_info: keras model
                       where model: multi_element_model for n_elem elements, overall model of n_elem sub-models
                             model.sub_models contains list of sub-models
    :return: list of loss functions for the multi-element model

    Note: all loss will be computed by the first sub-model, or on first element
    """
    elem_param, _ = comb_param
    _, _, _, _, jacob, _, _ = elem_param
    jacob_x, jacob_y = jacob
    nx_elem = jacob_x.shape[0]
    ny_elem = jacob_y.shape[0]
    #norm_fac = 1.0/(nx_elem*ny_elem)

    loss_func_set = []
    loss_weights = []
    for ix_elem in range(nx_elem):
        for iy_elem in range(ny_elem):
            loss_one = one_elem_loss_generator((ix_elem,iy_elem), comb_param, basis_info, model_info)
            loss_func_set.append(loss_one)
            loss_weights.append(0.0)
    loss_weights[0] = 1.0
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
                            input_shape=(layers[ix_lay],) ) )
    ix_lay = ix_lay + 1

    for i in range(N_layer - 1):
        this_model.add(klay.Dense(layers[ix_lay + i + 1], activation=activations[ix_lay + i + 1]))

    return this_model

def input_scale_func_generator(id_elem, jacob):
    """
    function generator for scaling input data to [-1, 1]
    :param id_elem: tuple, (ix_elem,iy_elem), id of current element
    :param jacob: jacobian data for all elements, tuple, (jacob_x,jacob_y), each has shape: (n_elem, 2)
                  jacob_x[:,0] contains Jx_0e, jacob_x[:,1] contains Jx_1e
                  jacob_y[:,0] -- Jy_0e; [:,1] -- Jy_1e
                  scaling is x = Jx_1e * xi + Jx_0e, where x is physical coordinate, xi is in [-1,1]
                  similarly, y = Jy_0e + Jy_1e * eta, where eta in [-1,1]
    :return: scaling function
    """
    jacob_x, jacob_y = jacob
    ix_elem, iy_elem = id_elem
    J0_x = jacob_x[ix_elem, 0]
    J1_x = jacob_x[ix_elem, 1]
    J1_x_inv = 1.0/J1_x

    J0_y = jacob_y[iy_elem, 0]
    J1_y = jacob_y[iy_elem, 1]
    J1_y_inv = 1.0/J1_y

    def scale_func(coord):
        """
        scaling coordinate to standard range: [-1,1]
        :param coord: tensor, shape: (?,2)
        :return: scaled tensor
        """
        x_in = coord[:,0:1] # shape: (?, 1)
        y_in = coord[:,1:] # shape: (?,1)
        x_out = J1_x_inv*(x_in - J0_x)
        y_out = J1_y_inv*(y_in - J0_y)
        coord_out = K.concatenate([x_out, y_out], 1) # shape: (?,2)
        return coord_out

    return scale_func


def build_submodel(id_elem, layers, activ):
    """
    build sub-model for one element
    :param: id_elem, id of this sub-model
    :param layers: list of nodes for each layer
    :param activ: list of activation functions for each layer
    :return: the sub-model
    """
    n_layers = len(layers)
    this_input = keras.layers.Input(shape=[layers[0]], name="input_"+str(id_elem)) # input layer

    # first hidden layer
    if n_layers == 2:
        this_name = "output_" + str(id_elem)
    else:
        this_name = "hidden_1_" + str(id_elem)
    this_layer = keras.layers.Dense(layers[1], activation=activ[1], name=this_name)(this_input)

    for i in range(n_layers-2):
        if i == n_layers-3:
            this_name = "output_" + str(id_elem)
        else:
            this_name = "hidden_" + str(i+2) + "_" + str(id_elem)
        this_layer = keras.layers.Dense(layers[i+2], activation=activ[i+2],
                                        name=this_name)(this_layer)

    # now this_layer contains the output layer for sub-model
    this_model = keras.Model(inputs=[this_input], outputs=[this_layer])
    return this_model

def build_submodel_A(id_elem, layers, activ, jacob):
    """
    build sub-model for one element
    :param: id_elem, tuple, (ix_elem,iy_elem), id of this sub-model
    :param layers: list of nodes for each layer
    :param activ: list of activation functions for each layer
    :param jacob: bacobian data, tuple (jacob_x,jacob_y),
                  jacob_x : shape: (nx_elem,2). [:,0] -- J_0e; [:,1] -- J_1e
                  jacob_y : shape: (ny_elem,2), [:,0] -- J_0e; [:,1] -- J_1e
    :return: the sub-model
    """
    ix_elem, iy_elem = id_elem
    n_layers = len(layers)
    this_input = keras.layers.Input(shape=[layers[0]],
                                    name="input_"+str(ix_elem)+"_"+str(iy_elem)) # input layer

    # +++++++++++++++++++++++++++++++++ #
    # scaling layer, lambda layer
    # add lambda layer to re-scale input data to [-1,1]
    func = input_scale_func_generator(id_elem, jacob)
    lambda_layer = keras.layers.Lambda(func,
                                       name="lambda_"+str(ix_elem)+"_"+str(iy_elem))(this_input)
    # now the lambda layer is added behind the input to re-scale data to [-1,1]

    # first hidden layer
    if n_layers == 2:
        this_name = "output_" + str(ix_elem) + "_" + str(iy_elem)
    else:
        this_name = "hidden_1_" + str(ix_elem) + "_" + str(iy_elem)
    this_layer = keras.layers.Dense(layers[1], activation=activ[1], name=this_name)(lambda_layer)

    for i in range(n_layers-2):
        if i == n_layers-3:
            this_name = "output_" + str(ix_elem) + "_" + str(iy_elem)
        else:
            this_name = "hidden_" + str(i+2) + "_" + str(ix_elem) + "_" + str(iy_elem)
        this_layer = keras.layers.Dense(layers[i+2], activation=activ[i+2],
                                        name=this_name)(this_layer)

    # now this_layer contains the output layer for sub-model
    this_model = keras.Model(inputs=[this_input], outputs=[this_layer])
    this_model.id_elem = id_elem

    return this_model

def build_multi_element_model(n_elem, layers, activ, jacob):
    """
    build model for multiple elements
    :param n_elem: number of elements, >=1
    :param layers: layers vector in each one-element model
    :param activ: activation vector in each one-element model
    :return: multi-element model
    """
    nx_elem, ny_elem = n_elem
    model_set = []
    input_set = [] # for all elements
    output_set = []
    L_input_set = [] # list of inputs for elements on left boundary
    L_output_set = [] # list of outputs for elements on right boundary
    R_input_set = [] # list of inputs for elements on right boundary
    R_output_set = [] # list of outputs for elemeents on right boundary
    B_input_set = [] # bottom boundary
    B_output_set = []
    T_input_set = [] # top boundary
    T_output_set = []
    for ix_elem in range(nx_elem):
        for iy_elem in range(ny_elem):
            elem_model = build_submodel_A((ix_elem, iy_elem), layers, activ, jacob)

            model_set.append(elem_model)
            input_set = input_set + elem_model.inputs
            output_set= output_set + elem_model.outputs

            if ix_elem == 0:
                L_input_set = L_input_set + elem_model.inputs
                L_output_set = L_output_set + elem_model.outputs
            if ix_elem == nx_elem-1:
                R_input_set = R_input_set + elem_model.inputs
                R_output_set = R_output_set + elem_model.outputs
            if iy_elem == 0:
                B_input_set = B_input_set + elem_model.inputs
                B_output_set = B_output_set + elem_model.outputs
            if iy_elem == ny_elem-1:
                T_input_set = T_input_set + elem_model.inputs
                T_output_set = T_output_set + elem_model.outputs
    # now model_set contains the list of element models
    #     input_set contains the list of input tensors
    #     output_set contains the list output tensors

    # build the overall model
    this_model = keras.Model(inputs=input_set, outputs=output_set)
    # now this_model contains the multi-input multi-output model
    this_model.sub_models = model_set

    # build the boundary models
    submodel_L = keras.Model(inputs=L_input_set, outputs=L_output_set) # model for left boundary
    submodel_R = keras.Model(inputs=R_input_set, outputs=R_output_set) # model for right boundary
    submodel_B = keras.Model(inputs=B_input_set, outputs=B_output_set) # model for bottom boundary
    submodel_T = keras.Model(inputs=T_input_set, outputs=T_output_set) # model for top boundary
    this_model.submodel_boundary_L = submodel_L
    this_model.submodel_boundary_R = submodel_R
    this_model.submodel_boundary_B = submodel_B
    this_model.submodel_boundary_T = submodel_T

    return this_model

def build_model(layers, activ, elem_param, elem_binfo):
    """
    build DNN models
    :param layers: list containing the number of nodes for each layer
    :param activ: list containing the activation function for each layer
    :param elem_param: tuple, (n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k)
            n_modes: element order in each element, in both x and y directions
            n_quads: number of quadrature points in both x and y directions in each element
            lambda_coeff: coefficient in Helmholtz equation
            bc: tuple, (bc_L, bc_R, bc_B, bc_T)
                      where bc_L = (coord, bc_data),
                                    coord -- list of ny_elem numpy arrays of shape (n_quads,2) containing
                                             coordinates of wuadrature points on left boundary
                                    bc_data -- list of ny_elem numpy arrays of shape (n_quads,2) containing
                                             boundary condition data for left boundary
                            bc_R = (coord, bc_data),
                                    coord -- list of ny_elem arrays of quadrature points on right boudnary
                                    bc_data -- list ny_elem numpy arrays with boundary data for right boundary
                            bc_B = (coord, bc_data),
                                    ... for bottom boundary
                            bc_T = (coord, bc_data),
                                    ... for top boundary
            jacob: tuple, (jacobian_x, jacobian_y), containing jacobian data for x and y directions
                         jacobian_x -- shape: (nx_elem,2), containing (J_0e, J_1e),
                                        mapping in x: x = J_0e + J_1e*xi, for xi in [-1,1]
                         jacobian_y -- shape: (ny_elem, 2), containing (J_0e,J_1e),
                                        mapping in y: y = J_0e + J_1e*eta, for eta in [-1,1]
            this_domain: tuple, (domain_x, domain_y)
            ck_k: continuity index
    :param elem_binfo : elementary boundary information
    :return: keras model
    """
    n_modes, n_quads, _, _, jacob, _, _ = elem_param
    jacob_x, jacob_y = jacob
    n_elem = (jacob_x.shape[0], jacob_y.shape[0])
    my_model = build_multi_element_model(n_elem, layers, activ, jacob)
    # now my_model contains the multi-element DNN model

    basis_info = get_basis_info(n_modes, n_quads) # basis data
    #elem_param = (n_modes, n_quads, lambda_coeff, bc, jacob, this_domain, ck_k)
    comb_param = (elem_param, elem_binfo)
    loss_func_list, loss_weight_list = multi_elem_loss_generator(comb_param, basis_info, my_model)
    # now loss_func_list contains list of loss functions
    #     loss_weight_list contains list of loss weights

    # compile model
    my_model.compile(optimizer='adam',
                     loss=loss_func_list, loss_weights = loss_weight_list,
                     metrics=['accuracy'])
    return my_model

def get_learning_rate(model):
    lr = K.get_value(model.optimizer.lr)
    return float(lr)

def set_learning_rate(model, LR):
    K.set_value(model.optimizer.lr, LR)
    #return get_learning_rate(model)

def gen_bc_data(jacob, n_quads):
    """
    generate boundary condition data
    :param jacob: jacobian, tuple, (jacob_x,jacob_y), each is numpy array of shape (n_elem,2)
    :param n_quads: number of quadrature points
    :return: (bc_L,bc_R,bc_B,bc_T)
             bc_L = (coord_L, bc_data_L)
             bc_R = (coord_R, bc_data_R)
             bc_B = (coord_B, bc_data_B)
             bc_T = (coord_T, bc_data_T)
    """
    jacob_x, jacob_y = jacob
    nx_elem = jacob_x.shape[0]
    ny_elem = jacob_y.shape[0]
    zz, _ = calc_zw(n_quads)

    x_left = jacob_x[0,0] - jacob_x[0,1]
    x_right = jacob_x[-1,0] + jacob_x[-1,1]
    y_bottom = jacob_y[0,0] - jacob_y[0,1]
    y_top = jacob_y[-1,0] + jacob_y[-1,1]

    # Left right boundaries
    coord_L = []
    coord_R = []
    bc_data_L = []
    bc_data_R = []
    for iy_elem in range(ny_elem):
        tmp_coord = np.zeros((2, n_quads, 2))

        tmp_coord[0,:,0] = x_left # x coordinate of left boundary
        tmp_coord[1,:,0] = x_right # x coordinate of right boundary
        tmp_coord[0,:,1] = jacob_y[iy_elem,0] + jacob_y[iy_elem,1]*zz # y coordintates of left boundary
        tmp_coord[1,:,1] = tmp_coord[0,:,1] # y coordinate of right boundary
        tmp_L = K.constant(tmp_coord[0,:,:]) # shape: (n_quads,2)
        tmp_R = K.constant(tmp_coord[1,:,:]) # shape: (n_quads,2)
        coord_L.append(tmp_L)
        coord_R.append(tmp_R)

        data_L = the_anal_soln(tmp_coord[0,:,:])
        data_R = the_anal_soln(tmp_coord[1,:,:])
        data_L_tensor = K.constant(data_L) # shape: (n_quads,2)
        data_R_tensor = K.constant(data_R) # shape: (n_quads,2)
        bc_data_L.append(data_L_tensor)
        bc_data_R.append(data_R_tensor)

    # top/bottom boundaries
    coord_B = []
    coord_T = []
    bc_data_B = []
    bc_data_T = []
    for ix_elem in range(nx_elem):
        tmp_coord = np.zeros((2, n_quads, 2))

        tmp_coord[0,:,1] = y_bottom # y coordinates of bottom boundary
        tmp_coord[1,:,1] = y_top # y coordinate of top boundary
        tmp_coord[0,:,0] = jacob_x[ix_elem,0] + jacob_x[ix_elem,1]*zz # x coordinate of bottom boundary
        tmp_coord[1,:,0] = tmp_coord[0,:,0] # x coordinate of top boundary
        tmp_B = K.constant(tmp_coord[0,:,:]) # shape: (n_quads,2)
        tmp_T = K.constant(tmp_coord[1,:,:]) # shape: (n_quads,2)
        coord_B.append(tmp_B)
        coord_T.append(tmp_T)

        data_B = the_anal_soln(tmp_coord[0,:,:])
        data_T = the_anal_soln(tmp_coord[1,:,:])
        data_B_tensor = K.constant(data_B) # shape: (n_quads,2)
        data_T_tensor = K.constant(data_T) # shape: (n_quads,2)
        bc_data_B.append(data_B_tensor)
        bc_data_T.append(data_T_tensor)

    bc_L = (coord_L, bc_data_L)
    bc_R = (coord_R, bc_data_R)
    bc_B = (coord_B, bc_data_B)
    bc_T = (coord_T, bc_data_T)

    return (bc_L, bc_R, bc_B, bc_T)

def output_data_tecplot(file, nx_elem, ny_elem, n_pred, x_coord, y_coord, v_pred, v_true):
    with open(file, 'w') as fileout:
        fileout.write("variables = x, y, value-predict, value-true, error\n")
        for ix in range(nx_elem):
            for iy in range(ny_elem):
                ie = ix*ny_elem + iy
                fileout.write("zone T=\"(%d,%d)\", I=%d, J=%d, K=1, DATAPACKING=POINT\n" % (ix, iy, n_pred, n_pred))
                for i in range(n_pred):
                    for j in range(n_pred):
                        k = i*n_pred + j
                        fileout.write("%.14e %.14e %.14e %.14e %.14e\n" % (x_coord[ie,k], y_coord[ie,k],
                                                                v_pred[ie,k], v_true[ie,k],
                                                                np.abs(v_pred[ie,k]-v_true[ie,k])))

def save_history(hist_file, hist_obj):
    """
    save training history
    :param hist_file: file name to store history data
    :param train_hist: history object from keras.model.fit()
    :return:
    """
    # convert the hist.history dict to pandas DataFrame
    hist_df = pd.DataFrame(hist_obj.history)

    # save to csv:
    with open(hist_file, mode='w') as f:
        hist_df.to_csv(f)

def save_run_param(param_file, param_data):
    """
    save parameters for this run
    :param param_file: file name to output to
    :param param_data: tuple, (domain, dnn_struct, mode_quad, to_read_init_weight,
                                _epsilon, Ck_cont, train_param, lr, elapsed_time)
                              where domain = (domain_x, domain_y)
                                    dnn_struct = (layers, activations)
                                    mode_quad = (n_modes, n_quads)
                                    to_read_init_weight : True or False
                                    _epsilon : epsilon
                                    Ck_cont : Ck continuity
                                    train_param = (batch_size, early_stop_patience, max_epochs)
                                    lr = (default_learning_rate, actual_learning_rate)
                                    elapsed_time = (training_elapsed_time, prediction_elapsed_time)
    :return:
    """
    domain, dnn_struct, mode_quad, to_read_init_weight, eps, ck, train_param, lr, elap_time, soln_param = param_data
    domain_x, domain_y = domain
    Layers, Activ = dnn_struct
    n_modes, n_quads = mode_quad
    batch_size, patience, max_epochs = train_param
    default_lr, actual_lr = lr
    train_elapsed_time, prediction_elapsed_time = elap_time
    lambda_coeff, aa = soln_param

    with open(param_file,"w") as fileout:
        fileout.write("domain_x = " + str(domain_x) + "\n")
        fileout.write("domain_y = " + str(domain_y) + "\n")
        fileout.write("layers = " + str(Layers) + "\n")
        fileout.write("activations = " + str(Activ) + "\n")
        fileout.write("N_mode = %d\n" % (n_modes))
        fileout.write("N_quad = %d\n" % (n_quads))
        fileout.write("to_read_init_weight = " + str(to_read_init_weight) + "\n")
        fileout.write("_epsilon = %.14e\n" % eps)
        fileout.write("Ck_continuity = %d\n" % ck)
        fileout.write("batch_size = %d\n" % batch_size)
        fileout.write("early_stop_patience = %d\n" % patience)
        fileout.write("maximum_epochs = %d\n" % max_epochs)
        fileout.write("default_learning_rate = %.12e\n" % default_lr)
        fileout.write("actual_learning_rate = %.12e\n" % actual_lr)
        fileout.write("training_elapsed_time (seconds) = %.14e\n" % train_elapsed_time)
        fileout.write("prediction_elapsed_time (seconds) = %.14e\n" % pred_elapsed_time)
        fileout.write("solution parameter: lambda = %.12e\n" % lambda_coeff)
        fileout.write("solution parameter: aa = %.12e\n" % aa)

if __name__ == '__main__':

    # files
    base_path = './data/C0Basis_localDNN'
    my_util.mkdir(base_path)
    problem = 'helm2d_dnn_'
    method = 'c0base'
    file_base = base_path + '/' + problem + method

    solution_file = file_base + '_soln.dat'
    parameter_file = file_base + '_param.dat'
    model_weight_file = file_base + '_weights.hd5'
    model_init_weight_file = file_base + '_init_weights.hd5'
    train_history_file = file_base + '_history.csv'

    # +++++++++++++++++++++++++++ #
    # parameters
    Ck_cont = CK
    Lambda_Coeff = LAMBDA_COEFF #np.float(2.0)

    # number of entries in input data: n_quads*n_elem
    batch_size = N_quad*N_quad

    early_stop_patience = 1000
    max_epochs = MAX_EPOCHS

    jacobian = calc_jacobian((domain_x, domain_y))
    # now jacobian contains the vector of jacobians
    jacob_x, jacob_y = jacobian

    bc_data = gen_bc_data(jacobian, N_quad)
    # boundary data
    elem_bound_info = gen_elem_bound_info(jacobian, N_quad)
    # now elem_bound_info contains element boundary information data

    # ++++++++++++++++++++++++++++++++ #
    # build model
    elem_param = (N_modes, N_quad, Lambda_Coeff, bc_data, jacobian, (domain_x,domain_y), Ck_cont)
    the_DNN = build_model(layers, activations, elem_param, elem_bound_info)
    if to_read_init_weight and my_util.is_file(model_init_weight_file):
        the_DNN.load_weights(model_init_weight_file)

    default_lr = get_learning_rate(the_DNN)
    print("Default learning rate: %f" % default_lr)
    if LR_factor>1.0 or LR_factor<1.0:
        LR = LR_factor*default_lr
        set_learning_rate(the_DNN, LR)
    else:
        LR = default_lr

    # +++++++++++++++++++++++++++++++++++++++ #
    # generate training data
    In_data = []
    Label_data = []

    zz, _ = calc_zw(N_quad) # polylib.zwgll(N_input)
    zz_x = np.reshape(zz, (N_quad,1,1)) # shape: (N_quad,1,1)
    zz_y = np.reshape(zz, (1,N_quad,1)) # shape: (1,N_quad,1)
    for i in range(Nx_elem):
        Ax = jacob_x[i,1]
        Bx = jacob_x[i,0]

        for j in range(Ny_elem):
            Ay = jacob_y[j,1]
            By = jacob_y[j,0]

            tmp_in = np.zeros((N_quad, N_quad, 2))
            tmp_in[:,:,0:1] = zz_x * Ax + Bx
            tmp_in[:,:,1:2] = zz_y * Ay + By
            tmp_in_data = np.reshape(tmp_in, (N_quad*N_quad,2)) # shape: (N_quad*N_quad,2)
            In_data.append(tmp_in_data)

            tmp_label = the_source_term(Lambda_Coeff, tmp_in_data) # shape: (N_quad*N_quad,1)
            Label_data.append(tmp_label)

    # +++++++++++++++++++++++++++++++++++++++++++ #
    # training
    print("training starting ...\n")
    from timeit import default_timer as timer
    begin_time = timer()

    early_stop = kclbk.EarlyStopping(monitor='loss', mode='min',
                                     verbose=1,
                                     patience=early_stop_patience,
                                     restore_best_weights=True)
    train_hist = the_DNN.fit(In_data, Label_data,
                            epochs=max_epochs, batch_size=batch_size, shuffle=False,
                            callbacks=[early_stop])

    end_time = timer()
    train_elapsed_time = end_time - begin_time
    print("Training time elapsed (seconds): %.14e" % (train_elapsed_time))

    # +++++++++++++++++++++++++++++++ #
    # save weights to file
    time_stamp = my_util.get_timestamp()
    save_weight_file = model_weight_file + time_stamp
    save_param_file = parameter_file + time_stamp
    save_init_weight_file = model_init_weight_file + time_stamp

    the_DNN.save_weights(save_weight_file)
    if to_read_init_weight == True:
        my_util.rename_file(model_init_weight_file, save_init_weight_file)
        my_util.copy_file(save_weight_file, model_init_weight_file)

    # save training history
    hist_file = train_history_file + time_stamp
    save_history(hist_file, train_hist)

    # +++++++++++++++++++++++++++++++ #
    # prediction
    N_pred = 100
    data_in = []
    for i in range(Nx_elem):
        x_coord_1 = np.linspace(domain_x[i], domain_x[i + 1], N_pred)
        x_coord = np.reshape(x_coord_1, (N_pred,1,1))

        for j in range(Ny_elem):
            y_coord_1 = np.linspace(domain_y[j], domain_y[j+1], N_pred)
            y_coord = np.reshape(y_coord_1, (1,N_pred,1))

            tmp_data = np.zeros((N_pred, N_pred, 2))
            tmp_data[:,:, 0:1] = x_coord
            tmp_data[:,:,1:2] = y_coord
            tmp_pred = np.reshape(tmp_data, (N_pred*N_pred,2))
            data_in.append(tmp_pred)

    pred_start_time = timer()
    Soln = the_DNN.predict(data_in) # list of N_elem arrays of shape: (N_pred*N_pred,1)
    pred_end_time = timer()
    pred_elapsed_time = pred_end_time - pred_start_time
    print("Prediction time elapsed (seconds): %.14e" % (pred_elapsed_time))

    soln_1 = np.stack(Soln, 0) # shape: (nx_elem*ny_elem, N_pred*N_pred, 1)
    soln_dnn = np.reshape(soln_1, (Nx_elem*Ny_elem, N_pred*N_pred))

    data_1 = np.stack(data_in, 0) # shape: (Nx_elem*Ny_elem, N_pred*N_pred, 2)
    xy_data_A = np.reshape(data_1, (Nx_elem*Ny_elem*N_pred*N_pred, 2))
    x_data = data_1[:,:,0]
    y_data = data_1[:,:,1]

    Exact_soln = the_anal_soln(xy_data_A) # shape: (Nx_elem*N_pred*Ny_elem*N_pred,1)
    soln_exact = np.reshape(Exact_soln, (Nx_elem*Ny_elem, N_pred*N_pred))

    soln_file = solution_file + time_stamp
    output_data_tecplot(soln_file, Nx_elem, Ny_elem, N_pred,
                x_data, y_data, soln_dnn, soln_exact)

    # ++++++++++++++++++++++++++++++++++++ #
    # save parameters
    param_data = ((domain_x,domain_y), (layers, activations), (N_modes,N_quad), to_read_init_weight,
                  _epsilon, Ck_cont, (batch_size, early_stop_patience, max_epochs), (default_lr,LR),
                  (train_elapsed_time,pred_elapsed_time), (Lambda_Coeff,aa))
    save_run_param(save_param_file, param_data)

    # ++++++++++++++++++++++++++++++++++++ #



