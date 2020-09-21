"""
Implementation of Polylib in python
common routines for polynomial calculus and interpolation
"""

import numpy as np
from scipy.special import eval_jacobi
from scipy.special import roots_jacobi
from scipy.special import gamma

_EPS = 1.0e-13


def jacobf(z, n, alpha, beta):
    """
    evaluate Jacobi polynomial P_n^{alpha,beta} on a set of points
    :param z: numpy array, values of the points to evaluate Jacobi polynomial on
    :param n: order of Jacobi polynomial in P_n^{alpha,beta}
    :param alpha: parameters, alpha > -1
    :param beta: parameters, beta > -1
    :return: vector of Jacobi polynomial values on these points
    """
    return eval_jacobi(n, alpha, beta, z)


def jacobd(z, n, alpha, beta):
    """
    compute derivative of jacobi polynomial on given points
    :param z: numpy array, contains values of points to evaluate derivative on
    :param n: order of jacobi polynomial, integer, n>=0
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: array of derivative values on the points

    d/dz P_n^{alpha,beta} = 0.5*Gamma(alpha+beta+n+2)/Gamma(alpha+beta+n+1) * P_{n-1}^{alpha+1,beta+1}
    """
    num_pts = len(z)
    polyd = np.zeros(num_pts)

    if n <= 0:
        return polyd
    else:
        polyd = jacobf(z, n - 1, alpha + 1, beta + 1)
        polyd = 0.5 * (alpha + beta + n + 1) * polyd
        return polyd


def jacobd_2nd(z, n, alpha, beta):
    """
    compute 2nd derivative of jacobi polynomial on given points
    :param z: np array, value of points to compute derivatives on
    :param n: order of Jacobi polynomial
    :param alpha: parameter
    :param beta: parameter
    :return: values of 2nd derivative of jacobi polynomial
    """
    num_pts = len(z)
    polyd = np.zeros(num_pts)

    if n <= 1:
        return polyd
    else:
        coeff = 0.25 * (alpha + beta + n + 2) * (alpha + beta + n + 1)
        polyd = coeff * jacobf(z, n - 2, alpha + 2, beta + 2)
        return polyd


def jacobd_kth(k, z, n, alpha, beta):
    """
    compute k-th derive of jacobi polynomial P_n^{alpha,beta}
    :param k: derivative order, integer
    :param z: points to compute derivative on
    :param n: P_n^{alpha, beta}
    :param alpha: > -1
    :param beta: > -1
    :return: values of k-th derivative on given points
    """
    if k < 0:
        return None
    elif k == 0:
        return jacobf(z, n, alpha, beta)
    else:
        num_pts = len(z)
        polyd = np.zeros(num_pts)

        if n < k:
            return polyd
        else:
            coeff = gamma(alpha + beta + n + 1 + k) / (pow(2, k) * gamma(alpha + beta + n + 1))
            polyd = coeff * jacobf(z, n - k, alpha + k, beta + k)
            return polyd


def jacobz(n, alpha, beta):
    """
    compute zeros jacobi polynomial P_n^{alpha,beta}
    :param n: order of jacobi polynomial, integer
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: zeros of jacobi polynomials
    """
    if n <= 0:
        return None
    z, _ = roots_jacobi(n, alpha, beta)
    return z


def zwgj(n, alpha, beta):
    """
    compute Gauss-Jacobi quadrature points and weights associated with
    polynomial P_n^{alpha,beta}
    :param n: order of Jacobi polynomial
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: tuple (z,w), where z contains quadrature points, w contains weights
    """
    if n <= 0:
        return None, None
    else:
        return roots_jacobi(n, alpha, beta)


def zwgrj(n, alpha, beta):
    """
    compute Gauss-Jacobi Radau quadrature points and weights
    associated with Jacobi polynomials P_{n-1}^{alpha,beta}
    \int_{-1}^{1}(1-x)^{alpha}*(1+x)^{beta} f(x) dx = w[0]*f(-1) + sum_{i=1}^n w[i]*f(z[i])
    :param n: number of quadrature points, first quadrature point is -1
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: tuple, (z,w), where z[0:n-1] contains quadrature points, w[0:n-1] contains weights
    """
    if n <= 0:
        return None, None
    elif n == 1:
        return 0.0, 2.0
    else:
        z = np.zeros(n)
        z[0] = -1.0
        z[1:] = jacobz(n - 1, alpha, beta + 1)
        w = jacobf(z, n - 1, alpha, beta)

        fac = pow(2.0, alpha + beta) * gamma(alpha + n) * gamma(beta + n)
        fac = fac / (gamma(n) * (beta + n) * gamma(alpha + beta + n + 1))

        for i in range(n):
            w[i] = fac * (1.0 - z[i]) / (w[i] * w[i])
        w[0] = w[0] * (beta + 1.0)

        return z, w


def zwglj(n, alpha, beta):
    """
    compute Gauss-Labotto-Jacobi quadrature points and weights associated
    with Jacobi polynomial P_n^{alpha,beta}
    \int_{-1}^{1}(1-x)^{alpha} * (1+x)^{beta} f(x) dx = w[0]*f(-1) * w[n-1]*f(1) + sum_{i=1}^{n-2} w[i]*f(z[i])
    :param n: number of quadrature points
    :param alpha: parameter, alpha>-1
    :param beta: parameter, beta > -1
    :return: tuple (z,w), where z[0,n-1] contains quadrature points, w[0:n-1] contains weights
    """
    if n <= 0:
        return (None, None)
    elif n == 1:
        return (0.0, 2.0)
    else:
        z = np.zeros(n)
        z[0] = -1.0
        z[n - 1] = 1.0
        z[1:n - 1] = jacobz(n - 2, alpha + 1, beta + 1)
        w = jacobf(z, n - 1, alpha, beta)

        fac = pow(2.0, alpha + beta + 1) * gamma(alpha + n) * gamma(beta + n)
        fac = fac / ((n - 1) * gamma(n) * gamma(alpha + beta + n + 1))

        for i in range(n):
            w[i] = fac / (w[i] * w[i])
        w[0] = w[0] * (beta + 1.0)
        w[n - 1] = w[n - 1] * (alpha + 1.0)

        return z, w


def dgj(z, n, alpha, beta):
    """
    compute derivative matrix associated with the n-th order Lagrangian interpolants
    through the n Gauss-Jacobi points z
    :param z: Gauss-Jacobi quadrature points of order n, of P_n^{alpha,beta}, these points are
              zeros of P_n^{alpha,beta}
    :param n: number of quadrature points
    :param alpha: parameter of Jacobi poly
    :param beta: parameter of
    :return: n x n derivative matrix D_{ij} such that du/dz = D_{ij} * u_j evaluated at z=z_i
    """
    if n <= 0:
        return None
    else:
        D = np.zeros((n, n))

        pd = jacobd(z, n, alpha, beta)  # derivative of P_n^{alpha,beta} on these points
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = pd[i] / (pd[j] * (z[i] - z[j]))
                else:
                    D[i][j] = (alpha - beta + (alpha + beta + 2.0) * z[i]) / (2.0 * (1.0 - z[i] * z[i]))

        return D


def dgrj(z, n, alpha, beta):
    """
    compute derivative matrix associated with n-th order Lagrangian interpolants on
    Gauss-Radau-Jacobi points given in z
    :param z: Gauss-Radau-Jacobi quadrature points of order n, vector, dimension [n]
              z[0]=-1, z[1:n-1] are zeros of P_{n-1}^{alpha,beta+1}
    :param n: number of quadrature points
    :param alpha: parameter, alpha>-1
    :param beta: parameter, beta>-1
    :return: derivative matrix D[n,n], such that du/dz = D_ij * u_j at z=z_i
    """
    if n <= 0:
        return None

    pd = np.zeros(n)
    D = np.zeros((n, n))

    pd[0] = pow(-1.0, n - 1) * gamma(n + beta + 1) / (gamma(n) * gamma(beta + 2))
    pd[1:] = jacobd(z[1:], n - 1, alpha, beta + 1)
    for i in range(1, n):
        pd[i] = pd[i] * (1.0 + z[i])

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = pd[i] / (pd[j] * (z[i] - z[j]))
            else:
                if i == 0:
                    D[i, j] = -(n + alpha + beta + 1.0) * (n - 1.0) / (2.0 * (beta + 2.0))
                else:
                    D[i, j] = (alpha - beta + 1.0 + (alpha + beta + 1.0) * z[i]) / (2.0 * (1.0 - z[i] * z[i]))

    return D


def dglj(z, n, alpha, beta):
    """
    compute derivative matrix associated with n-th order Lagrangian interpolants
    through the m Gauss-Lobatto-Jacobi quadrature points given in z
    :param z: Gauss-Lobatto-Jacobi quadrature points of order n, z[0]=-1, z[n-1]=1
              z[1:n-1] are zeros of P_{n-2}^{alpha+1,beta+1}
    :param n: number of quadrature points
    :param alpha: parameter, alpha>-1
    :param beta: parameter, beta>-1
    :return: derivative matrix D[n,n], such that du/dz = D_ij*u_j evaluated at z=z_i
    """
    if n <= 0:
        return None

    pd = np.zeros(n)
    D = np.zeros((n, n))

    pd[0] = 2.0 * pow(-1.0, n) * gamma(n + beta) / (gamma(n - 1) * gamma(beta + 2))
    pd[1:n - 1] = jacobd(z[1:n - 1], n - 2, alpha + 1, beta + 1)
    for i in range(1, n - 1):
        pd[i] = pd[i] * (1.0 - z[i]) * (1.0 + z[i])
    pd[n - 1] = -2.0 * gamma(n + alpha) / (gamma(n - 1) * gamma(alpha + 2))

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = pd[i] / (pd[j] * (z[i] - z[j]))
            else:
                if i == 0:
                    D[i, j] = (alpha - (n - 1.0) * (n + alpha + beta)) / (2.0 * (beta + 2.0))
                elif i == n - 1:
                    D[i, j] = -(beta - (n - 1.0) * (n + alpha + beta)) / (2.0 * (alpha + 2.0))
                else:
                    D[i, j] = (alpha - beta + (alpha + beta) * z[i]) / (2.0 * (1.0 - z[i]) * (1.0 + z[i]))

    return D


def hgj(id, z, zgj, n, alpha, beta):
    """
    compute the id-th Lagrangian interpolant L_{id}(x) through the n Gauss-Jacobi quadrature points,
    which are given in zgj, on the points given in z, i.e. L_{id}(x)|_{x=z}
    :param id: index of n-th order Lagrangian interpolant
    :param z: points to evaluate on, vector, numpy array
    :param zgj: quadrature points values, zeros of P_n^{alpha,beta}, np array of dimension n
    :param n: number of quadrature points
    :param alpha: parameter, alpha>-1
    :param beta: parameter, beta>-1
    :return: values of Lagrangian interpolant on given points
    """
    num_pts = len(z)
    h = np.zeros(num_pts)

    zi = np.zeros(1)
    zi[0] = zgj[id]

    pd = jacobd(zi, n, alpha, beta)
    p = jacobf(z, n, alpha, beta)

    for i in range(num_pts):
        dz = z[i] - zi[0]
        if np.abs(dz) < _EPS:
            h[i] = 1.0
        else:
            h[i] = p[i] / (pd[0] * dz)

    return h


def hgrj(id, z, zgrj, n, alpha, beta):
    """
    compute the id-th Lagrangian interpolant L_{id}(x) through the n Gauss-Radau-Jacobi quadrature points,
    which are given in zgj, on the points given in z, i.e. L_{id}(x)|_{x=z}
    :param id:
    :param z:
    :param zgrj:
    :param n:
    :param alpha:
    :param beta:
    :return:
    """
    num_pts = len(z)
    h = np.zeros(num_pts)

    zi = np.zeros(1)
    zi[0] = zgrj[id]

    p1 = jacobf(zi, n - 1, alpha, beta + 1)
    pd1 = jacobd(zi, n - 1, alpha, beta + 1)
    p = jacobf(z, n - 1, alpha, beta + 1)

    for i in range(num_pts):
        dz = z[i] - zi[0]
        if np.abs(dz) < _EPS:
            h[i] = 1.0
        else:
            h[i] = (1.0 + z[i]) * p[i] / (dz * (p1[0] + (1.0 + zi[0]) * pd1[0]))

    return h


def hglj(id, z, zglj, n, alpha, beta):
    """
    compute id-th Lagrangian interpolant through the n Gauss-Lobatto-Jacobi quadrature
    points, which are given in zglj, on the points given in z, i.e. L_{id}(x)|_{x=z}
    :param id: index of Lagrangian interpolant
    :param z: points to evaluate on, np array
    :param zglj: Gauss-Lobatto-Jacobi qaudrature points
    :param n: number of quadrature points
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: values of Lagrangian interpolant
    """
    num_pts = len(z)
    h = np.zeros(num_pts)

    zi = np.zeros(1)
    zi[0] = zglj[id]

    p1 = jacobf(zi, n - 2, alpha + 1, beta + 1)
    pd1 = jacobd(zi, n - 2, alpha + 1, beta + 1)
    p = jacobf(z, n - 2, alpha + 1, beta + 1)

    for i in range(num_pts):
        dz = z[i] - zi[0]
        if np.abs(dz) < _EPS:
            h[i] = 1.0
        else:
            coeff = -2.0 * zi[0] * p1[0] + (1.0 - zi[0]) * (1.0 + zi[0]) * pd1[0]
            h[i] = (1.0 - z[i]) * (1.0 + z[i]) * p[i] / (dz * coeff)

    return h


def igjm(zm_new, nz, zgj, alpha, beta):
    """
    compute interpolation matrix from Gauss-Jacobi quadrature points to
    another set of points. original Gauss-Jacobi points of order nz are given in zgj.
    coordinates of new points are given by zm_new
    :param zm_new: values of new points to interpolate to
    :param nz: number of original Gauss-Jacobi quadrature points
    :param zgj: Gauss-Jacobi quadrature points, numpy array, dimension: [nz]
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: interpolation matrix I_ij, such that,
             u[z_i] = sum_{j=1}^{nz} I_{ij} * u(x_j), for i=0, ..., len(zm_new)-1
             where x_j, j=0, ..., nz-1 are original Gauss-Jacobi points
                   I_{ij} = Lagrange_polynomial_{j}(z_i)
    """
    mz = len(zm_new)
    Imat = np.zeros((mz, nz))

    for j in range(nz):
        Imat[:, j] = hgj(j, zm_new, zgj, nz, alpha, beta)

    return Imat


def igrjm(zm_new, nz, zgrj, alpha, beta):
    """
    compute interpolation matrix from Gauss-Radau-Jacobi quadrature points to
    another set of points. original Gauss-Radau-Jacobi points of order nz are given in zgrj.
    coordinates of new points are given by zm_new
    :param zm_new: values of new points to interpolate to
    :param nz: number of original Gauss-Radau-Jacobi quadrature points
    :param zgrj: Gauss-Radau-Jacobi quadrature points, numpy array, dimension: [nz]
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: interpolation matrix I_ij, such that,
             u[z_i] = sum_{j=1}^{nz} I_{ij} * u(x_j), for i=0, ..., len(zm_new)-1
             where x_j, j=0, ..., nz-1 are original Gauss-Radau-Jacobi points
                   I_{ij} = Lagrange_polynomial_{j}(z_i)
    """
    mz = len(zm_new)
    Imat = np.zeros((mz, nz))

    for j in range(nz):
        Imat[:, j] = hgrj(j, zm_new, zgrj, nz, alpha, beta)

    return Imat


def igljm(zm_new, nz, zglj, alpha, beta):
    """
    compute interpolation matrix from Gauss-Lobatto-Jacobi quadrature points to
    another set of points. original Gauss-Lobatto-Jacobi points of order nz are given in zglj.
    coordinates of new points are given by zm_new
    :param zm_new: values of new points to interpolate to
    :param nz: number of original Gauss-Lobatto-Jacobi quadrature points
    :param zglj: Gauss-Lobatto-Jacobi quadrature points, numpy array, dimension: [nz]
    :param alpha: parameter, alpha > -1
    :param beta: parameter, beta > -1
    :return: interpolation matrix I_ij, such that,
             u[z_i] = sum_{j=1}^{nz} I_{ij} * u(x_j), for i=0, ..., len(zm_new)-1
             where x_j, j=0, ..., nz-1 are original Gauss-Lobatto-Jacobi points
                   I_{ij} = Lagrange_polynomial_{j}(z_i)
    """
    mz = len(zm_new)
    Imat = np.zeros((mz, nz))

    for j in range(nz):
        Imat[:, j] = hglj(j, zm_new, zglj, nz, alpha, beta)

    return Imat


""" Other routines for Legendre and Chebyshev polynomials
    Legendre polynomial, L_n(x) = JacobiPolynomial(n, alpha=1, beta=0)
    Chebyshev polynomial, T_n(x) = JacobiPolynomial(n, alpha=-0.5, beta=-0.5)
"""


def legendref(z, n):
    """
    evaluate Legendre polynomials on given points
    :param z: points to evaluate on
    :param n: order of Legendre polynomials
    :return: values on points
    """
    return jacobf(z, n, 0, 0)


def chebyshevf(z, n):
    """
    evaluate Chebyshev polynomials on given points
    :param z: points to evaluate on
    :param n: order of Chebyshev polynomials
    :return: values on points
    """
    return jacobf(z, n, -0.5, -0.5)


def legendred(z, n):
    """
    compute derivative of Legendre polynomial on given point
    :param z: points to evaluate on
    :param n: order or Legendre polynomial
    :return: values on points
    """
    return jacobd(z, n, 0, 0)


def chebyshevd(z, n):
    """
    compute derivative of Chebyshev polynomial on given points z
    :param z: points to evaluate on
    :param n: order of Chebyshev polynomial
    :return: values on points
    """
    return jacobd(z, n, -0.5, -0.5)


def zwgl(n):
    """
    zeros and weights of Gauss-Legendre quadrature
    :param n: order
    :return: tuple (z,w), where z[n] contains zeros, w[n] contains weights
    """
    return zwgj(n, 0, 0)


def zwgrl(n):
    """
    zeros and weights of Gauss-Radau-Legendre quadrature
    :param n: order
    :return: tuple (z,w) as (zeros, weights)
    """
    return zwgrj(n, 0, 0)


def zwgll(n):
    """
    zeros and weights of Gauss-Lobatto-Legendre quadrature
    :param n: order
    :return: tuple (z,w) as (zeros,weights)
    """
    return zwglj(n, 0, 0)


def zwgc(n):
    """
    zeros and weights of Gauss-Chebyshev quadrature
    :param n: order
    :return: (z,w) as (zeros,weights)
    """
    return zwgj(n, -0.5, -0.5)


def zwgrc(n):
    """
    zeros and weights of Gauss-Radau-Chebyshev quadrature
    :param n: order
    :return: (z,w) as (zeros,weights)
    """
    return zwgrj(n, -0.5, -0.5)


def zwglc(n):
    """
    zeros and weights of Gauss-Lobatto-Chebyshev quadrature
    :param n: order
    :return: (z,w) as (zeros,weights)
    """
    return zwglj(n, -0.5, -0.5)


def dgl(z, n):
    """
    derivative matrix for Gauss-Legendre quadrature points
    :param z: Gauss-Legendre quadrature points
    :param n: order
    :return: derivative matrix
    """
    return dgj(z, n, 0, 0)


def dgrl(z, n):
    """
    derivative matrix of Gauss-Radau-Legedre quadrature
    :param z: Gauss-Radau-Legedre quadrature points
    :param n: order
    :return: derivative matrix
    """
    return dgrj(z, n, 0, 0)


def dgll(z, n):
    """
    derivative matrix of Gauss-Lobatto-Legendre quadrature
    :param z: Gauss-Lobatto-Legendre quadrature
    :param n: order
    :return: derivative matrix
    """
    return dglj(z, n, 0, 0)


def dgc(z, n):
    """
    derivative matrix of Gauss-Chebyshev quadrature
    :param z: Gauss-Chebyshev quadratrure points
    :param n: order
    :return: derivative matrix
    """
    return dgj(z, n, -0.5, -0.5)


def dgrc(z, n):
    """
    derivative matrix of Gauss-Radau-Chebyshev quadrature
    :param z: Gauss-Radau-Chebyshev quadrature points
    :param n: order
    :return: derivative matrix
    """
    return dgrj(z, n, -0.5, -0.5)


def dglc(z, n):
    """
    derivative matrix of Gauss-Lobatto-Chebyshev quadrature
    :param z: Gauss-Lobatto-Chebyshev quadrature points
    :param n: order
    :return: derivative matrix
    """
    return dglj(z, n, -0.5, -0.5)


def hgl(id, z, zgl, n):
    """
    compute id-th Lagrangian interpolant through Gauss-Legendre points, on given points in z
    :param id: index of Lagrangian interpolant
    :param z: points to compute Lagrangian interpolant on
    :param zgl: Gauss-Legendre quadrature points
    :param n: number of quadrature points
    :return: values on these points
    """
    return hgj(id, z, zgl, n, 0, 0)


def hgrl(id, z, zgrl, n):
    """
    compute id-th Lagrangian interpolant through Gauss-Radau-Legendre points, on given points in z
    :param id: index of Lagrangian interpolant
    :param z: points to compute Lagrangian interpolant on
    :param zgrl: Gauss-Radau-Legendre quadrature points
    :param n: number of quadrature points
    :return: values on these points
    """
    return hgrj(id, z, zgrl, n, 0, 0)


def hgll(id, z, zgll, n):
    """
    compute id-th Lagrangian interpolant through Gauss-Lobatto-Legendre points, on given points in z
    :param id: index of Lagrangian interpolant
    :param z: points to compute Lagrangian interpolant on
    :param zgll: Gauss-Lobatto-Legendre quadrature points
    :param n: number of quadrature points
    :return: values on these points
    """
    return hglj(id, z, zgll, n, 0, 0)


def hgc(id, z, zgc, n):
    """
    compute id-th Lagrangian interpolant through Gauss-Chebyshev points, on given points in z
    :param id: index of Lagrangian interpolant
    :param z: points to compute Lagrangian interpolant on
    :param zgc: Gauss-Chebyshev quadrature points
    :param n: number of quadrature points
    :return: values on these points
    """
    return hgj(id, z, zgc, n, -0.5, -0.5)


def hgrc(id, z, zgrc, n):
    """
    compute id-th Lagrangian interpolant through Gauss-Radau-Chebyshev points, on given points in z
    :param id: index of Lagrangian interpolant
    :param z: points to compute Lagrangian interpolant on
    :param zgrc: Gauss-Radau-Chebyshev quadrature points
    :param n: number of quadrature points
    :return: values on these points
    """
    return hgrj(id, z, zgrc, n, -0.5, -0.5)


def hglc(id, z, zglc, n):
    """
    compute id-th Lagrangian interpolant through Gauss-Lobatto-Chebyshev points, on given points in z
    :param id: index of Lagrangian interpolant
    :param z: points to compute Lagrangian interpolant on
    :param zglc: Gauss-Lobatto-Chebyshev quadrature points
    :param n: number of quadrature points
    :return: values on these points
    """
    return hglj(id, z, zglc, n, -0.5, -0.5)


def iglm(zm_new, nz, zgl):
    """
    interpolation matrix from Gauss-Legendre points to another set of points
    :param zm_new: new set of points
    :param nz: number of Gauss-Legendre points
    :param zgl: Gauss-Legendre points
    :return: interpolation matrix
    """
    return igjm(zm_new, nz, zgl, 0, 0)


def igrlm(zm_new, nz, zgrl):
    """
    interpolation matrix from Gauss-Radau-Legendre points to another set of points
    :param zm_new: new set of points
    :param nz: number of Gauss-Radau-Legendre points
    :param zgrl: Gauss-Radau-Legendre points
    :return: interpolation matrix
    """
    return igrjm(zm_new, nz, zgrl, 0, 0)


def igllm(zm_new, nz, zgll):
    """
    interpolation matrix from Gauss-Lobatto-Legendre points to another set of points
    :param zm_new: new set of points
    :param nz: number of Gauss-Lobatto-Legendre points
    :param zgll: Gauss-Lobatto-Legendre points
    :return: interpolation matrix
    """
    return igljm(zm_new, nz, zgll, 0, 0)


def igcm(zm_new, nz, zgc):
    """
    interpolation matrix from Gauss-Chebyshev points to another set of points
    :param zm_new: new set of points
    :param nz: number of Gauss-Shebyshev points
    :param zgc: Gauss-Chebyshev points
    :return: interpolation matrix
    """
    return igjm(zm_new, nz, zgc, -0.5, -0.5)


def igrcm(zm_new, nz, zgrc):
    """
    interpolation matrix from Gauss-Radau-Chebyshev points to another set of points
    :param zm_new: new set of points
    :param nz: number of Gauss-Radau-Shebyshev points
    :param zgrc: Gauss-Radau-Chebyshev points
    :return: interpolation matrix
    """
    return igrjm(zm_new, nz, zgrc, -0.5, -0.5)


def iglcm(zm_new, nz, zglc):
    """
    interpolation matrix from Gauss-Lobatto-Chebyshev points to another set of points
    :param zm_new: new set of points
    :param nz: number of Gauss-Lobatto-Shebyshev points
    :param zglc: Gauss-Lobatto-Chebyshev points
    :return: interpolation matrix
    """
    return igljm(zm_new, nz, zglc, -0.5, -0.5)


if __name__ == '__main__':

    num_pts = 4
    order = 3
    alpha = 0
    beta = 0

    x = 2.0 * np.random.rand(num_pts) - 1.0
    y = jacobf(x, order, alpha, beta)
    y_deriv = jacobd(x, order, alpha, beta)
    z = jacobz(order, alpha, beta)

    print(x)
    print(y)
    print(y_deriv)
    print(z)

    # test for integrals
    def f(xi):
        return pow(xi, 6)


    z_lj, w_lj = zwgrj(num_pts, alpha, beta)
    func = f(z_lj)

    res = 0.0
    for i in range(num_pts):
        res = res + w_lj[i] * func[i]
    print("numerical integral result = %.15e" % (res))
