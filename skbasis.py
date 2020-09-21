"""
Routines for Sherwin-Karniadakis Basis functions
"""

import numpy as np
import polylib


def calc_basis(id, z):
    """
    Calculate SK basis values on given points
    :param id: identifies the mode
                id = 0:  0.5*(1-z)
                id = 1:  0.5*(1+z)
                id = k:  0.5*(1-z)*0.5*(1+z)J_{k-2}^{1,1}(z), for k>=2
                         where J_n^{1,1} is the Jacobi polynomial of degree n and (1,1)
    :param z: points to compute values on
    :return:
    """
    nz = len(z)
    sk_basis = np.zeros(nz)

    if id == 0:
        sk_basis = 0.5*(1.0-z)
    elif id == 1:
        sk_basis = 0.5*(1.0+z)
    else:
        n = id - 2
        jacobf = polylib.jacobf(z, n, 1, 1)
        phi_0 = 0.5*(1.0-z)
        phi_1 = 0.5*(1.0+z)
        sk_basis = phi_0 * phi_1 * jacobf

    return sk_basis


def calc_basis_deriv(id, z):
    """
    calculate derivatives of SK-basis function on given points
    :param id: identifies the mode of SK basis function
    :param z: coordinates to compute on
    :return: derivatives of SK basis on z
    """
    nz = len(z)
    skb_deriv = np.zeros(nz)

    if id == 0:
        skb_deriv = skb_deriv - 0.5
        return skb_deriv
    elif id == 1:
        skb_deriv = skb_deriv + 0.5
        return skb_deriv
    else:
        n = id - 2
        jacobf = polylib.jacobf(z, n, 1, 1)
        jacobf_deriv = polylib.jacobd(z, n, 1, 1)
        phi_0 = 0.5*(1.0-z)
        phi_1 = 0.5*(1.0+z)
        skb_deriv = -0.5*z*jacobf + phi_0*phi_1*jacobf_deriv
        return skb_deriv


if __name__ == '__main__':

    N_quad = 20
    z, _ = polylib.zwgll(N_quad)

    N_modes = 10
    phi = np.zeros((N_modes, N_quad))
    phi_deriv = np.zeros((N_modes, N_quad))

    for i in range(N_modes):
        phi[i, :] = calc_basis(i, z)
        phi_deriv[i, :] = calc_basis_deriv(i, z)
