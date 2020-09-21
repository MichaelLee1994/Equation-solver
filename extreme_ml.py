import numpy as np
import numpy.polynomial.legendre as leg
from scipy import integrate
from typing import List, Tuple, Callable
from matplotlib import pyplot as plt
from scipy.optimize import least_squares


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2)).flatten()[0]

np.random.seed(1234)


def block_diagonal(x: np.ndarray, offset: int = 0) -> np.ndarray:
    """
    build a block diagonal matrix based on x: (num_matrices, m, n)
    @param offset: (n-offset) indents of the next block
    @param x: a 3rd-order tensor (num_matrices, m, n)
    @return: block diagonal matrix (num_matrices*m, num_matrices*(n-offset)+offset)
    """
    if len(x.shape) != 3:
        raise ValueError("x should be a 3rd-order tensor. Got {}.".format(x.shape))

    shape = x.shape
    nrow = shape[1]
    ncol = shape[2]

    row_index = np.repeat(np.arange(nrow), ncol)
    col_index = np.tile(np.arange(ncol), nrow)

    row_index = np.concatenate([row_index + k * nrow for k in range(shape[0])])
    col_index = np.concatenate([col_index + k * (ncol - offset) for k in range(shape[0])])

    block = np.zeros((shape[0] * nrow, shape[0] * (ncol - offset) + offset))
    block[(row_index, col_index)] = x.flatten()

    return block


def vdp1(_y0: Tuple[float, float], _mu: float) \
        -> Tuple[Callable[[float, int, str], np.ndarray],
                 Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                 Tuple[float, Tuple[float, float]]]:
    """
    ver der Pol equation: y''-mu*(1-y^2)*y'+y=0
    @param _y0: initial condition (y0, dy0)
    @param _mu: a scalar parameter
    @return:
    """
    if _mu < 0:
        raise ValueError("mu has to be non-negative.")

    def loss_term(x: np.ndarray, y: np.ndarray, dy: np.ndarray, d2y: np.ndarray) -> np.ndarray:
        return d2y - _mu * (1 - y ** 2) * dy + y

    def _van_der_pol(x: np.ndarray, y):
        """
        change into first-order ODEs
        y0'=y1
        y1'=mu*(1-y0^2)*y1-y0
        """
        return np.array([y[1], (1 - y[0] ** 2) * y[1] - y[0]])

    def solution_ode45(end: float, points: int, method: str = "dopri5", clip: bool = True) -> np.ndarray:
        """
        numerical method to solve van der Pol equation's initial problem
        @param clip: whether to only keep y0, by default, only y0
        @param end: end of the interval
        @param points: number of points to be computed
        @param method: the numerical method, by default ode45
        @return: y and dy value at designated points, shape: (points, 2)
        """
        t0, t1 = 0.0, end  # start and end
        t = np.linspace(t0, t1, points)  # the points of evaluation of solution
        y = np.zeros((len(t), len(_y0)))  # array for solution
        y[0, :] = _y0
        r = integrate.ode(_van_der_pol).set_integrator(method)  # choice of method: ode45
        r.set_initial_value(_y0, t0)  # initial values
        for i in range(1, t.size):
            y[i, :] = r.integrate(t[i])  # get one more value, add it to the array
            if not r.successful():
                raise RuntimeError("Could not integrate")

        result = y[:, 0:1] if clip else y
        return result

    return solution_ode45, loss_term, (0.0, _y0)


def vdp_st(_mu: float):
    if _mu < 0:
        raise ValueError("mu has to be non-negative.")

    def _solution(_x):
        return _x * np.sin(_x)

    def _solution_x(_x):
        return np.sin(_x) + _x * np.cos(_x)

    def solution_wrapper(end, points, **kwargs):
        x0, x1 = 0.0, end
        x = np.linspace(x0, x1, points)
        return _solution(x)

    def source_term(x):
        f = 2*np.cos(x) - _mu*(1 - np.square(x*np.sin(x))) * (np.sin(x) + x*np.cos(x))
        return f

    def jocobian(y, dydx, dydw, d2ydxdw, d3ydx2dw):
        return d3ydx2dw - _mu * ((1 - 2 * y) * dydw * dydx + (1 - y ** 2) * d2ydxdw) + dydw

    def loss_term(x, y, dy, d2y):
        left = d2y - _mu*(1 - y**2)*dy+y
        right = source_term(x)
        return left - right

    init = (_solution(0.0), _solution_x(0.0))

    return solution_wrapper, loss_term, jocobian, (0.0, init)


def non_linear_spring(_w: float, _beta: float, start: float):
    if _beta < 0:
        raise ValueError("the spring is NOT stable when beta is negative!")

    def _solution(_x):
        return _x * np.sin(_x)

    def _solution_x(_x):
        return np.sin(_x) + _x * np.cos(_x)

    def source_term(x):
        # f = 2*np.cos(x) - (1-_w*_w)*_solution(x) + _beta*(_solution(x)**3)
        f = 2 * np.cos(x) - (1 - _w * _w) * _solution(x) + _beta * np.sin(_solution(x))
        return f

    def loss_term(x, y, dy, d2y):
        # left = d2y + _w*_w*y + _beta*(y**3)
        left = d2y + _w*_w*y + _beta*(np.sin(y))
        right = source_term(x)
        return left - right

    def solution_wrapper(end, points, **kwargs):
        x0, x1 = start, end
        x = np.linspace(x0, x1, points)
        return _solution(x)

    def jocobian(y, dydx, dydw, d2ydxdw, d3ydx2dw):
        return d3ydx2dw + _w * _w * dydw + _beta * np.cos(y) * dydw
        # return d3ydx2dw + _w*_w*dydw + 3 * _beta * (y ** 2) * dydw

    init = (_solution(start), _solution_x(start))

    return solution_wrapper, loss_term, jocobian, (start, init)


def auto_ode1d(_y0: float) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray],
                                    Tuple[float, float]]:
    """
    autonomous ode: y'=y*(2-y), y(0)=y0
    @param _y0: initial condition
    @return: tuple of solution function, source_term function and initial condition
    """
    def solution_equilibrium0(x: np.ndarray) -> np.ndarray:
        y = np.zeros(x.shape)
        return y

    def solution_equilibrium2(x: np.ndarray) -> np.ndarray:
        y = 2.0 * np.ones(x.shape)
        return y

    def solution_reg(x: np.ndarray) -> np.ndarray:
        c = 2.0 / _y0 - 1
        y = 2.0 / (1.0 + c * np.exp(-2.0 * x))
        return y

    def source_term(x, y):
        return y * (2.0 - y)

    if _y0 == 0:
        solution = solution_equilibrium0
    elif _y0 == 2:
        solution = solution_equilibrium2
    else:
        solution = solution_reg

    return solution, source_term, (0.0, _y0)


def test_ode(_y0: float) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray],
                                  Tuple[float, float]]:
    """
    ode equation: y'=xy, y(0)=y0
    @param _y0: initial condition
    @return: tuple of solution function, source_term function and initial condition
    """
    def solution_equilibrium(x: np.ndarray) -> np.ndarray:
        y = np.zeros(x.shape)
        return y

    def solution_reg(x: np.ndarray) -> np.ndarray:
        c = _y0
        y = c * np.exp(0.5 * np.square(x))
        return y

    def source_term(x, y):
        return x * y

    if _y0 == 0:
        solution = solution_equilibrium
    else:
        solution = solution_reg

    return solution, source_term, (0.0, _y0)


def test_simple(_y0: float) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     Tuple[float, float]]:
    """
    ode equation: y'=x*sin(x), with initial condition y(0)=y0
    @param _y0: initial condition
    @return: tuple of solution function, source_term function and initial condition
    """
    def solution_reg(x: np.ndarray) -> np.ndarray:
        c = _y0
        y = -x * np.cos(x) + np.sin(x) + c
        return y

    def source_term(x, y):
        return x * np.sin(x)

    solution = solution_reg

    return solution, source_term, (0.0, _y0)


def activation(name: str) -> Callable[[int], Callable[[np.ndarray], np.ndarray]]:
    def _tanh_wrapper(order: int = 0) -> Callable[[np.ndarray], np.ndarray]:
        def _tanh(x):
            return np.tanh(x)

        def _tanh_d1(x):
            return 1.0 - np.square(np.tanh(x))

        def _tanh_d2(x):
            return 2 * np.tanh(x) * (np.square(np.tanh(x)) - 1)

        if order == 0:
            return _tanh
        elif order == 1:
            return _tanh_d1
        elif order == 2:
            return _tanh_d2
        else:
            raise NotImplementedError("No more than 2nd derivative. Got %d." % order)

    lower_name = name.lower()
    if lower_name == 'tanh':
        activation_wrapper = _tanh_wrapper
    else:
        raise NotImplementedError("Activation Function available: tanh.")

    return activation_wrapper


def initialize_nn(num_elem: int, width: int, std: float,
                  dim_x: int = 1, dim_y: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    def _xavier_initialization(s: float, size: tuple) -> np.ndarray:
        # np.random.seed(1234)
        fan_in = size[-2]
        fan_out = size[-1]
        # xavier_stddev = np.sqrt(2 / (fan_in + fan_out))
        gen = np.random.uniform(low=-s, high=s, size=size)
        # gen = np.random.normal(scale=s, size=size)
        return gen

    w0 = _xavier_initialization(std, (num_elem, dim_x, width))
    w1 = _xavier_initialization(std, (num_elem, width, dim_y))
    b0 = np.random.normal(size=1)[0]
    return [w0, w1], [b0]


class ExtremeLearning:
    def __init__(self, num_quadrature, width, partition, equation, activation_name, init_para):
        self.partition = partition
        self.lower_bound = partition[0]
        self.upper_bound = partition[-1]
        self.num_elements = len(partition) - 1
        # if self.num_elements > 1:
        #     raise NotImplementedError("Only support one element now.")
        self.solution, self.loss_term, *jacob, (self.x0, self.y0) = equation
        self.jacobian = jacob[0]
        self.width = width
        self.num_quadrature = num_quadrature
        self.xi, self.q_weights = leg.leggauss(num_quadrature)
        self.q_weights = self.q_weights.reshape((1, -1, 1))  # shape: (1, Q, 1)

        # self.x_input = self.multi_linear_map()  # shape: (M, Q, 1)
        self.x_input = self.uniform_x(num_quadrature)
        # self.x_input = self.rand_x(num_quadrature)

        # need transposing to re-index!
        self.key_bd = np.array([partition[:-1], partition[1:]]).T.reshape((self.num_elements, 2, 1))
        aa = self.key_bd[:, 0:1, 0:1]
        bb = self.key_bd[:, 1:2, 0:1]
        self.element_c = 0.5 * (bb - aa)  # shape: (M, 1, 1)
        self.element_inv_c = 2 / (bb - aa)  # shape: (M, 1, 1)
        self.xi_c = -(aa+bb) / (bb-aa)
        assert self.xi_c.shape == (self.num_elements, 1, 1)
        self.weights, self.biases = initialize_nn(self.num_elements, self.width, init_para)
        self.act_wrapper = activation(activation_name)

    def multi_linear_map(self) -> np.ndarray:
        def _linear_map(xi, lower, upper):
            x = (upper + lower) / 2.0 + (upper - lower) * xi / 2.0
            return x.reshape((-1, 1))

        mapped_x = np.empty([self.num_elements, self.num_quadrature, 1])
        for i in range(self.num_elements):
            mapped_x[i, :] = _linear_map(self.xi, self.partition[i], self.partition[i + 1])
        return mapped_x

    def uniform_x(self, num):
        x = np.zeros((self.num_elements, num, 1))
        for i in range(self.num_elements):
            x[i, :, 0] = np.linspace(self.partition[i], self.partition[i+1], num)
        self.q_weights = np.ones(self.q_weights.shape)
        return x

    def rand_x(self, num):
        x = np.zeros((self.num_elements, num, 1))
        for i in range(self.num_elements):
            x[i, :, 0:1] = (self.partition[i-1] - self.partition[i]) * np.random.random_sample((num, 1)) \
                           + self.partition[i]
        self.q_weights = np.ones(self.q_weights.shape)
        return x

    def turbulent_training(self, method: str, max_epoch: int, turb_range: float = 1.0, loss_tol: float = 1e-2) \
            -> Tuple[str, float]:
        def _turbulence(r=2.0):
            w = self.weights[-1]   # shape: (num_elem, width, 1)
            random_v = np.random.uniform(low=-r, high=r, size=w.shape)
            return random_v.flatten()

        epoch = 1
        x0 = np.zeros(self.weights[-1].shape).flatten()
        while True:
            error = getattr(self, method)(x0=x0)
            if error < loss_tol:
                return "loss", error
            elif epoch > max_epoch:
                return "max", error
            else:
                epoch += 1
                pre_x0 = x0
                x0 = _turbulence(turb_range)
                print("Epoch: %d, Check --> %.4f\n" % (epoch, rmse(pre_x0, x0)))

    def neural_net(self, x_input: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray],
                   act: Callable[[np.ndarray], np.ndarray], ls: bool) -> np.ndarray:
        num_layers = len(weights)
        if num_layers != len(biases) + 1:
            raise ValueError("Difference between numbers of weights and biases should be one. #W: %d, #B: %d"
                             % (num_layers, len(biases)))

        output = self.element_inv_c * x_input + self.xi_c
        for layer in range(num_layers - 1):
            output = act(np.matmul(output, weights[layer]) + biases[layer])

        if ls:
            return output
        # last linear layer
        output = np.matmul(output, weights[-1])
        return output

    def compute_ls_simple(self, activation_code: str = None) -> float:
        if activation_code is not None:
            self.act_wrapper = activation(activation_code)
        x1_d = self.neural_net(self.x_input, self.weights, self.biases, self.act_wrapper(1), True)
        x1 = self.neural_net(self.x_input, self.weights, self.biases, self.act_wrapper(0), True)
        mat_dy_cube = self.element_c * (np.sqrt(self.q_weights) * (self.weights[0] * x1_d))
        mat_f_cube = self.element_c * np.sqrt(self.q_weights) * self.loss_term(self.x_input, x1)
        assert mat_dy_cube.shape == (self.num_elements, self.num_quadrature, self.width)
        assert mat_f_cube.shape == (self.num_elements, self.num_quadrature, 1)
        # "flatten" cube
        mat_diag = block_diagonal(mat_dy_cube)
        assert mat_diag.shape == (self.num_elements * self.num_quadrature, self.num_elements * self.width)

        mat_b = np.concatenate([mat_f_cube.reshape((self.num_elements*self.num_quadrature, 1)),
                                np.array([[self.y0]])], axis=0)

        mat_cube_bd = self.neural_net(self.key_bd, self.weights, self.biases, self.act_wrapper(0), True)
        assert mat_cube_bd.shape == (self.num_elements, 2, self.width)
        mat_bd = mat_cube_bd.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                    (self.num_elements, 1))
        mat_init = mat_bd[0:1, :]
        mat_a = np.block([[mat_diag], [mat_init, np.zeros((1, (self.num_elements - 1) * self.width))]])
        if self.num_elements > 1:
            mat_coupling = block_diagonal(mat_bd[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                          offset=self.width)
            assert mat_coupling.shape == (self.num_elements-1, self.width*self.num_elements)
            mat_a = np.concatenate([mat_a, mat_coupling], axis=0)
            mat_b = np.concatenate([mat_b, np.zeros((self.num_elements - 1, 1))], axis=0)

        ls_solution, residual, rank, singular = np.linalg.lstsq(a=mat_a, b=mat_b, rcond=None)

        for e in range(self.num_elements):
            self.weights[-1][e] = ls_solution[e * self.width:(e + 1) * self.width, 0:1]

        return np.sqrt(np.mean(residual ** 2)).flatten()[0]

    def compute_ls(self, activation_code: str = None) -> float:
        # if self.num_elements > 1:
        #     raise NotImplementedError("No more than one element now.")
        if activation_code is not None:
            self.act_wrapper = activation(activation_code)
        x1_d = self.neural_net(self.x_input, self.weights, self.biases, self.act_wrapper(1), True)
        x1 = self.neural_net(self.x_input, self.weights, self.biases, self.act_wrapper(0), True)
        mat_d_cube = self.element_c * (np.sqrt(self.q_weights) * (self.weights[0] * x1_d))
        mat_f_cube = self.element_c * (np.sqrt(self.q_weights) * self.loss_term(self.x_input, x1))
        assert mat_d_cube.shape == (self.num_elements, self.num_quadrature, self.width)
        assert mat_f_cube.shape == (self.num_elements, self.num_quadrature, self.width)
        # flatten mat cube
        mat_diag = block_diagonal(mat_d_cube - mat_f_cube)

        mat_b = np.concatenate([np.zeros((self.num_elements*self.num_quadrature, 1)), np.array([[self.y0]])], axis=0)

        mat_cube_bd = self.neural_net(self.key_bd, self.weights, self.biases, self.act_wrapper(0), True)
        assert mat_cube_bd.shape == (self.num_elements, 2, self.width)
        mat_bd = mat_cube_bd.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                    (self.num_elements, 1))

        mat_init = mat_bd[0:1, :]
        mat_a = np.block([[mat_diag], [mat_init, np.zeros((1, (self.num_elements - 1) * self.width))]])
        if self.num_elements > 1:
            mat_coupling = block_diagonal(mat_bd[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                          offset=self.width)
            assert mat_coupling.shape == (self.num_elements-1, self.width*self.num_elements)
            mat_a = np.concatenate([mat_a, mat_coupling], axis=0)
            mat_b = np.concatenate([mat_b, np.zeros((self.num_elements - 1, 1))], axis=0)

        ls_solution, residual, rank, singular = np.linalg.lstsq(a=mat_a, b=mat_b, rcond=None)

        for e in range(self.num_elements):
            self.weights[-1][e] = ls_solution[e * self.width:(e + 1) * self.width, 0:1]

        return np.sqrt(np.mean(residual ** 2)).flatten()[0]

    def compute_nls(self, activation_code: str = None) -> float:
        # if self.num_elements > 1:
        #     raise NotImplementedError("No more than one element now.")
        if activation_code is not None:
            self.act_wrapper = activation(activation_code)

        def _residual(weights: np.ndarray) -> np.ndarray:
            # weights shape: (M*N, 1)
            weights = weights.reshape((-1, 1))
            x1_d = self.neural_net(self.x_input, self.weights, self.biases, self.act_wrapper(1), True)
            x1 = self.neural_net(self.x_input, self.weights, self.biases, self.act_wrapper(0), True)
            mat_dy_cube = self.weights[0] * x1_d
            mat_y_cube = x1
            assert mat_dy_cube.shape == (self.num_elements, self.num_quadrature, self.width)
            assert mat_y_cube.shape == (self.num_elements, self.num_quadrature, self.width)

            dy = np.matmul(block_diagonal(mat_dy_cube), weights)
            y = np.matmul(block_diagonal(mat_y_cube), weights)
            q_weights = np.tile(self.q_weights.reshape((-1, 1)), (self.num_elements, 1))
            element_c = np.repeat(self.element_c.reshape((-1, 1)), self.num_quadrature, axis=0)
            assert y.shape == (self.num_elements*self.num_quadrature, 1)

            quad_loss = (element_c * (np.sqrt(q_weights) * (dy - 2 * y + y**2)))  # TODO: square sum?

            mat_cube_bd = self.neural_net(self.key_bd, self.weights, self.biases, self.act_wrapper(0), True)
            mat_bd = mat_cube_bd.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                        (self.num_elements, 1))

            init = mat_bd[0:1, :]
            assert init.shape == (1, self.width)
            bd_loss = np.matmul(init, weights[0:self.width, 0:1]) - np.array([self.y0])  # TODO: square sum?
            coupling_loss = np.zeros((1, 1))
            if self.num_elements > 1:
                mat_coupling = block_diagonal(mat_bd[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                              offset=self.width)
                assert mat_coupling.shape == (self.num_elements - 1, self.width * self.num_elements)
                coupling_loss = np.matmul(mat_coupling, weights)  # TODO: square sum?

            return np.concatenate([quad_loss, bd_loss, coupling_loss], axis=0).flatten()

        result = least_squares(_residual, x0=self.weights[-1].flatten(), verbose=2)
        solution = result.x

        for e in range(self.num_elements):
            self.weights[-1][e, :, 0] = solution[e * self.width:(e + 1) * self.width]

        return result.cost

    def compute_vdp(self, x0=None, activation_code: str = None) -> float:
        # if self.num_elements > 1:
        #     raise NotImplementedError("No more than one element now.")
        if activation_code is not None:
            self.act_wrapper = activation(activation_code)
        if x0 is None:
            x0 = np.zeros(self.weights[-1].shape).flatten()

        def _dx(x: np.ndarray):
            if x.shape[0] != self.num_elements:
                raise ValueError("Incompatible with #elements.")
            m = self.neural_net(x, self.weights, self.biases, self.act_wrapper(0), True)
            m_d = self.neural_net(x, self.weights, self.biases, self.act_wrapper(1), True)
            m_dd = self.neural_net(x, self.weights, self.biases, self.act_wrapper(2), True)

            assert m.shape[0] == self.num_elements

            ym = m
            ym_d = self.element_inv_c * self.weights[0] * m_d
            ym_dd = np.square(self.element_inv_c * self.weights[0]) * m_dd

            return ym, ym_d, ym_dd

        def _jacobian(weights: np.ndarray) -> np.ndarray:
            weights = weights.reshape((-1, 1))

            ym, dym, d2ym = _dx(self.x_input)

            d3ydx2dw = block_diagonal(d2ym)
            dydw = block_diagonal(ym)
            d2ydxdw = block_diagonal(dym)

            y = np.matmul(block_diagonal(ym), weights)
            dydx = np.matmul(block_diagonal(dym), weights)

            quad_block = self.jacobian(y, dydx, dydw, d2ydxdw, d3ydx2dw)

            mat_cube_c0, mat_cube_c1, _ = _dx(self.key_bd)
            bd_c0 = mat_cube_c0.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                       (self.num_elements, 1))
            bd_c1 = mat_cube_c1.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                       (self.num_elements, 1))

            init_block = np.block([[bd_c0[0:1, :], np.zeros((1, (self.num_elements-1)*self.width))],
                                   [bd_c1[0:1, :], np.zeros((1, (self.num_elements-1)*self.width))]])

            coupling_block = np.zeros((1, self.width))
            if self.num_elements > 1:
                coupling_c0 = block_diagonal(bd_c0[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                             offset=self.width)
                coupling_c1 = block_diagonal(bd_c1[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                             offset=self.width)
                assert coupling_c0.shape == (self.num_elements - 1, self.width * self.num_elements)
                assert coupling_c1.shape == (self.num_elements - 1, self.width * self.num_elements)
                coupling_block = np.concatenate([coupling_c0, coupling_c1], axis=0)

            return np.concatenate([quad_block, init_block, coupling_block], axis=0)

        def _residual(weights: np.ndarray) -> np.ndarray:
            # weights shape: (M*N, 1)
            weights = weights.reshape((-1, 1))
            ym, dym, d2ym = _dx(self.x_input)

            y = np.matmul(block_diagonal(ym), weights)
            dy = np.matmul(block_diagonal(dym), weights)
            d2y = np.matmul(block_diagonal(d2ym), weights)

            q_weights = np.tile(self.q_weights.reshape((-1, 1)), (self.num_elements, 1))
            element_c = np.repeat(self.element_c.reshape((-1, 1)), self.num_quadrature, axis=0)
            assert y.shape == (self.num_elements * self.num_quadrature, 1)

            x_input = self.x_input.reshape((-1, 1))
            quad_loss = (element_c * (q_weights * self.loss_term(x_input, y, dy, d2y)))

            mat_cube_c0, mat_cube_c1, _ = _dx(self.key_bd)
            bd_c0 = mat_cube_c0.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                       (self.num_elements, 1))
            bd_c1 = mat_cube_c1.reshape((self.num_elements * 2, self.width)) * np.tile(np.array([[1], [-1]]),
                                                                                       (self.num_elements, 1))

            init = np.concatenate([bd_c0[0:1, :], bd_c1[0:1, :]], axis=0)
            assert init.shape == (2, self.width)
            bd_loss = np.matmul(init, weights[0:self.width, 0:1]) - np.array(self.y0).reshape((-1, 1))
            coupling_loss = np.zeros((1, 1))
            if self.num_elements > 1:
                coupling_c0 = block_diagonal(bd_c0[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                             offset=self.width)
                coupling_c1 = block_diagonal(bd_c1[1:-1, :].reshape((self.num_elements - 1, 1, 2 * self.width)),
                                             offset=self.width)
                assert coupling_c0.shape == (self.num_elements - 1, self.width * self.num_elements)
                assert coupling_c1.shape == (self.num_elements - 1, self.width * self.num_elements)
                coupling_loss = np.concatenate([np.matmul(coupling_c0, weights), np.matmul(coupling_c1, weights)],
                                               axis=0)

            return np.concatenate([quad_loss, bd_loss, coupling_loss], axis=0).flatten()

        # result = least_squares(_residual, x0=self.weights[-1].flatten(), verbose=2)
        result = least_squares(_residual, x0=x0,
                               jac=_jacobian,
                               verbose=2)
        solution = result.x

        for e in range(self.num_elements):
            self.weights[-1][e, :, 0] = solution[e * self.width:(e + 1) * self.width]

        return result.cost

    def test(self, x: np.ndarray) -> np.ndarray:
        y = self.neural_net(x, self.weights, self.biases, self.act_wrapper(0), False)
        return y.flatten()


if __name__ == "__main__":

    elem = [0.0, 3.0]
    # y0 = (2.0, 0.0)
    mu = 2.0
    beta = 0.5
    rand_init = 1.0
    rand_train = 1.0
    max_cycles = 100
    tolerance = 1e-3
    # equation_info = auto_ode1d(y0)
    # equation_info = test_simple(y0)
    # equation_info = test_ode(y0)
    # equation_info = vdp1(y0, mu)
    # equation_info = vdp_st(mu)
    equation_info = non_linear_spring(_w=mu, _beta=beta, start=elem[0])

    hidden_neurons = 100
    q = 50
    activation_function = "tanh"

    test_num = 1000

    model = ExtremeLearning(q, hidden_neurons, elem, equation_info, activation_function, rand_init)
    # model.compute_ls_simple()
    # model.compute_ls()
    # model.compute_nls()
    # model.compute_vdp()
    status = model.turbulent_training("compute_vdp", max_cycles, turb_range=rand_train, loss_tol=tolerance)

    total_test_points = test_num * model.num_elements
    x_test = np.linspace(model.lower_bound, model.upper_bound, total_test_points)

    y_result = model.test(x_test.reshape((model.num_elements, -1, 1)))
    # y_truth = model.solution(x_test)
    y_truth = model.solution(model.upper_bound, total_test_points)
    print("l1: %.7e" % np.amax(np.abs(y_result - y_truth)))
    print("l2: %.7e" % rmse(y_truth, y_result))

    fig = plt.figure()
    fig.set_rasterized(True)

    pred = fig.add_subplot(111)
    pred.plot(x_test, y_truth, linestyle="dashed")
    pred.plot(x_test, y_result)
    pred.set_title('Prediction')
    pred.set_ylabel('y')
    pred.set_xlabel('x')
    pred.legend(['truth', 'ls_est'], loc='upper right')
    textstr = r'$error=%.3e$' % rmse(y_truth, y_result)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    pred.text(0.75, 0.15, textstr, transform=pred.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    pred.set_rasterized(True)

    plt.show()
