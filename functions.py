import numpy as np


class func():
    def __init__(self, f, grad_f, hess_f, name, xmins):
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.name = name
        self.xmins = xmins

    def __str__(self):
        return self.name


def rosenbrock_(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock_(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])

def hess_rosenbrock_(x):
    return np.array([
        [
            2 - 4 * 100 * x[1] + 12 * 100 * x[0] ** 2,
            -4 * 100 * x[0]
        ],
        [
            -4 * 100 * x[0],
            2 * 100
        ]
    ])

rosenbrock_mins_ = [np.array([1, 1])]

rosenbrock = func(rosenbrock_, grad_rosenbrock_, hess_rosenbrock_, 'Rosenbrock function',
                  rosenbrock_mins_)


def himmelblau_(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def grad_himmelblau_(x):
    dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx, dy])

def hess_himmelblau_(x):
    x[0], x[1] = x

    a = x[0]**2 + x[1] - 11
    b = x[0] + x[1]**2 - 7

    return np.array([
        [
            4 * a + 8 * x[0]**2 + 2,
            4 * (x[0] + x[1])
        ],
        [
            4 * (x[0] + x[1]),
            4 * b + 8 * x[1]**2 + 2
        ]
    ])

himmelblau_mins_ = [
    np.array([3, 2]),
    np.array([-2.805118, 3.131312]),
    np.array([-3.779310, -3.283186]),
    np.array([3.584428, -1.848126]),
]

himmelblau = func(himmelblau_, grad_himmelblau_, hess_himmelblau_, 'Himmelblau function',
                  himmelblau_mins_)
