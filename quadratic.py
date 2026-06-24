import numpy as np


class Quadratic:
    def __init__(self, n, k):
        self._n = n
        self._k = k

        lambda_min = np.random.rand() / n
        lambda_max = k * lambda_min

        lambdas = np.random.uniform(lambda_min, k * lambda_min, n)
        i, j = np.random.choice(n, 2, replace=False)
        lambdas[i], lambdas[j] = lambda_min, lambda_max
        lambda_matrix = np.diag(lambdas)

        v = np.random.randn(n) / (n ** 2)
        Q = np.eye(n) - 2 * (v @ v.T)
        self._A = Q.T @ lambda_matrix @ Q

        self._xmin = np.random.randn(n)
        self._b = self._A @ self._xmin

        self._c = np.random.rand()

    def eval_func(self, x) -> float:
        return 0.5 * x.T @ self._A @ x - self._b @ x + self._c

    def eval_grad(self, x) -> np.ndarray:
        return self._A @ x - self._b

    def eval_hess(self, x) -> np.ndarray:
        return self._A

    def get_minimum(self):
        return self._xmin, self._c

    def __call__(self, x):
        return self.eval_func(x)
