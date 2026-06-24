import numpy as np

from . import QuasiNewton
import scipy.optimize as opt


class LBFGS(QuasiNewton):
    def __init__(self, fun, grad_f, hess_f, tol: float, starting_pos: np.ndarray, name: str = "",
                 max_iter: int = 10_000, do_history: bool = False, m: int = 10):
        super().__init__(fun, grad_f, hess_f, tol, starting_pos, name, max_iter, do_history)
        self._m = m
        self._tuples = []

    def _iteration(self):
        mu_k = min(self._m, len(self._tuples))
        if self._grad_cur_pos is None:
            self._update_grad()
        q = self._grad_cur_pos.copy()
        alphas = [0.0] * mu_k
        for i in range(mu_k - 1, -1, -1):
            s_i, y_i, rho_i = self._tuples[i]
            alphas[i] = rho_i * np.dot(s_i, q)
            q -= alphas[i] * y_i

        if mu_k > 0:
            sk, yk, _ = self._tuples[mu_k - 1]
            gamma_k = np.dot(sk, yk) / np.dot(yk, yk)
            r = gamma_k * q
        else:
            r = q
        for i in range(0, mu_k):
            s_i, y_i, rho_i = self._tuples[i]
            beta_i = rho_i * np.dot(y_i, r)
            r += s_i * (alphas[i] - beta_i)
        pk = -r
        alpha_k, *_ = opt.line_search(self._calculate_func, self._calculate_grad, self._cur_pos, pk, self._grad_cur_pos)
        if alpha_k is None:
            alpha_k = 1.0
        sk = alpha_k * pk
        self._cur_pos += sk
        gk_prev = self._grad_cur_pos
        self._update_grad()
        yk = self._grad_cur_pos - gk_prev
        rho_k_inv = np.dot(yk, sk)
        if rho_k_inv > 0:
            rho_k = 1 / np.dot(yk, sk)
            self._tuples.append((sk, yk, rho_k))
            if len(self._tuples) > self._m:
                self._tuples.pop(0)

    def _calculate_pk(self):
        pass

    def reset(self):
        super().reset()
        self._tuples = []
