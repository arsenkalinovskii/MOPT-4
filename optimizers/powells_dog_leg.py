import numpy as np
import scipy.linalg
from scipy._lib.array_api_compat.numpy import linalg

from . import BaseOptimizer

class PowellsDogLeg(BaseOptimizer):

    def __init__(self, fun, grad_f, hess_f, tol, starting_pos,
                 name="", max_iter=10_000, do_history=False, delta_init : float = 1,
                 eta : float = 0.1):
        super().__init__(fun, grad_f, hess_f, tol, starting_pos,
                         name, max_iter, do_history)
        self._delta = delta_init
        self._eta = eta

    def _solve_newton_step(self):
        H = 0.5 * (self._hess_cur_pos + self._hess_cur_pos.T)
        g = self._grad_cur_pos

        L = scipy.linalg.cholesky(H, lower=True)
        y = np.linalg.solve(L, g)
        p = -np.linalg.solve(L.T, y)
        return p

    def _solve_cauchy_step(self):
        g = self._grad_cur_pos
        H = self._hess_cur_pos

        gHg = g @ (H @ g)
        if gHg <= 0:
            return -g

        alpha = (g @ g) / gHg
        return -alpha * g

    def _dogleg_step(self, p_u, p_n, delta):
        norm_pu = np.linalg.norm(p_u)
        norm_pb = np.linalg.norm(p_n)

        if norm_pb <= delta:
            return p_n

        if norm_pu >= delta:
            return (delta / norm_pu) * p_u

        p_diff = p_n - p_u

        a = np.dot(p_diff, p_diff)
        b = np.dot(p_u, p_diff)
        c = np.dot(p_u, p_u) - delta**2

        tau = (-b + np.sqrt(max(0, b*b - a*c))) / a
        return p_u + tau * p_diff

    def _model(self, p):
        g = self._grad_cur_pos
        H = self._hess_cur_pos
        return g @ p + 0.5 * p @ (H @ p)

    def _iteration(self):
        self._update_hess()
        self._update_grad()
        g = self._grad_cur_pos

        try:
            p_u = self._solve_cauchy_step()
            p_n = self._solve_newton_step()
            p = self._dogleg_step(p_u, p_n, self._delta)

        except np.linalg.LinAlgError:
            p = -g
            if np.linalg.norm(p) > self._delta:
                p = p * (self._delta / np.linalg.norm(p))

        x = self._cur_pos.copy()
        if self._f_cur_pos is None:
            self._update_func()
        f_old = self._f_cur_pos
        f_new = self._calculate_func(x + p)

        actual_reduction = f_old - f_new
        predicted_reduction = -self._model(p)

        if predicted_reduction <= 0:
            self._cur_pos = x
            self._delta *= 0.25
            return True

        rho = actual_reduction / (predicted_reduction + 1e-12)

        if rho > self._eta:
            self._cur_pos = x + p
            self._f_cur_pos = f_new
            self._update_grad()
        else:
            self._cur_pos = x

        if rho < 0.25:
            self._delta *= 0.25
        elif rho > 0.75 and np.linalg.norm(p) > 0.9 * self._delta:
            self._delta = min(self._delta * 2, 1e6)

        return True