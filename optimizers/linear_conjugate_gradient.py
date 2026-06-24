import numpy as np

from . import BaseOptimizer


class LinearConjugateGradient(BaseOptimizer):
    def __init__(self, fun, grad_f, hess_f, tol: float, starting_pos: np.ndarray, name: str = "",
                 max_iter: int = 10_000, do_history: bool = False):
        super().__init__(fun, grad_f, hess_f, tol, starting_pos, name, max_iter, do_history)
        self._gk_prev = None
        self._pk = None

    def _perform_preparations(self):
        if self._grad_cur_pos is None:
            self._update_grad()
            self._pk = self._grad_cur_pos.copy()

    def _iteration(self):
        self._perform_preparations()
        self._gk_prev = self._grad_cur_pos.copy()
        if np.linalg.norm(self._gk_prev) < self._eps:
            return
        self._update_hess()
        alpha_k = self._calculate_alpha()
        self._cur_pos -= alpha_k * self._pk
        self._update_grad()
        beta_k = self._calculate_beta()
        self._pk = self._grad_cur_pos + beta_k * self._pk

    def _calculate_alpha(self):
        norm_gk_prev_sqr = np.linalg.norm(self._gk_prev) ** 2
        return norm_gk_prev_sqr / np.dot(self._hess_cur_pos @ self._pk, self._pk)

    def _calculate_beta(self):
        norm_gk_prev_sqr = np.linalg.norm(self._gk_prev) ** 2
        norm_gk_sqr = np.linalg.norm(self._grad_cur_pos) ** 2
        return norm_gk_sqr / norm_gk_prev_sqr
