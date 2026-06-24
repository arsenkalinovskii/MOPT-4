from abc import abstractmethod

import scipy.optimize as opt
import numpy as np

from . import BaseOptimizer

class NonLinearConjugateGradient(BaseOptimizer):
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
        if np.dot(self._grad_cur_pos, self._pk) <= 0:
            self._pk = self._grad_cur_pos.copy()
        alpha_k = self._calculate_alpha()
        self._cur_pos -= alpha_k * self._pk
        self._update_grad()
        beta_k = self._calculate_beta()
        self._pk = self._grad_cur_pos + beta_k * self._pk


    def _calculate_alpha(self) -> float:
        gk = self._pk
        self._update_func()
        f_prev = self._f_cur_pos
        alpha_border = 1
        while self._calculate_func(self._cur_pos - alpha_border * gk) < f_prev:
            alpha_border *= 2.0

        fun = lambda t: self._calculate_func(self._cur_pos - t * gk)
        result = opt.minimize_scalar(fun=fun, bounds=(0, alpha_border), method='bounded')
        return result.x

    @abstractmethod
    def _calculate_beta(self):
        pass