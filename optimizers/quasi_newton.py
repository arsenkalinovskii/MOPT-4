from abc import abstractmethod

import numpy as np

from . import BaseOptimizer
import scipy.optimize as opt


class QuasiNewton(BaseOptimizer):
    def __init__(self, fun, grad_f, hess_f, tol: float, starting_pos: np.ndarray, name: str = "",
                 max_iter: int = 10_000, do_history: bool = False):
        super().__init__(fun, grad_f, hess_f, tol, starting_pos, name, max_iter, do_history)
        self._sk = None
        self._yk = None

    def _iteration(self):
        pk = self._calculate_pk()
        alpha, *_ = opt.line_search(self._calculate_func, self._calculate_grad, self._cur_pos, pk, self._grad_cur_pos)
        if alpha is None:
            alpha = 1.0
        self._cur_pos += alpha * pk
        gk = self._grad_cur_pos.copy()
        self._update_grad()
        self._sk = alpha * pk
        self._yk = self._grad_cur_pos - gk

    @abstractmethod
    def _calculate_pk(self):
        pass

    def _get_identity(self):
        return np.identity(self._starting_pos.shape[0])
