import numpy as np
import scipy
import scipy.optimize as opt

from . import BaseOptimizer


class NewtonDirection(BaseOptimizer):
    def _iteration(self):
        self._hess_cur_pos = self._calculate_hess(self._cur_pos)
        self._update_grad()
        try:
            L = scipy.linalg.cholesky(self._hess_cur_pos, lower=True)
            y = np.linalg.solve(L, self._grad_cur_pos)
            pk = -np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            pk = -self._grad_cur_pos
        alpha, *_ = opt.line_search(self._calculate_func, self._calculate_grad, self._cur_pos, pk, self._grad_cur_pos)
        if alpha is None:
            alpha = 1.0
        self._cur_pos += alpha * pk
