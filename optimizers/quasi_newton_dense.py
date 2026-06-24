from abc import abstractmethod

import numpy as np

from . import QuasiNewton


class DenseQuasiNewton(QuasiNewton):
    def __init__(self, fun, grad_f, hess_f, tol: float, starting_pos: np.ndarray, name: str = "",
                 max_iter: int = 10_000, do_history: bool = False):
        super().__init__(fun, grad_f, hess_f, tol, starting_pos, name, max_iter, do_history)
        self._Gk = None

    def _calculate_pk(self):
        self._calculate_Gk()
        pk = -self._Gk @ self._grad_cur_pos
        return pk

    def _calculate_Gk(self):
        if self._Gk is None:
            self._Gk = self._get_identity()
            self._update_grad()
        else:
            self._Gk = self._calculate_Gk_impl()

    @abstractmethod
    def _calculate_Gk_impl(self):
        pass
