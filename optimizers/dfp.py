import numpy as np

from . import DenseQuasiNewton

class DFP(DenseQuasiNewton):
    def _calculate_Gk_impl(self):
        rho_k_inv = np.dot(self._yk, self._sk)
        if rho_k_inv <= 0:
            return self._Gk
        rho_k = 1 / rho_k_inv
        first = self._Gk
        second = rho_k * np.outer(self._sk, self._sk)
        third = -(1 / np.dot(self._yk, self._Gk @ self._yk.T)) * self._Gk @ np.outer(self._yk, self._yk) @ self._Gk
        return first + second + third