import numpy as np

from . import DenseQuasiNewton


class BFGS(DenseQuasiNewton):
    def _calculate_Gk_impl(self):
        rho_k_inv = np.dot(self._yk, self._sk)
        if rho_k_inv <= 0:
            return self._Gk
        rho_k = 1 / rho_k_inv
        ident = self._get_identity()
        left = ident - rho_k * np.outer(self._sk, self._yk)
        mid = self._Gk
        right = ident - rho_k * np.outer(self._yk, self._sk)
        tail = rho_k * np.outer(self._sk, self._sk.T)
        return left @ mid @ right + tail

