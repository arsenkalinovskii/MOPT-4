import numpy as np

from . import NonLinearConjugateGradient


class PolakRibiere(NonLinearConjugateGradient):
    def _calculate_beta(self):
        norm_gk_prev_sqr = np.linalg.norm(self._gk_prev) ** 2
        beta_k_PR = np.dot(self._grad_cur_pos, self._grad_cur_pos - self._gk_prev) / norm_gk_prev_sqr
        return max(beta_k_PR, 0)