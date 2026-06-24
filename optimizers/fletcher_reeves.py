import numpy as np

from . import NonLinearConjugateGradient


class FletcherReeves(NonLinearConjugateGradient):
    def _calculate_beta(self):
        norm_gk_prev_sqr = np.linalg.norm(self._gk_prev) ** 2
        norm_gk_sqr = np.linalg.norm(self._grad_cur_pos) ** 2
        return norm_gk_sqr / norm_gk_prev_sqr
