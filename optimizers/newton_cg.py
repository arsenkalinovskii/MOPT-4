from . import BaseOptimizer
import scipy.optimize as opt


class NewtonCG(BaseOptimizer):
    def find_minimum(self):
        if self._do_history:
            opt_result = opt.minimize(self._calculate_func, self._starting_pos, method='Newton-CG',
                                  jac=self._calculate_grad,
                                  hess=self._calculate_hess, tol=self._eps,
                                  callback=self._increment_nit_and_append)
        else:
            opt_result = opt.minimize(self._calculate_func, self._starting_pos, method='Newton-CG',
                                      jac=self._calculate_grad,
                                      hess=self._calculate_hess, tol=self._eps,
                                      callback=self._increment_nit)
        if not opt_result.success:
            self.stop_reason = opt_result.message
            return None
        self.stop_reason = "OK Found solution"
        self._cur_pos, self._f_cur_pos = opt_result.x, opt_result.fun
        self._grad_cur_pos = opt_result.jac
        self.iteration_count = opt_result.nit
        self.func_call_count = opt_result.nfev
        self.grad_call_count = opt_result.njev
        self.hess_call_count = opt_result.nhev

        return self._cur_pos, self._f_cur_pos

    def _increment_nit_and_append(self, xk):
        self._increment_nit(xk)
        self._history.append(xk)

    def _increment_nit(self, xk):
        self._cur_pos = xk
        self.iteration_count += 1

    def _iteration(self):
        pass