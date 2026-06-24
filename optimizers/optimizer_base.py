from abc import abstractmethod, ABC
import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, fun, grad_f, hess_f, tol: float, starting_pos: np.ndarray, name: str = "",
                 max_iter: int = 10_000, do_history: bool = False):
        self._function = fun
        self._grad_f = grad_f
        self._hess_f = hess_f
        self._eps = tol
        self._starting_pos = starting_pos
        self._name = name
        self._max_iter = max_iter
        self._do_history = do_history

        self._history = [starting_pos.copy()]
        self._cur_pos = starting_pos.copy()

        self._grad_cur_pos = None
        self._f_cur_pos = None
        self._hess_cur_pos = None
        self._stop = False
        self._found_solution = False
        self.stop_reason = None

        self.iteration_count = 0
        self.func_call_count = 0
        self.grad_call_count = 0
        self.hess_call_count = 0

    def _calculate_func(self, x) -> float:
        result = self._function(x)
        self.func_call_count += 1
        return result

    def _calculate_grad(self, x) -> np.ndarray:
        result = self._grad_f(x)
        self.grad_call_count += 1
        return result

    def _calculate_hess(self, x) -> np.ndarray:
        result = self._hess_f(x)
        self.hess_call_count += 1
        return result

    def _update_func(self):
        self._f_cur_pos = self._calculate_func(self._cur_pos)

    def _update_grad(self):
        self._grad_cur_pos = self._calculate_grad(self._cur_pos)

    def _update_hess(self):
        self._hess_cur_pos = self._calculate_hess(self._cur_pos)

    def reset(self):
        self._history = [self._starting_pos]
        self._cur_pos = self._starting_pos.copy()

        self._grad_cur_pos = None
        self._f_cur_pos = None
        self._hess_cur_pos = None

        self.iteration_count = 0
        self.func_call_count = 0
        self.grad_call_count = 0
        self.hess_call_count = 0

    def get_history(self):
        return self._history if self._do_history else None

    def get_grad(self):
        self._update_grad()
        return self._grad_cur_pos.copy()

    def __str__(self):
        return self._name

    def _can_do_iteration(self) -> bool:
        if self._stop:
            return False
        if self.iteration_count >= self._max_iter:
            self.stop_reason = "Iteration limit reached"
            self._stop = True
        elif self._grad_cur_pos is None:
            return True
        elif not np.all(np.isfinite(self._cur_pos)) or not np.all(np.isfinite(self._grad_cur_pos)):
            self.stop_reason = "NaN or infinity reached"
            self._stop = True
        elif np.linalg.norm(self._grad_cur_pos) < self._eps:
            self.stop_reason = "OK Found solution"
            self._found_solution = True
            self._stop = True
        if self._stop:
            return False
        return True

    def find_minimum(self):
        while self._can_do_iteration():
            self._iteration()
            if self._do_history:
                self._history.append(self._cur_pos.copy())
            self.iteration_count += 1
        if self._found_solution:
            self._update_func()
            return self._cur_pos, self._f_cur_pos
        return None

    @abstractmethod
    def _iteration(self):
        pass

    def _f_tilde(self, pk, alpha: float) -> float:
        return self._calculate_func(self._step(alpha, pk))

    def _step(self, alpha: float, direction):
        return self._cur_pos + alpha * direction
