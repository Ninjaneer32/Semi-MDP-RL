import numpy as np


class ParameterEstimator:
    # Maximum likelihood parameters of Poisson distributions
    def __init__(self, params):
        self._params = params
        self._mean_time = {p: 0. for p in params}
        self._mean_count = {p: 0. for p in params}
        self._n_samples = 0

    def observe(self, time_intervals, counts):
        for p in self._params:
            self._mean_time[p] = (self._mean_time[p] * self._n_samples + time_intervals) / (self._n_samples + 1)
            self._mean_count[p] = (self._mean_count[p] * self._n_samples + counts[p]) / (self._n_samples + 1)
        self._n_samples += 1

    def infer(self):
        if self._n_samples > 0:
            return {p: self._mean_count[p] / self._mean_time[p] for p in self._params}
        else:
            # TODO : better initialization?
            # return .01 * np.ones(self._n_params)
            return {p : .01 for p in self._params}