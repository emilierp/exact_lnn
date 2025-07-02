"""This module contains the class to find the best resampling factor for the approximated solution, to minimize the error with a classical ODE_
-solved solution. This is done to be in the same range of value as the ODE solution."""
import copy
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .solutions import ClosedFormApproximation


class FindBestResamplingFactor:
    """This class aims to find the best resampling factor for the approximated solution, to minimize the error with a classical ODE-solved solution."""
    def __init__(self, neuron, time_samples, input_signal, delta_t_min = 0.00001, delta_t_max = 1.0, number_of_samples=1000, type_of_signal=ClosedFormApproximation, type_of_experiment='ecg', neuron_index=None):
        self.delta_t_min = delta_t_min
        self.delta_t_max = delta_t_max
        self.best_resampling_factor = {}
        self.deltas = np.linspace(self.delta_t_min, self.delta_t_max, number_of_samples)
        self.errors = {}
        self.neuron = neuron
        self.time_samples = time_samples
        self.input_signal = input_signal
        if type_of_experiment == 'ecg':
            self.reference_solution = np.load(f'./ode_solved_ecg/neuron_ecg/solution_neuron_ecg_for_{len(neuron.synapses_list)}_synapses.npy')
        elif neuron_index is not None:
            self.reference_solution = np.load(f'./ode_solved/neuron_{neuron_index}/solution_neuron_{neuron_index}_for_{len(neuron.synapses_list)}_synapses.npy')
        else:
            self.reference_solution = np.load(f'./ode_solved_random/neuron_random/solution_neuron_random_for_{len(neuron.synapses_list)}_synapses.npy')
        self.reference_solution.resize(len(time_samples))
        self.type_of_signal = type_of_signal

    def solve(self):
        """Find the best resampling factor for the approximated solution."""
        self.errors['MSE'] = {}
        self.errors['MAE'] = {}
        self.errors['RMSE'] = {}
        self.errors['R2'] = {}
        for delta in self.deltas:
            resampled_time_samples = np.linspace(0, self.time_samples[-1]*delta-delta, len(self.time_samples))
            solution = self.type_of_signal(neuron = copy.deepcopy(self.neuron), time_samples=resampled_time_samples, input_signal=self.input_signal).solve()
            mse, mae, rmse, r2 = self.compute_error(solution)
            self.errors['MSE'][delta] = mse
            self.errors['MAE'][delta] = mae
            self.errors['RMSE'][delta] = rmse
            self.errors['R2'][delta] = r2
        self.best_resampling_factor['MSE'] = self.deltas[np.argmin(list(self.errors['MSE'].values()))]
        self.best_resampling_factor['MAE'] = self.deltas[np.argmin(list(self.errors['MAE'].values()))]
        self.best_resampling_factor['RMSE'] = self.deltas[np.argmin(list(self.errors['RMSE'].values()))]
        self.best_resampling_factor['R2'] = self.deltas[np.argmax(list(self.errors['R2'].values()))]
        return self.best_resampling_factor
    
    def compute_error(self, solution):
        """Compute the error between the approximated solution and the reference solution."""
        mse = mean_squared_error(self.reference_solution, solution)
        mae = mean_absolute_error(self.reference_solution, solution)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.reference_solution, solution)
        return mse, mae, rmse, r2
