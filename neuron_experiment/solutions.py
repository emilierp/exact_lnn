"""This file aims to provide function to approximate post-synaptic neuron potential based on the input current and the synapses."""

import numpy as np

from copy import deepcopy
from typing import List
from neuron import Neuron

class ClosedFormApproximation:
    """This class aims to provide the closed form approximation of the post-synaptic neuron potential."""
    def __init__(self, neuron : Neuron, time_samples : List[float], input_signal : List[float]):
        self.neuron = deepcopy(neuron)
        self.time_samples = time_samples
        self.input_signal = input_signal  
    def solve(self):
        """This function aims to solve the post-synaptic neuron potential."""
        current_synaptic_current = np.zeros(len(self.time_samples))
        current_synaptic_current[0] = self.neuron.initial_neuron_potential
        synapses_current = np.zeros((len(self.time_samples), len(self.neuron.synapses_list)))
        for index_synapse, synapse in enumerate(self.neuron.synapses_list):
            synapses_current[:, index_synapse] = (self.neuron.initial_neuron_potential - synapse.reversal_potential) * np.exp(-self.time_samples/self.neuron.postsynaptic_neuron_time_constant - synapse.synaptic_release_non_linearity(self.input_signal[index_synapse])*self.time_samples)*(synapse.synaptic_release_non_linearity(-self.input_signal[index_synapse] + 2*synapse.mu)) + synapse.reversal_potential 
            current_synaptic_current += synapses_current[:, index_synapse]
        current_synaptic_current[0] = self.neuron.initial_neuron_potential
        return current_synaptic_current
    

class PiecewiseConstantInputSolutionSeveralSynapses:
    """This class aims to provide the piecewise constant input solution for several synapses."""
    def __init__(self, neuron, time_samples: List[float], input_signal: np.ndarray):
        self.neuron = neuron
        self.time_samples = time_samples    
        self.input_signal = input_signal
        self.integral_value = 0
        self.time_length = self.time_samples.shape[0]
        self.time_samples_diff = np.diff(self.time_samples)
        self.number_of_synapses = len(self.neuron.synapses_list)
        self.w_T = 1 / self.neuron.postsynaptic_neuron_time_constant
        self.array_A = np.asarray([synapse.reversal_potential for synapse in self.neuron.synapses_list])
        self.integral = None
        self.diff_multiplicator = None
        self.pre_computation()

    def solve(self):
        """This function aims to solve the post-synaptic neuron potential."""
        potential = np.zeros(shape=(self.time_length))
        potential[0] = self.neuron.initial_neuron_potential
        potential_t = 0
        integral_t =0
        
        for time_index in range(1, self.time_length):
            integral_t = self.integral[time_index-1]
            potential_t = potential_t*np.exp(self.diff_multiplicator[time_index-1])+integral_t
            potential[time_index] = potential_t  
        return potential

    def pre_computation(self):
        """This function aims to pre-compute the integral and the diff multiplicator."""
        delta_t = self.time_samples_diff[0]
        synapses_release_non_linearity = np.array([
            synapse.synaptic_release_non_linearity(self.input_signal[synapse_index, :])
            for synapse_index, synapse in enumerate(self.neuron.synapses_list)
        ])
        integral_synaptic_release_non_linearity = np.cumsum(synapses_release_non_linearity, axis=1).T
        integral_synaptic_release_non_linearity = np.vstack(
            (np.zeros(integral_synaptic_release_non_linearity.shape[1]), integral_synaptic_release_non_linearity)
        )[:-1, :]
        multiplicator = -self.time_samples*self.w_T-delta_t*np.sum(integral_synaptic_release_non_linearity, axis=1)

        all_synaptic_release = np.sum(synapses_release_non_linearity, axis=0)
        cumulative_release = np.cumsum(all_synaptic_release)
        exp_term = self.w_T * self.time_samples + delta_t * cumulative_release

        first_exp_term = exp_term[1:] +multiplicator[1:]
        second_exp_term = exp_term[:-1]+ multiplicator[1:]
        exp_diff = np.exp(first_exp_term) - np.exp(second_exp_term)
        contributions = exp_diff / (self.w_T + all_synaptic_release[:-1])
        mult = synapses_release_non_linearity[ :, :-1] * contributions
        diff_multiplicator = np.diff(multiplicator)

        multiplication=self.array_A[:, None]* mult
        self.integral = np.sum(multiplication, axis=0)
        self.diff_multiplicator = diff_multiplicator