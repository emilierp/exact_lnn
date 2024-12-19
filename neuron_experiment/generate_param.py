"""This file aims to create classes and methods to generate synapses and input signal for the neuron."""
import numpy as np
from .synapse import Synapse
from .neuron import Neuron


class GenerateSynapsesBatch:
    """This class aims to generate a batch of synapse with random parameters for several synaptic input to one neuron."""
    def __init__(self, number_of_synapses: int, ranges_for_params: dict = {'adjency_factor': (0.001, 1.0), 'mu': (0.3, 0.8), 'sigma': (3, 8)}, seed: int = None):
        self.number_of_synapses = number_of_synapses
        self.range_for_params = ranges_for_params
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Initialize the random number generator with a seed
        self.post_synaptic_time_constant = self.rng.random(1) * (1.6 - 0.0025) + 0.0025

    def _generate_params(self):
        """This function aims to generate the parameters for the synapses."""
        reversal_potentials = self.rng.choice([-1, 1], size=self.number_of_synapses)
        params_dict = {'reversal_potential': reversal_potentials}
        for key, value in self.range_for_params.items():
            params_dict[key] = self._init_params(value)
        return params_dict

    def create_list_of_synapses(self):
        """This function aims to create a list of synapses for the neuron."""
        params_dict = self._generate_params()
        synapses_list = []
        for synapse_index in range(self.number_of_synapses):
            synapse = Synapse(params_dict['reversal_potential'][synapse_index], params_dict['sigma'][synapse_index], params_dict['mu'][synapse_index], params_dict['adjency_factor'][synapse_index])
            synapses_list.append(synapse)
        return synapses_list
    
    def create_neuron(self):
        """This function aims to create a neuron with the synapses."""
        synapses = self.create_list_of_synapses()
        return Neuron(synapses, self.post_synaptic_time_constant)

    def _init_params(self, range_param : tuple):
        """This function aims to initialize the parameters of the thanks to the range."""
        min_val, max_val = range_param
        if min_val == max_val:
            return np.ones(self.number_of_synapses) * min_val
        else:
            return self.rng.random(self.number_of_synapses) * (max_val - min_val) + min_val

class GeneratePresynapticInput:
    """This class aims to generate the random input signal for the neuron."""
    def __init__(self, number_of_synapses: int, number_of_time_samples : int, seed: int = None):
        self.number_of_synapses = number_of_synapses
        self.number_of_time_samples = number_of_time_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_input_signal(self): 
        """This function aims to generate the input signal for the neuron."""  
        return self.rng.random((self.number_of_synapses, self.number_of_time_samples))