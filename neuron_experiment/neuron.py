"""This file aims to define the class of neuron, whose behaviour is defined by a differential equation."""

from typing import List
from synapse import Synapse

class Neuron:
    """This class aims to define the neuron."""
    def __init__(self, synapses_list : List[Synapse], postsynaptic_neuron_time_constant : float):
        self.synapses_list = synapses_list
        self.postsynaptic_neuron_time_constant = postsynaptic_neuron_time_constant
        self.initial_neuron_potential = 0
        self.neuron_potential = self.initial_neuron_potential
        self.current_neuron_potential = self.initial_neuron_potential

    def limit_synapses(self, number_of_synapses):
        """This function aims to limit the number of synapses, to make experiment with a growing number of synapses."""
        neuron_with_less_synapses =  Neuron(synapses_list=self.synapses_list[:number_of_synapses], postsynaptic_neuron_time_constant=self.postsynaptic_neuron_time_constant)
        return neuron_with_less_synapses
        


    
    