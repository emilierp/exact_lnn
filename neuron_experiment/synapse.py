"""This file aims to define the class of synapse who is responsible for the communication between two neurons, and between the input and the neuron."""
import numpy as np
import tensorflow as tf

class Synapse:
    """This class aims to define the synapse."""
    def __init__(self, reversal_potential, sigma, mu, adjency_factor):
        self.reversal_potential = reversal_potential
        self.sigma = sigma
        self.mu = mu
        self.adjency_factor = adjency_factor
    
    def calculate_synaptic_current(self, input_current, post_synaptic_neuron_potential):
        """This function aims to calculate the synaptic current."""
        return self.synaptic_release_non_linearity(input_current) * (self.reversal_potential - post_synaptic_neuron_potential)
    
    def __call__(self, input_current, post_synaptic_neuron_potential):
        """This function aims to return the synaptic current."""
        return self.calculate_synaptic_current(input_current, post_synaptic_neuron_potential)

    def synaptic_release_non_linearity(self, x):
        """This function aims to return the sigmoid non-linearity."""
        return 1/ (1 + np.exp(-self.sigma * (x - self.mu)))
    