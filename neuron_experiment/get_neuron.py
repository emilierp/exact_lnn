"""This module aims to get the information from the matlab file, from neurons and synapses to inputs."""
from .neuron import Neuron
from .synapse import Synapse

class GetInfoMatlabFile:
    """This class aims to get the information from the matlab file."""
    def __init__(self, file_matlab, number_of_neurons : int):
        self.file_matlab = file_matlab
        self.number_of_neurons = number_of_neurons
    def get_inputs_per_neurons(self):
        """This function aims to get the inputs per neurons."""
        v_pre = self.file_matlab['v_pre']
        input_per_neurons = []
        for v_pre_neuron in v_pre[0][0:self.number_of_neurons]:
            input_per_neuron = []
            number_of_synapses = v_pre_neuron.shape[1]
            for i in range(number_of_synapses):
                input_per_neuron.append(v_pre_neuron[:, i])
            input_per_neurons.append(input_per_neuron)
        return input_per_neurons

    def get_neurons(self):
        """This function aims to get the neurons' parameters."""
        neuron_parameters = self.file_matlab['n_values']
        neurons = []
        all_synapses = self.get_synapses_per_neuron()
        for i in range(self.number_of_neurons):
            post_synaptic_neuron_time_constant = neuron_parameters[0][i][0][0]/neuron_parameters[0][i][0][2] 
            neuron = Neuron(synapses_list=all_synapses[i], postsynaptic_neuron_time_constant=1/post_synaptic_neuron_time_constant)
            neurons.append(neuron)
        return neurons
    def get_synapses_per_neuron(self):
        """This function aims to get the synapses' parameters per neuron."""
        synapses_for_neurons = self.file_matlab['s_values']
        all_synapses = []
        for i in range(self.number_of_neurons):
            synapses_per_neuron = []
            synapses_for_neuron = synapses_for_neurons[0][i]
            number_of_synapses = synapses_for_neuron.shape[0]
            for s in range(number_of_synapses):
                reversal_potential = synapses_for_neuron[s][3]
                sigma = synapses_for_neuron[s][0]
                mu = synapses_for_neuron[s][1]
                adjency_factor = synapses_for_neuron[s][2]
                synapse = Synapse(reversal_potential=reversal_potential, sigma=sigma, mu=mu, adjency_factor=adjency_factor)
                synapses_per_neuron.append(synapse)
            all_synapses.append(synapses_per_neuron)
        return all_synapses
        
