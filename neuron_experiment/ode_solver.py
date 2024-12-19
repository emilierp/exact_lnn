from scipy.integrate import odeint
import numpy as np

def model_paper(x, t, tau, synapses_list, input_signal):
    """Model of the post neuronal potential."""
    dxdt = -x/tau
    for synapse_index, synapse in enumerate(synapses_list):
        dxdt += synapse.calculate_synaptic_current(input_signal[synapse_index]([t]), x) 
    return dxdt

def solve_neuron(neuron, time_samples, input_signal):
    """This function aims to solve the neuron with ODEINT from python using LSODA method."""
    tau = neuron.postsynaptic_neuron_time_constant
    initial_neuron_potential = neuron.initial_neuron_potential
    synapses_list = neuron.synapses_list
    x = odeint(model_paper, initial_neuron_potential, time_samples, args=(tau, synapses_list, input_signal))
    return x


def euler_evaluation(x, t, tau, synapses_list, input_signal):
    """Model of the post neuronal potential for euler."""
    dxdt = -x/tau
    for synapse_index, synapse in enumerate(synapses_list):
        dxdt += synapse.calculate_synaptic_current(input_signal[synapse_index][t], x) 
    return dxdt

def solve_euler_method(neuron, time_samples, input_signal, delta_t):
    """This function aims to solve the neuron with Euler method."""
    tau = neuron.postsynaptic_neuron_time_constant
    initial_neuron_potential = neuron.initial_neuron_potential
    synapses_list = neuron.synapses_list
    x = np.zeros(len(time_samples))
    for i, _ in enumerate(time_samples):
        dxdt = euler_evaluation(initial_neuron_potential, i, tau, synapses_list, input_signal)
        x[i] = initial_neuron_potential + delta_t*dxdt
        initial_neuron_potential = x[i]
    return x