import argparse
import numpy as np
import copy
import time
import scipy.io
import wfdb
import os

from ode_solver import solve_euler_method, solve_neuron
from generate_param import GenerateSynapsesBatch, GeneratePresynapticInput
from get_neuron import GetInfoMatlabFile
from neuron_input import PiecewiseConstantFunction

# Specify the path to your MIT-BIH files

ode_parameters = {
    7: {"delta_t": 0.0001, "initial_potential": 0},
    11: {"delta_t": 0.02, "initial_potential": -0.5},
    17: {"delta_t": 0.044, "initial_potential": -0.5},
    1: {"delta_t": 0.1, "initial_potential": 0},
    15: {"delta_t": 0.1, "initial_potential": 0},
    12: {"delta_t": 0.2, "initial_potential": -0.7246},
    4: {"delta_t": 0.5, "initial_potential": 0},
    13: {"delta_t": 0.5, "initial_potential": 0},
    14: {"delta_t": 0.5, "initial_potential": 0},
    9: {"delta_t": 0.6, "initial_potential": 0},
    18: {"delta_t": 0.7, "initial_potential": 0},
    19: {"delta_t": 0.8, "initial_potential": 0},
    10: {"delta_t": 0.9, "initial_potential": 0},
    5: {"delta_t": 1, "initial_potential": 0},
    6: {"delta_t": 1, "initial_potential": 0},
    2: {"delta_t": 1, "initial_potential": 0},
    16: {"delta_t": 1, "initial_potential": 0},
    8: {"delta_t": 1, "initial_potential": -1.43},
    3: {"delta_t": 2, "initial_potential": 0}
} # This has been extracted from Hasani's MatLab file.


def solve_ode_neuron(experiment_type):
    if experiment_type == "random":
        
        link_id = "_random"
        neuron_index = ""
        seed = 42
        time_samples = 1000
        number_of_synapses = 2
        inputs = GeneratePresynapticInput(number_of_synapses=number_of_synapses, number_of_time_samples=time_samples,seed = seed).generate_input_signal()
        neuron = GenerateSynapsesBatch(number_of_synapses=number_of_synapses, seed = seed).create_neuron()

        delta_t = 0.1
        time_samples = np.linspace(0, time_samples*delta_t-delta_t, time_samples)
        neuron.initial_neuron_potential = 0
        number_of_synapses = len(neuron.synapses_list)
        
    elif experiment_type == "ecg":
        path_to_files = "./mit-bih-arrhythmia-database-1.0.0"

        records = wfdb.get_record_list('mitdb')
        all_signals = []

        for record in records:

            # Read the record
            signal, _ = wfdb.rdsamp(os.path.join(path_to_files, record))
            all_signals.append(signal[:1000,0])
            all_signals.append(signal[:1000,1])

        all_signals = np.array(all_signals)
        link_id = "_ecg"
        neuron_index = ""
        seed = 42
        inputs = all_signals
        time_samples = inputs.shape[-1]
        neuron = GenerateSynapsesBatch(number_of_synapses=inputs.shape[0], seed = seed).create_neuron()
        delta_t = 0.1
        time_samples = np.linspace(0, time_samples*delta_t-delta_t, time_samples)
        neuron.initial_neuron_potential = 0
        number_of_synapses = len(neuron.synapses_list)
    else:
        neuron_index = int(experiment_type)
        link_id = "_"
        hasani_matlab_file = scipy.io.loadmat('./hasani_data.mat')
        number_of_neurons = 19
        get_info = GetInfoMatlabFile(file_matlab=hasani_matlab_file, number_of_neurons=number_of_neurons)
        all_neurons = get_info.get_neurons()
        all_inputs = get_info.get_inputs_per_neurons()
        number_of_samples = 7331
        neuron = all_neurons[neuron_index]
        delta_t = ode_parameters[neuron_index+1]["delta_t"]
        initial_potential = ode_parameters[neuron_index+1]["initial_potential"]
        time_samples = np.linspace(0, number_of_samples*delta_t-delta_t, number_of_samples)
        neuron.initial_neuron_potential = initial_potential
        number_of_synapses = len(neuron.synapses_list)
        inputs = all_inputs[neuron_index]

    
    time_per_synapse_euler = {}
    time_per_synapse_lsoda = {}
    inputs_lsoda = [PiecewiseConstantFunction(time_samples,signal) for signal in inputs]
    for index_synapse in range(number_of_synapses):
        neuron_synapse = copy.deepcopy(neuron)
        neuron_synapse.synapses_list = neuron.synapses_list[:index_synapse+1]
        input_synapses = inputs[0:number_of_synapses]
        input_synapses_lsoda = inputs_lsoda[0:number_of_synapses]
        start_lsoda = time.time()
        solution_lsoda = solve_neuron(neuron=neuron_synapse, time_samples = time_samples, input_signal=np.asarray(input_synapses_lsoda))
        end_lsoda = time.time()
        start_euler = time.time()
        solution_euler = solve_euler_method(neuron=neuron_synapse, time_samples = time_samples, input_signal=np.asarray(input_synapses), delta_t=delta_t)
        end_euler = time.time()
        np.save(f"./ode_solved{link_id}/neuron{link_id + str(neuron_index)}/solution_neuron{link_id + str(neuron_index)}_for_{index_synapse+1}_synapses_lsoda", solution_lsoda)
        np.save(f"./ode_solved{link_id}/neuron{link_id + str(neuron_index)}/solution_neuron{link_id + str(neuron_index)}_for_{index_synapse+1}_synapses_euler", solution_euler)
        time_per_synapse_euler[index_synapse] = end_euler-start_euler
        time_per_synapse_lsoda[index_synapse] = end_lsoda-start_lsoda


# Save the time taken to solve the differential equation
    
    np.save(f"./ode_solved{link_id }/neuron{link_id + str(neuron_index)}/time_per_synapse_neuron{link_id + str(neuron_index)}_for_{number_of_synapses}_synapses_euler", time_per_synapse_euler)
    np.save(f"./ode_solved{link_id }/neuron{link_id + str(neuron_index)}/time_per_synapse_neuron{link_id + str(neuron_index)}_for_{number_of_synapses}_synapses_lsoda", time_per_synapse_lsoda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment type.")
    
    parser.add_argument('experiment_type', type=str, help="Type of experiment (string or integer)")
    
    args = parser.parse_args()
    
    try:
        experiment_type = int(args.experiment_type)
    except ValueError:
        experiment_type = args.experiment_type
    
    solve_ode_neuron(experiment_type)

