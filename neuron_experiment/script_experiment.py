import scipy
import numpy as np
import time
import argparse
import pandas as pd
import wfdb
import os
import copy
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from solutions import ClosedFormApproximation, PiecewiseConstantInputSolutionSeveralSynapses
from grid_search_resampling import FindBestResamplingFactor
from generate_param import GenerateSynapsesBatch, GeneratePresynapticInput, GetInfoMatlabFile

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
}


def solution_neuron(experiment_type='ecg'):
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
    number_max_of_synapses = len(neuron.synapses_list)
    errors_cf = []
    errors_ode_euler = []
    errors_ode_lsoda = []
    times_cf = []
    times_exact = []
    begin = 0
    end = time_samples[-1]
    number_of_steps = time_samples.shape[-1]
    for number_of_synapses in range(1, number_max_of_synapses+1):
        input_per_neuron = inputs[0:number_of_synapses]
        neuron_ = neuron.limit_synapses(number_of_synapses)
        sampling_grid_search_cf = FindBestResamplingFactor(neuron = copy.deepcopy(neuron_), time_samples=time_samples, input_signal=np.asarray(input_per_neuron), number_of_samples=10000, delta_t_min=1e-6, delta_t_max=1, type_of_signal = ClosedFormApproximation, type_of_experiment='ecg')
        solution_sampling_cf = sampling_grid_search_cf.solve()['MSE']
        time_sample_cf = np.linspace(begin, end*solution_sampling_cf - solution_sampling_cf, number_of_steps)
        exact = PiecewiseConstantInputSolutionSeveralSynapses(neuron=neuron_, time_samples=time_samples, input_signal=np.asarray(input_per_neuron))
        start_time_exact = time.time()
        exact_solution = exact.solve()
        end_time_exact = time.time()
        time_exact = end_time_exact - start_time_exact
        cf = ClosedFormApproximation(neuron=neuron_, time_samples=time_sample_cf, input_signal=np.asarray(input_per_neuron))
        start_time_cf = time.time()
        cf_solution = cf.solve()
        end_time_cf = time.time()
        time_cf = end_time_cf - start_time_cf
        ode_euler = np.load(f'./ode_solved{link_id}/neuron{link_id + str(neuron_index)}/solution_neuron{link_id + str(neuron_index)}_for_{len(neuron_.synapses_list)}_synapses_euler.npy')
        ode_lsoda = np.load(f'./ode_solved{link_id}/neuron{link_id + str(neuron_index)}/solution_neuron{link_id + str(neuron_index)}_for_{len(neuron_.synapses_list)}_synapses_lsoda.npy')
        mse_cf = mean_squared_error(exact_solution,cf_solution)
        mse_ode_euler = mean_squared_error(exact_solution, ode_euler)
        mse_ode_lsoda = mean_squared_error(exact_solution, ode_lsoda)
        errors_cf.append(mse_cf)
        errors_ode_euler.append(mse_ode_euler)
        errors_ode_lsoda.append(mse_ode_lsoda)
        times_cf.append(time_cf)
        times_exact.append(time_exact)
        
    synapses_index = [i for i in range(1, number_max_of_synapses+1)]
    plt.figure()
    plt.plot(cf_solution, label='Closed form approximation')
    plt.plot(exact_solution, label='Exact solution')
    plt.plot(ode_euler, label='Euler')
    plt.plot(ode_lsoda, label='Lsoda')
    plt.legend()
    plt.title('Comparison of the solutions')
    plt.xlabel('Time samples')
    plt.ylabel('Potential')
    plt.savefig(f'./results{link_id}/comparison_neuron{link_id + str(neuron_index)}_synapses_{number_max_of_synapses}.png')

    plt.figure()
    plt.plot(synapses_index,errors_cf, label='Closed form approximation')
    plt.plot(synapses_index,errors_ode_euler, label='Euler solution')
    plt.plot(synapses_index,errors_ode_lsoda, label='Lsoda solution')
    plt.legend()
    # X axis should be integers
    plt.xticks(np.arange(1, number_max_of_synapses+1, dtype=int))
    plt.title('MSE for different methods (compared to the exact solution)')
    plt.xlabel('Number of synapses')    
    plt.ylabel('MSE')
    plt.savefig(f'./results{link_id}/MSE_comparison_neuron{link_id + str(neuron_index)}_synapses_{number_max_of_synapses}.png')

    time_ode_euler = np.load(f'./ode_solved{link_id}/neuron{link_id + str(neuron_index)}/time_per_synapse_neuron{link_id + str(neuron_index)}_for_{number_max_of_synapses}_synapses_euler.npy', allow_pickle=True)
    time_ode_euler = np.asanyarray(time_ode_euler)
    time_ode_euler = time_ode_euler.reshape(-1)
    time_ode_euler = list(time_ode_euler[0].values())
    time_ode_lsoda = np.load(f'./ode_solved{link_id}/neuron{link_id + str(neuron_index)}/time_per_synapse_neuron{link_id + str(neuron_index)}_for_{number_max_of_synapses}_synapses_lsoda.npy', allow_pickle=True)
    time_ode_lsoda = np.asanyarray(time_ode_lsoda)
    time_ode_lsoda = time_ode_lsoda.reshape(-1)
    time_ode_lsoda = list(time_ode_lsoda[0].values())

    plt.figure() 
    plt.plot(synapses_index,times_cf, label='Closed form approximation')
    plt.plot(synapses_index,times_exact, label='Exact solution')
    plt.plot(synapses_index,time_ode_euler, label='Euler method')
    plt.plot(synapses_index,time_ode_lsoda, label='Lsoda method')
    plt.legend()
    plt.xticks(np.arange(1, number_max_of_synapses+1, dtype=int))
    plt.title('Time of ODE solving for different methods')
    plt.xlabel('Number of synapses')
    plt.ylabel('Time in seconds')
    plt.savefig(f'./results{link_id}/time_comparison_neuron{link_id + str(neuron_index)}_synapses_{number_max_of_synapses}.png')

    error_sc_cf = [f"{x:.2e}" for x in errors_cf]
    error_sc_euler = [f"{x:.2e}" for x in errors_ode_euler]
    error_sc_lsoda = [f"{x:.2e}" for x in errors_ode_lsoda]

    time_cf_sc = [f"{x:.2e}" for x in times_cf]
    time_exact_sc = [f"{x:.2e}" for x in times_exact]
    time_ode_euler_sc = [f"{x:.2e}" for x in time_ode_euler]
    time_ode_lsoda_sc = [f"{x:.2e}" for x in time_ode_lsoda]
    # Put in scientific notation for the number 
    data_mse = {'Number of synapses': np.asarray(range(1, number_max_of_synapses+1), dtype=int), 'MSE closed form approximation': error_sc_cf, 'MSE Euler method': error_sc_euler,'MSE Lsoda method': error_sc_lsoda}
    df_mse = pd.DataFrame(data_mse)
    df_mse.to_csv(f'./results{link_id}/MSE_comparison_neuron{link_id + str(neuron_index)}_synapses_{number_max_of_synapses}.csv')
    # Make a table with the results
    data_time = { 'Number of synapses':range(1, number_max_of_synapses+1),'Time closed form approximation (s)': time_cf_sc, 'Time exact solution (s)': time_exact_sc, 'Time Euler method (s)': time_ode_euler_sc,'Time Lsoda method (s)': time_ode_lsoda_sc}
    df_time = pd.DataFrame(data_time)
    df_time.to_csv(f'./results{link_id}/time_comparison_neuron{link_id + str(neuron_index)}_synapse_{number_max_of_synapses}_synapses.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment type.")
    
    # Add argument that can be either a string or an integer.
    parser.add_argument('experiment_type', type=str, help="Type of experiment (string or integer)")
    
    args = parser.parse_args()
    
    # Convert to integer if possible
    try:
        experiment_type = int(args.experiment_type)
    except ValueError:
        experiment_type = args.experiment_type

    solution_neuron(experiment_type=experiment_type)