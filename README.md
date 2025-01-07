# Exact LNN
Exact Implementation of Closed-Form Liquid Neural Networks With Arbitrary Precision

Appendix: [https://github.com/emilierp/exact_lnn/Appendix.pdf](https://github.com/emilierp/exact_lnn/blob/main/Appendix.pdf)

# Single neuron

Solve the ODE integration by running:
```bash
python3 neuron_experiment/script_ode_solver.py ecg 
```

To visualize the results: 
```bash
python3 neuron_experiment/script_experiment.py ecg
```

# Neural networks

The LNN layers are defined in `ltc_src/layers.py`. 

The experiments can be ran using the scripts in the `lnn_experiments` folder. For example, train on the HAR dataset by running: 
```bash
python3 lnn_experiments/har.py
```

# Datasets

To download the dataset: 

- MIT-BIH (ECG): https://www.physionet.org/content/mitdb/1.0.0/
- MNIST: https://yann.lecun.com/exdb/mnist/
- HAR: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones 
- PAR: https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity


