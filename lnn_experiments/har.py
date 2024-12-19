"""
    Link to dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones 
    Code for data pre-processing adapted from: https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/har.py
"""

import os
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import io
import time

from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Literal
from tensorflow.keras.layers import Input, Bidirectional, RNN, Dense, TimeDistributed, LSTM, Dropout
from ltc_src.layers import *

SEQ_LEN = 16
INPUT_DIM = 561
NB_CLASSES = 6
BATCH_SIZE = 64
EPOCHS = 100
UNITS = 8
LEARNING_RATE = 0.05
SEED = 0


np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

path_res = f'./results/har/'
while True:
    exp_id = str(np.random.randint(0, 100))
    exp_path = path_res + f'exp_{exp_id}_{SEED}/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        break
logger = tf.get_logger()  
log_file = exp_path+'har_experiment.log'
logging.basicConfig(filename=log_file, level=logging.INFO)
logger.info(f"Saving results to: {exp_path}")
logger.info(f"Seed: {SEED}")


class ModelHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.lrs = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.lrs.append(self.model.optimizer.learning_rate.numpy())


def cut_in_sequences(data_x, data_y, SEQ_LEN, inc=None):
    if inc is None:
        inc = SEQ_LEN
    num_samples = (len(data_x) - SEQ_LEN) // inc + 1
    sequences_x = np.array([data_x[i * inc:i * inc + SEQ_LEN] for i in range(num_samples)])
    sequences_y = np.array([data_y[i * inc:i * inc + SEQ_LEN] for i in range(num_samples)])
    return sequences_x, sequences_y

class HarData:
    def __init__(self, SEQ_LEN=16, BATCH_SIZE=16):
        train_x = np.loadtxt("../data/UCI HAR Dataset/train/X_train.txt")
        train_y = (np.loadtxt("../data/UCI HAR Dataset/train/y_train.txt") - 1).astype(np.int32)
        test_x = np.loadtxt("../data/UCI HAR Dataset/test/X_test.txt")
        test_y = (np.loadtxt("../data/UCI HAR Dataset/test/y_test.txt") - 1).astype(np.int32)

        train_x, train_y = cut_in_sequences(train_x, train_y, SEQ_LEN)
        test_x, test_y = cut_in_sequences(test_x, test_y, SEQ_LEN, inc=8)

        logger.info(f"Total number of training sequences: {train_x.shape[0]}")
        permutation = np.random.RandomState(893429).permutation(train_x.shape[0])
        valid_size = int(0.1 * train_x.shape[0])
        logger.info(f"Validation split: {valid_size}, training split: {train_x.shape[0] - valid_size}")

        val_x = train_x[permutation[:valid_size]]
        val_y = train_y[permutation[:valid_size]]
        train_x = train_x[permutation[valid_size:]]
        train_y = train_y[permutation[valid_size:]]

        logger.info(f"Total number of test sequences: {test_x.shape[0]}")

        self.train_dataset = self._create_dataset(train_x, train_y, BATCH_SIZE, shuffle=True)
        self.val_dataset = self._create_dataset(val_x, val_y, BATCH_SIZE, shuffle=False)
        self.test_dataset = self._create_dataset(test_x, test_y, BATCH_SIZE, shuffle=False)

    def _create_dataset(self, data_x, data_y, BATCH_SIZE, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data_x))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

def create_model(cell:str='exact_relu', units:int=8, input_shape:Tuple = (SEQ_LEN, INPUT_DIM), nb_classes:int=6):

    learning_rates = {
        'exact_sigmoid': 0.05,
        'exact_relu': 0.05,
        'exact_dense': 0.005,  
        'ode': 0.05,
        'hasani': 0.001,
    }
    if cell == 'ode':
        rnn = ODELayer(units, dt=0.001, return_sequences=True)
    elif cell == 'exact_sigmoid':
        rnn = ExactLTCLayer(units, nonlinearity='sigmoid', return_sequences=True)
    elif cell == 'exact_relu':
        rnn = ExactLTCLayer(units, nonlinearity='relu', return_sequences=True)
    elif cell == 'exact_dense':
        rnn = ExactLTCLayer(units, nonlinearity='dense', return_sequences=True)
    elif cell == 'exact4':
        rnn = tf.keras.layers.RNN(ExactLTCCell2(units, nonlinearity='relu',), return_sequences=True)   
    elif cell == 'hasani':
        rnn = tf.keras.layers.RNN(CfCCell(units, mode='pure'), return_sequences=True) 
    else:
        raise ValueError(f"Unknown cell type {cell}")
    logger.info(f"Model with {cell} cell: lr={learning_rates[cell]} and config: {rnn.cell.get_config()}")

    model = Sequential([
        Input(shape=input_shape),
        rnn,
        TimeDistributed(Dense(nb_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates[cell]), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    logger.info(f"Number of parameters: {model.count_params()}")
    if model.layers[-1].layer.kernel_regularizer:
        logger.info(f"Kernel regularizer at last dense: {model.layers[-1].layer.kernel_regularizer.l2}")

    return model

def main(args):

    data = HarData(SEQ_LEN=SEQ_LEN, BATCH_SIZE=BATCH_SIZE)
    print("\nCreating dataset")
    train_dataset, val_dataset, test_dataset = data.train_dataset, data.val_dataset, data.test_dataset

    loss = {}
    accuracy = {}
    training_time = {}
    early_stopped = {}
    learning_rates = {}
    
    layers = ['exact_sigmoid', 'exact_relu', 'exact_dense', 'hasani', 'ode']
    # layers = ['exact4']
    for layer in layers:
        logger.info(f"\nExperiment with {layer}")
        model = create_model(layer, args.units)
        buffer = io.StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + "\n"))
        logger.info("Model summary:\n%s", buffer.getvalue())
        hist_callback = ModelHistory()
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs,
                            callbacks=[
                                early_stopping,
                                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, min_lr=1e-5),
                                hist_callback,]
                            )
        for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(
            zip(history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']), 
            1):
            logger.info("Epoch %d: train_loss=%.4f, val_loss=%.4f, train_acc=%.4f, val_acc=%.4f", epoch, train_loss, val_loss, train_acc, val_acc)
        test_loss, test_acc = model.evaluate(test_dataset)
        logger.info(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

        
        try: 
            model.save(exp_path + f'model_{layer}.h5')   
            np.save(exp_path + f'history_{layer}', history.history)
        except:
            pass
        loss[layer] = test_loss
        accuracy[layer] = test_acc
        training_time[layer] = hist_callback.times[:5]
        learning_rates[layer] = hist_callback.lrs
        early_stopped[layer] = early_stopping.stopped_epoch

    df = pd.DataFrame({'Test loss': loss, 'Test accuracy': accuracy, 'Training Time': training_time}, index=layers).reset_index().rename(columns={'index': 'LTC layer'})

    df['Batch Size'] = args.batch_size
    df['Total Epochs'] = args.epochs
    df['Units'] = args.units
    df['Early Stopped'] = early_stopped.values()
    df['Learning Rates'] = learning_rates.values()
    df.to_csv(exp_path + f'results_{args.units}.csv')

    return test_loss, test_acc
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that takes an integer argument.")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--units', type=int, default=UNITS, help="Units in LTC layer")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    test_loss, test_acc = main(args)