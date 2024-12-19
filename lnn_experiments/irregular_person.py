"""
    Link to dataset: https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity
    Code for data pre-processing adapted from: https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/person.py
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
from ltc_src.irregular_layers import *

NB_CLASSES = 7
SEQ_LEN = 32
INPUT_DIM = 7
BATCH_SIZE = 64
EPOCHS = 100
UNITS = 64
LEARNING_RATE = 0.05
SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

path_res = f'./results/person/'
while True:
    exp_id = str(np.random.randint(0, 100))
    exp_path = path_res + f'exp_{exp_id}_{SEED}/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        break
logger = tf.get_logger()  
log_file = exp_path+'person_experiment.log'
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


class PersonData:
    class_map = {
        "lying down": 0,
        "lying": 0,
        "sitting down": 1,
        "sitting": 1,
        "standing up from lying": 2,
        "standing up from sitting": 2,
        "standing up from sitting on the ground": 2,
        "walking": 3,
        "falling": 4,
        "on all fours": 5,
        "sitting on the ground": 6,
    }

    sensor_ids = {
        "010-000-024-033": 0,
        "010-000-030-096": 1,
        "020-000-033-111": 2,
        "020-000-032-221": 3,
    }

    def __init__(self, seq_len=32, batch_size=64):

        self.seq_len = seq_len
        self.num_classes = 7
        all_x, all_t, all_y = self.load_crappy_formated_csv()
        all_x, all_t, all_y = self.cut_in_sequences(
            all_x, all_t, all_y, seq_len=seq_len, inc=seq_len // 2
        )

        print("all_x.shape: ", str(all_x.shape))
        print("all_t.shape: ", str(all_t.shape))
        print("all_y.shape: ", str(all_y.shape))
        total_seqs = all_x.shape[0]
        print("Total number of sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(98841).permutation(total_seqs)
        test_size = int(0.2 * total_seqs)

        test_x = all_x[permutation[:test_size]]
        test_y = all_y[permutation[:test_size]]
        test_t = all_t[permutation[:test_size]]
        train_x = all_x[permutation[test_size:]]
        train_t = all_t[permutation[test_size:]]
        train_y = all_y[permutation[test_size:]]

        permutation = np.random.RandomState(98841).permutation(train_x.shape[0])
        val_size = int(0.1 * train_x.shape[0])

        val_x = train_x[permutation[:val_size]]
        val_y = train_y[permutation[:val_size]]
        val_t = train_t[permutation[:val_size]]
        train_x = train_x[permutation[val_size:]]
        train_t = train_t[permutation[val_size:]]
        train_y = train_y[permutation[val_size:]]

        self.feature_size = int(train_x.shape[-1])

        print("train_x.shape: ", str(train_x.shape))
        print("train_t.shape: ", str(train_t.shape))
        print("train_y.shape: ", str(train_y.shape))
        print("Total number of train sequences: {}".format(train_x.shape[0]))
        print("Total number of validation sequences: {}".format(val_x.shape[0]))
        print("Total number of test  sequences: {}".format(test_x.shape[0]))

        def _create_dataset( data_x, data_y, BATCH_SIZE, shuffle=False):
            dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(data_x))
            dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            return dataset

        self.train_dataset = _create_dataset((train_x, train_t), train_y, batch_size, True)
        self.val_dataset = _create_dataset((val_x, val_t), val_y, batch_size)
        self.test_dataset = _create_dataset((test_x, test_t), test_y, batch_size)

    def load_crappy_formated_csv(self):

        all_x = []
        all_y = []
        all_t = []

        series_x = []
        series_t = []
        series_y = []

        last_millis = None
        if not os.path.isfile("../data/person/ConfLongDemo_JSI.txt"):
            print("ERROR: File '../data/person/ConfLongDemo_JSI.txt' not found")
            print("Please execute the command")
            print("source download_dataset.sh")
            import sys

            sys.exit(-1)
        with open("../data/person/ConfLongDemo_JSI.txt", "r") as f:
            current_person = "A01"

            for line in f:
                arr = line.split(",")
                if len(arr) < 6:
                    break
                if arr[0] != current_person:
                    # Enqueue and reset
                    series_x = np.stack(series_x, axis=0)
                    series_t = np.stack(series_t, axis=0)
                    series_y = np.array(series_y, dtype=np.int32)
                    all_x.append(series_x)
                    all_t.append(series_t)
                    all_y.append(series_y)
                    last_millis = None
                    series_x = []
                    series_y = []
                    series_t = []

                millis = np.int64(arr[2]) / (100 * 1000)
                # 100ms will be normalized to 1.0
                millis_mapped_to_1 = 10.0
                if last_millis is None:
                    elapsed_sec = 0.05
                else:
                    elapsed_sec = float(millis - last_millis) / 1000.0
                elapsed = elapsed_sec * 1000 / millis_mapped_to_1

                last_millis = millis
                current_person = arr[0]
                sensor_id = self.sensor_ids[arr[1]]
                label_col = self.class_map[arr[7].replace("\n", "")]
                feature_col_2 = np.array(arr[4:7], dtype=np.float32)
                # Last 3 entries of the feature vector contain sensor value

                # First 4 entries of the feature vector contain sensor ID
                feature_col_1 = np.zeros(4, dtype=np.float32)
                feature_col_1[sensor_id] = 1

                feature_col = np.concatenate([feature_col_1, feature_col_2])
                series_x.append(feature_col)
                series_t.append(elapsed)
                series_y.append(label_col)

        return all_x, all_t, all_y

    def cut_in_sequences(self, all_x, all_t, all_y, seq_len, inc=1):

        sequences_x = []
        sequences_t = []
        sequences_y = []

        for i in range(len(all_x)):
            x, t, y = all_x[i], all_t[i], all_y[i]

            for s in range(0, x.shape[0] - seq_len, inc):
                start = s
                end = start + seq_len
                sequences_x.append(x[start:end])
                sequences_t.append(t[start:end])
                sequences_y.append(y[start:end])

        return (
            np.stack(sequences_x, axis=0),
            np.stack(sequences_t, axis=0).reshape([-1, seq_len, 1]),
            np.stack(sequences_y, axis=0),
        )

def create_model(layer:str='exact_relu', units:int=64, input_shape:Tuple = (SEQ_LEN,INPUT_DIM), nb_classes:int=7):

    learning_rates = {
        'exact_sigmoid': 0.05,
        'exact_relu': 0.05,
        'exact_dense': 0.005,  
        'ode': 0.05,
        'hasani': 0.001,
    }
    if layer == 'exact_sigmoid':
        cell = ExactLTCCell(units, nonlinearity='sigmoid')
    elif layer == 'exact_relu':
        cell = ExactLTCCell(units, nonlinearity='relu')
    elif layer == 'exact_dense':
        cell = ExactLTCCell(units, nonlinearity='dense')
    elif layer == 'hasani':
        cell = CfCCell(units, mode='pure')
    elif layer == 'ode':
        cell = ODECell(units)
    else:
        raise ValueError(f"Unknown cell type {layer}")
    logger.info(f"Model with {layer} cell: lr={learning_rates[layer]} and config: {cell.get_config()}")

    feature_input = tf.keras.Input(shape=(input_shape[0], input_shape[1]), name="input")
    time_input = tf.keras.Input(shape=(input_shape[0], 1), name="time")

    rnn = IrregularRNN(cell, return_sequences=True)  
    x = rnn((feature_input, time_input))    
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nb_classes, activation='softmax'))(x)

    model = tf.keras.Model(inputs=[feature_input, time_input], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rates[layer]),
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    logger.info(f"Number of parameters: {model.count_params()}")
    
    if model.layers[-1].layer.kernel_regularizer:
        logger.info(f"Kernel regularizer at last dense: {model.layers[-1].layer.kernel_regularizer.l2}")
    else:
        logger.info(f"No kernel regularizer at last dense")
    
    return model


def main(args):

    data = PersonData(batch_size=args.batch_size)
    train_dataset = data.train_dataset
    val_dataset = data.val_dataset
    test_dataset = data.test_dataset
    print("\nCreating dataset")

    loss = {}
    accuracy = {}
    training_time = {}
    early_stopped = {}
    learning_rates = {}
    
    layers = ['exact_sigmoid', 'exact_relu', 'exact_dense', 'hasani', 'ode']
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
    # df = pd.DataFrame({'Test loss': loss, 'Test accuracy': accuracy}, index=layers).reset_index().rename(columns={'index': 'LTC layer'})
 
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