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

NB_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 100
UNITS = 8
LEARNING_RATE = 0.05
SEED = 5

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

path_res = f'./results/mnist/'
while True:
    exp_id = str(np.random.randint(0, 100))
    exp_path = path_res + f'exp_{exp_id}_{SEED}/'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        break
logger = tf.get_logger()  
log_file = exp_path+'mnist_experiment.log'
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

def load_mnist(dataset_name='digits', batch_size : int = 64):
    """This function aims to load the MNIST dataset for digits or for fashion.
    Returns:
    train_dataset: tf.data.Dataset: The training dataset.
    val_dataset: tf.data.Dataset: The validation dataset.
    test_dataset: tf.data.Dataset: The test dataset."""

    if dataset_name == 'digits':
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name=='fashion':
        # Load Fashion MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError('Unknown dataset name. Please use digits or fashion MNIST.')

    # Reshape and normalize the data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)).astype('float32') / 255

    # Create train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=SEED)

    def _create_dataset( data_x, data_y, BATCH_SIZE, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data_x))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = _create_dataset(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_dataset = _create_dataset(x_val, y_val, BATCH_SIZE, shuffle=False)
    test_dataset = _create_dataset(x_test, y_test, BATCH_SIZE, shuffle=False)

    return train_dataset, val_dataset, test_dataset

def create_model(cell:str='exact_relu', units:int=64, input_shape:Tuple = (28,28,1), nb_classes:int=10):

    learning_rates = {
        'exact_sigmoid': 0.05,
        'exact_relu': 0.05,
        'exact_dense': 0.005,  
        'ode': 0.05,
        'hasani': 0.001,
    }
    if cell == 'rnn':
        rnn = tf.keras.layers.SimpleRNN(units, return_sequences=True)
    elif cell == 'ode':
        rnn = ODELayer(units, return_sequences=True)
    elif cell == 'exact_sigmoid':
        rnn = ExactLTCLayer(units, return_sequences=True, nonlinearity='sigmoid')
    elif cell == 'exact_relu':
        rnn =  ExactLTCLayer(units, return_sequences=True, nonlinearity='relu')
    elif cell == 'exact_dense':
        rnn = ExactLTCLayer(units, return_sequences=True, nonlinearity='dense')
    elif cell == 'hasani':
        rnn = tf.keras.layers.RNN(CfCCell(units, mode='pure'), return_sequences=True) 
    else:
        raise ValueError(f"Unknown cell type {cell}")
    logger.info(f"Model with {cell} cell: lr={learning_rates[cell]} and config: {rnn.cell.get_config()}")
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((-1,1)),
        rnn,
        tf.keras.layers.BatchNormalization(),   
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(nb_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates[cell]), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.info(f"Number of parameters: {model.count_params()}")
    if model.layers[-1].kernel_regularizer:
        logger.info(f"Kernel regularizer at last dense: {model.layers[-1].kernel_regularizer.l2}")
    else:
        logger.info(f"No kernel regularizer at last dense")
    
    return model


def main(args):

    train_dataset, val_dataset, test_dataset = load_mnist(dataset_name='digits', batch_size=BATCH_SIZE)
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
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=30)
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