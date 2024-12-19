import tensorflow as tf
from typing import Tuple, Literal

class IrregularRNN(tf.keras.layers.Layer):
    def __init__(self, cell, return_sequences: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            input_shape = input_shape[0]
        self.cell.build(input_shape)
        self.sequence_length = input_shape[1]   
        super().build(input_shape)

    def call(self, inputs, states=None):
        if isinstance(inputs, tuple):
            features, time_steps = inputs
        else:
            features = inputs
            time_steps = None
        batch_size = tf.shape(features)[0]

        if states is None:
            states = self.cell.get_initial_state(inputs=features, batch_size=batch_size, dtype=features.dtype)

        outputs = []
        for t in range(self.sequence_length):
            input_t = features[:, t, :]
            if time_steps is None:
                output, states = self.cell(input_t, states)
            else:
                time_t = time_steps[:, t]
                output, states = self.cell((input_t, time_t), states)
            if self.return_sequences:
                outputs.append(output)

        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return output

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.cell.units, "return_sequences": self.return_sequences, "cell": self.cell}
    
    @classmethod
    def from_config(cls, config):
        cell = config.pop("cell")
        return cls(cell.units, cell.omega, cell=cell, **config)
