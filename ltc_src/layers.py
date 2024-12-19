"""
    Definition of layers: Euler, CfC and Exact
    Implementation of CfC from https://github.com/mlech26l/ncps/blob/master/ncps/keras/cfc_cell.py
"""

import tensorflow as tf
from typing import Tuple, Literal

class BaseCell(tf.keras.layers.Layer):
    def __init__(self, units:int=32, omega:float=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.omega = omega
        self.state_size = units
        
        assert self.omega > 0, "Omega must be greater than 0 for numerical stability."

    def build(self, input_shape: Tuple):
        raise NotImplementedError("Build method not implemented.")
        
    def call(self, inputs: tf.Tensor, states: tf.Tensor):
        raise NotImplementedError("Call method not implemented.")
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.tile(self.x0[tf.newaxis, :], [batch_size, 1])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "omega": self.omega}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)
        

class ODECell(BaseCell):
    def __init__(self, units:int=32, omega:float=0.1, dt:float=0.1, ode_unfolds:int=2, **kwargs):
        super().__init__(units, omega, **kwargs)
        self.dt = dt
        self.ode_unfolds = ode_unfolds

    def build(self, input_shape: Tuple):
        self.input_dim = input_shape[-1]
        self.A = self.add_weight(shape=(self.units, self.input_dim),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='A')
        self.sigma = self.add_weight(shape=(self.units, self.input_dim),
                                     initializer='glorot_uniform',
                                     trainable=True,
                                     name='sigma')
        self.mu = self.add_weight(shape=(self.units, self.input_dim),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  name='mu')
        self.x0 = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True,
                                  name='x0')

    def call(self, inputs: tf.Tensor, states: tf.Tensor):   
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            tt_ = inputs[1]
            inputs = inputs[0]
            t = tf.reshape(tt_, [-1, 1])
        else: 
            t = 1.0

        x_prev = states[0] if isinstance(states, (list, tuple)) else states
        dt_unfold = self.dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = tf.math.sigmoid(self.sigma * (tf.expand_dims(inputs, 1) - self.mu))
            dx = -self.omega * x_prev + tf.reduce_sum(f * (self.A - tf.expand_dims(x_prev, -1)), axis=-1)
            x_prev = x_prev + dt_unfold * dx
        return x_prev, [x_prev] 


class ExactLTCCell(BaseCell):
    def __init__(self, units:int=32, omega:float=0.1, nonlinearity:Literal['relu', 'sigmoid']='relu', **kwargs):
        super().__init__(units, omega, **kwargs)
        self.nonlinearity = nonlinearity    
        if nonlinearity == 'relu':
            self.f_func = tf.nn.relu
        elif nonlinearity == 'sigmoid':
            self.f_func = tf.nn.sigmoid
        elif nonlinearity == 'dense':
            self.f_func = None
        else: 
            raise ValueError(f"Unknown nonlinearity {nonlinearity}")
        
    def build(self, input_shape: Tuple):
        self.input_dim = input_shape[-1]
        self.A = self.add_weight(shape=(self.units, self.input_dim),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='A')
        
        if self.nonlinearity == 'dense':
            self.f_func = tf.keras.layers.Dense(
                self.units*self.input_dim,
                activation='relu',
                kernel_initializer='glorot_uniform',
                trainable=True,
                name='f_func', 
            )
            self.f_func.build((None, self.input_dim))
        else:
            self.sigma = self.add_weight(shape=(self.units, self.input_dim),
                                        initializer='glorot_uniform',
                                        trainable=True,
                                        name='sigma')
            self.mu = self.add_weight(shape=(self.units, self.input_dim),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    name='mu')
        self.x0 = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True,
                                  name='x0')

    def call(self, inputs: tf.Tensor, states: tf.Tensor): 
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1: # irregular sampling 
            tt_ = inputs[1]
            inputs = inputs[0]
            t = tf.reshape(tt_, [-1, 1])
        else: # regular sampling
            t = 1.0
        
        x_prev = states[0] if isinstance(states, (list, tuple)) else states
        if self.nonlinearity == 'dense':
            f = self.f_func(inputs)
            f = tf.reshape(f, (tf.shape(inputs)[0], self.units, self.input_dim))
        else:
            f = self.f_func(self.sigma * (tf.expand_dims(inputs, 1) - self.mu))
        u = f / (self.omega + tf.reduce_sum(f, axis=-1, keepdims=True))
        gamma = tf.exp(-t * (self.omega + tf.reduce_sum(f, axis=-1)))
        x = gamma * (x_prev - tf.reduce_sum(self.A * u, axis=-1)) + tf.reduce_sum(self.A * u, axis=-1)
        return x, [x]
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "nonlinearity": self.nonlinearity}
    

class BaseLayer(tf.keras.layers.RNN):
        def __init__(self, units:int=32, omega=0.1, return_sequences: bool = False, return_state: bool = False, cell=None, **kwargs):
            if cell is None:
                cell = BaseCell(units, omega)
            self.cell = cell
            super().__init__(cell=cell, return_sequences=return_sequences, return_state=return_state, **kwargs)

        def call(self, sequences, initial_state=None, **kwargs):
            if sequences.ndim == 4:
                sequences = tf.reshape(sequences, [-1, sequences.shape[1] * sequences.shape[2], sequences.shape[3]])
            return super().call(sequences, initial_state=initial_state, **kwargs)
        
        def get_config(self):
            base_config = super().get_config()
            config = {"cell": tf.keras.layers.serialize(self.cell)}
            base_config.update(config)
            return base_config
        
        @classmethod
        def from_config(cls, config):
            cell_config = config.pop("cell", None)
            cell = tf.keras.layers.deserialize(cell_config) if cell_config else None
            return cls(cell=cell, **config)

@tf.keras.utils.register_keras_serializable(package="exact_ltc", name="ODELayer")
class ODELayer(BaseLayer):
    def __init__(self, units:int=32, omega=0.1, dt:float=0.1, ode_unfolds:int=3, return_sequences: bool = False, return_state: bool = False, cell=None, **kwargs):
        if cell is None:
            cell = ODECell(units, omega, dt, ode_unfolds)
        super().__init__(units, omega, return_sequences, return_state, cell, **kwargs)

@tf.keras.utils.register_keras_serializable(package="exact_ltc", name="ExactLayer")
class ExactLTCLayer(BaseLayer):
    def __init__(self, units:int=32, omega=0.1, nonlinearity:Literal['relu', 'sigmoid']='relu', 
                 return_sequences: bool = False, return_state: bool = False, cell=None, **kwargs):
        if cell is None: 
            cell = ExactLTCCell(units, omega, nonlinearity)
        super().__init__(units, omega, return_sequences, return_state, cell, **kwargs)



# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
@tf.keras.utils.register_keras_serializable(package="", name="lecun_tanh")
def lecun_tanh(x):
    return 1.7159 * tf.keras.activations.tanh(0.666 * x)

@tf.keras.utils.register_keras_serializable(package="ncps", name="CfCCell")
class CfCCell(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        mode="pure",
        activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.1,
        sparsity_mask=None,
        **kwargs,
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps.
            To get a full RNN that can process sequences,
            see `ncps.tf.keras.CfC` or wrap the cell with a `tf.keras.layers.RNN <https://www.tensorflow.org/api_docs/python/tf/tf.keras/layers/RNN>`_.


        :param units: Number of hidden units
        :param input_sparsity:
        :param recurrent_sparsity:
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.units = units
        self.sparsity_mask = sparsity_mask
        if sparsity_mask is not None:
            # No backbone is allowed
            if backbone_units > 0:
                raise ValueError("If sparsity of a CfC cell is set, then no backbone is allowed")

        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {str(allowed_modes)}")
        self.mode = mode
        self.backbone_fn = None
        self._activation = lecun_tanh
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout
        self._cfc_layers = []

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple) or isinstance(input_shape[0], tf.keras.KerasTensor):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        if self._backbone_layers > 0:
            backbone_layers = []
            for i in range(self._backbone_layers):
                backbone_layers.append(tf.keras.layers.Dense(self._backbone_units, self._activation, name=f"backbone{i}"))
                backbone_layers.append(tf.keras.layers.Dropout(self._backbone_dropout))

            self.backbone_fn = tf.keras.models.Sequential(backbone_layers)
            self.backbone_fn.build((None, self.state_size + input_dim))
            cat_shape = int(self._backbone_units)
        else:
            cat_shape = int(self.state_size + input_dim)

        self.ff1_kernel = self.add_weight(
            shape=(cat_shape, self.state_size),
            initializer="glorot_uniform",
            name="ff1_weight",
        )
        self.ff1_bias = self.add_weight(
            shape=(self.state_size,),
            initializer="zeros",
            name="ff1_bias",
        )

        if self.mode == "pure":
            self.w_tau = self.add_weight(
                shape=(1, self.state_size),
                initializer=tf.keras.initializers.Zeros(),
                name="w_tau",
            )
            self.A = self.add_weight(
                shape=(1, self.state_size),
                initializer=tf.keras.initializers.Ones(),
                name="A",
            )
        else:
            self.ff2_kernel = self.add_weight(
                shape=(cat_shape, self.state_size),
                initializer="glorot_uniform",
                name="ff2_weight",
            )
            self.ff2_bias = self.add_weight(
                shape=(self.state_size,),
                initializer="zeros",
                name="ff2_bias",
            )

            self.time_a = tf.keras.layers.Dense(self.state_size, name="time_a")
            self.time_b = tf.keras.layers.Dense(self.state_size, name="time_b")
            input_shape = (None, self.state_size + input_dim)
            if self._backbone_layers > 0:
                input_shape = self.backbone_fn.output_shape
            self.time_a.build(input_shape)
            self.time_b.build(input_shape)
        self.built = True

    def call(self, inputs, states, **kwargs):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = tf.keras.ops.reshape(t, [-1, 1])
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            t = kwargs.get("time") or 1.0
        # x = tf.keras.layers.Concatenate()([inputs, states[0]])
        hidden_state = states[0] if isinstance(states, (list, tuple)) else states
        x = tf.keras.layers.Concatenate()([hidden_state, inputs])
        if self._backbone_layers > 0:
            x = self.backbone_fn(x)
        if self.sparsity_mask is not None:
            ff1_kernel = self.ff1_kernel * self.sparsity_mask
            ff1 = tf.keras.ops.matmul(x, ff1_kernel) + self.ff1_bias
        else:
            ff1 = tf.keras.ops.matmul(x, self.ff1_kernel) + self.ff1_bias
        if self.mode == "pure":
            # Solution
            new_hidden = (
                -self.A
                * tf.keras.ops.exp(-t * (tf.keras.ops.abs(self.w_tau) + tf.keras.ops.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Cfc
            if self.sparsity_mask is not None:
                ff2_kernel = self.ff2_kernel * self.sparsity_mask
                ff2 = tf.keras.ops.matmul(x, ff2_kernel) + self.ff2_bias
            else:
                ff2 = tf.keras.ops.matmul(x, self.ff2_kernel) + self.ff2_bias
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = tf.keras.activations.sigmoid(-t_a * t + t_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, [new_hidden]

    def get_config(self):
        config = {
            "units": self.units,
            "mode": self.mode,
            "activation": self._activation,
            "backbone_units": self._backbone_units,
            "backbone_layers": self._backbone_layers,
            "backbone_dropout": self._backbone_dropout,
            "sparsity_mask": self.sparsity_mask,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros((batch_size, self.units))