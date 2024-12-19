"""This file aims to provide function to approximate temporal sampled input data either into constant piecewise function or linear piecewise function."""
import numpy as np


class PiecewiseConstantFunction:
    """This class aims to create piecewise constant function from sampled data points."""
    def __init__(self, time_samples, data_samples):
        self.time_samples = time_samples
        self.data_samples = data_samples
        self.piecewise_constant_function = self.approximate()
        
    
    def approximate(self):
        """
        Approximate temporal sampled input data into piecewise constant function.
        :return: list of piecewise constant function
        """
        # Initialize piecewise constant function
        piecewise_constant_function = []

        # Initialize start time and data
        start_time = self.time_samples[0]
        start_data = self.data_samples[0]   

        # Iterate over time samples
        for i in range(1, len(self.time_samples)):
            # If data sample is different from previous data sample
            if self.data_samples[i] != start_data:
                # Append piecewise constant function
                piecewise_constant_function.append((start_time, start_data))

                # Update start time and data
                start_time = self.time_samples[i]
                start_data = self.data_samples[i]

        # Append last piecewise constant function
        piecewise_constant_function.append((start_time, start_data))

        return piecewise_constant_function
    
    def __call__(self, time_values_array):
        """This function aims to return the piecewise constant function values evaluated at given time values."""
        return np.array([self.evaluate(time_value) for time_value in time_values_array])
    

    def evaluate(self, time_value):
        """This function aims to return the piecewise constant function value evaluated at given time."""
        try:
            assert time_value >= self.time_samples[0] and time_value <= self.time_samples[-1], "Time value is out of range."
        except AssertionError:
            time_value = self.time_samples[-1]
        for i in range(len(self.piecewise_constant_function)-1):
            if self.piecewise_constant_function[i][0] <= time_value < self.piecewise_constant_function[i + 1][0] :
                return self.piecewise_constant_function[i][1]
        # Last value of piecewise constant function
        if time_value >= self.piecewise_constant_function[-1][0]:
            return self.piecewise_constant_function[-1][1]
        return 0
    

class PiecewiseLinearFunction:
    """This class aims to create piecewise linear function from sampled data points."""
    def __init__(self, time_samples, data_samples):
        self.time_samples = time_samples
        self.data_samples = data_samples
        self.piecewise_linear_function = self.approximate()
        
    
    def approximate(self):
        """
        Approximate temporal sampled input data into piecewise linear function.
        :return: list of piecewise linear function
        """
        # Initialize piecewise linear function
        piecewise_linear_function = []

        # Initialize start time and data
        start_time = self.time_samples[0]
        start_data = self.data_samples[0]   

        # Iterate over time samples
        for i in range(1, len(self.time_samples)):
            # If data sample is different from previous data sample
            if self.data_samples[i] != start_data:
                # Append piecewise linear function
                piecewise_linear_function.append((start_time, start_data))

                # Update start time and data
                start_time = self.time_samples[i]
                start_data = self.data_samples[i]

        # Append last piecewise linear function
        piecewise_linear_function.append((start_time, start_data))

        return piecewise_linear_function
    
    def __call__(self, time_values_array):
        """This function aims to return the piecewise linear function values evaluated at given time values."""
        return np.array([self.evaluate(time_value) for time_value in time_values_array])
    

    def evaluate(self, time_value):
        """This function aims to return the piecewise linear function value evaluated at given time."""
        try:
            assert time_value >= self.time_samples[0] and time_value <= self.time_samples[-1], "Time value is out of range."
        except AssertionError:
            time_value = self.time_samples[-1]

        for i in range(len(self.piecewise_linear_function)):
            if self.piecewise_linear_function[i][0] <= time_value <= self.piecewise_linear_function[i + 1][0] :
                return self.piecewise_linear_function[i][1] + (self.piecewise_linear_function[i + 1][1] - self.piecewise_linear_function[i][1]) * (time_value - self.piecewise_linear_function[i][0]) / (self.piecewise_linear_function[i + 1][0] - self.piecewise_linear_function[i][0])
        # Last value of piecewise linear function
        if time_value >= self.piecewise_linear_function[-1][0]:
            return 0
    

class Input:
    def __init__(self, time_samples, data_samples):
        self.time_samples = time_samples
        self.data_samples = data_samples
        self.start_time = self.time_samples[0]
        self.end_time = self.time_samples[-1]
        self.piecewise_constant_function = self.piecewise_constant_approximation()
        self.piecewise_linear_function = self.piecewise_linear_approximation()

    def piecewise_constant_approximation(self):
        """This function aims to approximate input data into piecewise constant function."""
        return PiecewiseConstantFunction(self.time_samples, self.data_samples)
    
    def piecewise_linear_approximation(self):
        """This function aims to approximate input data into piecewise linear function."""
        return PiecewiseLinearFunction(self.time_samples, self.data_samples)