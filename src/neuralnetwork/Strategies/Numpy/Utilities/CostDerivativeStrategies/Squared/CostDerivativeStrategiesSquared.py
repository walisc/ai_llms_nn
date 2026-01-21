
from neuralnetwork.Utilities.CostDerivativeStrategies.CostDerivateStrategy import CostDerivateStrategy
import numpy as np


class CostDerivativeStrategiesSquared(CostDerivateStrategy):

    def get_cost_difference(self, value, expected):
        return value - expected

    def get_derivative_value_from_cost_differece(self, value):
        return 2 * value
    
    def get_derivative_value(self, value, expected):
        return 2 * self.get_cost_difference(value, expected)
    
    def get_error_cost_value(self, value, expected):
        return np.square(self.get_cost_difference(value, expected))
    
    def get_error_cost_value_from_cost_differece(self, value):
        return np.square(value)


