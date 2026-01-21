
from neuralnetwork.Utilities.CostDerivativeStrategies.CostDerivateStrategy import CostDerivateStrategy


class CostDerivativeStrategiesSquared(CostDerivateStrategy):

    def get_derivative_value(self, value, expected):
        return 2 * (value - expected)
    
    def get_error_cost_value(self, value, expected):
        return (value - expected) ** 2


