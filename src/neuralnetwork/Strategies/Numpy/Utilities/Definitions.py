from dataclasses import dataclass

from neuralnetwork.Utilities.ActivationStrategies.ActivationFunctionStrategy import ActivationFunctionStrategy
from neuralnetwork.Utilities.CostDerivativeStrategies.CostDerivateStrategy import CostDerivateStrategy




@dataclass(slots=True)
class NumpyNeuralNetworkProps:
    learning_rate: float
    activation_function_strategy: ActivationFunctionStrategy
    cost_derivative_function_strategy: CostDerivateStrategy  