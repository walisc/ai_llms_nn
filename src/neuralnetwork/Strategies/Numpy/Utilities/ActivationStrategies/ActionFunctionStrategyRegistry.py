
from neuralnetwork.Strategies.Numpy.Utilities.ActivationStrategies.Sigmoid.SigmoidActivationStrategy import SigmoidActivationStrategy
from neuralnetwork.Utilities.ActivationStrategies import ACTIVATION_FUNCTION_STRATEGY_NAMES
from neuralnetwork.Utilities.ActivationStrategies.ActivationFunctionStrategy import ActivationFunctionStrategy


class ActionFunctionStrategyRegistry:
    registry: dict[str, ActivationFunctionStrategy] = {
        ACTIVATION_FUNCTION_STRATEGY_NAMES.SIGMOID: SigmoidActivationStrategy,
    }


    @classmethod
    def get_strategy(cls, name: str) -> ActivationFunctionStrategy:
        if name not in cls.registry:
            raise ValueError(f"ActivationFunctionStrategy '{name}' not found in registry")
        return cls.registry[name]()
