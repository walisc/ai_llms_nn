from neuralnetwork.Strategies.Numpy.Utilities.CostDerivativeStrategies.Squared.CostDerivativeStrategiesSquared import CostDerivativeStrategiesSquared
from neuralnetwork.Utilities.CostDerivativeStrategies import COST_DERIVATIVE_STRATEGY_NAMES
from neuralnetwork.Utilities.CostDerivativeStrategies.CostDerivateStrategy import CostDerivateStrategy


class CostDerivateStrategyRegistry:
    registry: dict[str, CostDerivateStrategy] = {
        COST_DERIVATIVE_STRATEGY_NAMES.MSE: CostDerivativeStrategiesSquared,
    }

    @classmethod
    def get_strategy(cls, name: str) -> CostDerivateStrategy:
        if name not in cls.registry:
            raise ValueError(f"CostDerivateStrategy '{name}' not found in registry")
        return cls.registry[name]()