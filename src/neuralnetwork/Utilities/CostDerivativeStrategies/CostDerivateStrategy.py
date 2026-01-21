from abc import ABC, abstractmethod


class CostDerivateStrategy(ABC):
    @abstractmethod
    def get_derivative_value(self, value, expected):
        pass

    @abstractmethod
    def get_error_cost_value(self, value, expected):
        pass



