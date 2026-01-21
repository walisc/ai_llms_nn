from abc import ABC, abstractmethod

class ActivationFunctionStrategy(ABC):
    @abstractmethod
    def get_value(self, in_value):
        pass

    @abstractmethod
    def get_derivative_value(self,in_value):
        pass



