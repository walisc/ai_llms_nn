
import numpy as np

from neuralnetwork.Utilities.ActivationStrategies.ActivationFunctionStrategy import ActivationFunctionStrategy



class SigmoidActivationStrategy(ActivationFunctionStrategy):

    def get_value(self, in_value):
        return 1.0/(1.0 + np.exp (-in_value))


    def get_derivative_value(self, in_value_processed):
        return in_value_processed *(1 - in_value_processed)
