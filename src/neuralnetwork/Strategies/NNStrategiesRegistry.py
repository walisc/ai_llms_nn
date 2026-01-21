from neuralnetwork.Strategies.Numpy.NumpyNeuralNetwork import NumpyNeuralNetwork
from neuralnetwork.Strategies.PyTorch.PyTorchNeuralNetwork import PyTorchNeuralNetwork
from neuralnetwork.Strategies.TensorFlow.TensorFlowNeuralNetwork import TensorFlowNeuralNetwork
from neuralnetwork.Strategies.Vanilla.VanillaNeuralNetwork import VanillaNeuralNetwork


class NN_STRATEGY_NAMES:
    VANILLA = "Vanilla"
    PY_TORCH = "PyTorch"
    NUMPY = "Numpy"
    TENSOR_FLOW = "TensorFlow"
    

class NNStrategiesRegistry:
    _strategies = {
        NN_STRATEGY_NAMES.VANILLA: VanillaNeuralNetwork,
        NN_STRATEGY_NAMES.PY_TORCH: PyTorchNeuralNetwork,
        NN_STRATEGY_NAMES.NUMPY : NumpyNeuralNetwork,
        NN_STRATEGY_NAMES.TENSOR_FLOW: TensorFlowNeuralNetwork
    }

    @classmethod
    def get_strategy(cls, strategy_type) :
        strategy = cls._strategies.get(strategy_type)
        if not strategy:
            raise ValueError(f"Strategy {strategy_type} not found in registry.")
        return strategy
    



