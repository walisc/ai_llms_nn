from abc import ABC, abstractmethod
from neuralnetwork.Utilities.Definitions import NeuralNetworkProps


class NNBaseStrategy(ABC):

    properties: NeuralNetworkProps


    def __init__(self, properties:NeuralNetworkProps):
        self.properties = properties


    @abstractmethod
    def add_entry_layer(self, entry_nodes: int):
        pass

    @abstractmethod
    def add_hidden_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        pass

    @abstractmethod
    def add_output_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        pass

    @abstractmethod
    def do_forward_propagation(self, entry_nodes_values: list[float], expect_values:list[float]):
        pass

    @abstractmethod
    def do_backpropagation(self):
        pass