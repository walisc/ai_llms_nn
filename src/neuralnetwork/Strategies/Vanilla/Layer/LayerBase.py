from typing import Any
from neuralnetwork.Strategies.Vanilla.Utilities.Definitions import VanillaNeuralNetworkProps


class NN_Node:

    neural_network_properties: VanillaNeuralNetworkProps

    child_nodes:list[Any] 

    current_value: float
    accumulated_values: list[float]

    current_activation_value: float
    accumulated_activation_values: list[float]

    def __init__(self, neural_network_properties:VanillaNeuralNetworkProps):
        self.neural_network_properties = neural_network_properties
        self.child_nodes = []
        self.do_reset()

    def set_value(self, value):
        self.current_value = value
        self.accumulated_values.append(value)

    def do_reset(self):
        self.current_value = None
        self.accumulated_values = []

        self.current_activation_value = None
        self.accumulated_activation_values = []


    def set_parent_nodes(self, nodes:list[Any]):
        for n in nodes:
            n.set_child(self)

    def set_child(self, n:Any):
        self.child_nodes.append(n)

    
class NNNodeWeight:
    value: float
    node: NN_Node

    def __init__(self, value:float, node:NN_Node):
        self.value = value
        self.node = node

        
class NN_Layer:
    nodes: list[NN_Node]

    def __init__(self, neural_network_properties:VanillaNeuralNetworkProps):
        self.neural_network_properties = neural_network_properties
        self.nodes = []
        

    def set_parent_layer(self, parent_layer:Any, incoming_weights:list[list[float]]=None):
        if incoming_weights is None:
            incoming_weights = [[0.00 for _ in self.nodes] for _ in parent_layer.nodes]

        if len(parent_layer.nodes) != len(incoming_weights):
            raise ValueError("The output nodes do not match the number of weights specified")
        
        if (len(incoming_weights[0]) != len(self.nodes)):
            raise ValueError("The weights does not match the number of layer nodes")
        
        for i, n in enumerate(self.nodes):
            node_weights = []
            for w in incoming_weights:
                node_weights.append(w[i])
                
            n.set_parent_nodes(parent_layer.nodes, node_weights)




