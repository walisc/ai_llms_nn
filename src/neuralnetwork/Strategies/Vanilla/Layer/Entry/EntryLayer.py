from neuralnetwork.Strategies.Vanilla.Layer.LayerBase import NN_Layer, NN_Node, NNNodeWeight
from neuralnetwork.Strategies.Vanilla.Utilities.Definitions import VanillaNeuralNetworkProps
from neuralnetwork.Strategies.Vanilla.Utilities.Utilities import convert_to_grad_weigths
from neuralnetwork.Utilities.Definitions import GradInfo


class NN_Entry_Node(NN_Node):

    outgoing_weights: list[NNNodeWeight]
    grad_info: GradInfo

    def __init__(self, neural_network_properties:VanillaNeuralNetworkProps):
        super().__init__(neural_network_properties)
        self.outgoing_weights = []

    def set_parent_nodes(self, nodes:list[NN_Node], default_weights:list[float]=None):
        super().set_parent_nodes(nodes)
        if default_weights is not None and len(nodes) != len(default_weights):
            raise ValueError("default weights need to match the outgoing wieghts")
        
        weights = default_weights if default_weights else [0.00 for _ in nodes]

        for i, n in enumerate(nodes):
            self.outgoing_weights.append(NNNodeWeight(weights[i], n))

    def set_value(self, value):
        super().set_value(value)

        self.current_activation_value = value
        self.accumulated_activation_values.append(value)
        return self

    def do_forward_propagation(self, entry_values:float):
        self.set_value(entry_values)


    def bb_update_outgoing_weights(self):
        for i, w in enumerate(self.outgoing_weights):
            # Taking into account multiple run i.e mini-batches
            if len(self.accumulated_activation_values) != len(w.node.accumulated_derivate_deltas):
                raise ValueError("Inconsistent run values")
            
            total_weight_deltas = 0.00
            for wi, a in enumerate(self.accumulated_activation_values):
                total_weight_deltas += a * w.node.accumulated_derivate_deltas[wi]

            weigth_delta = total_weight_deltas/len(self.accumulated_activation_values)

            self.grad_info.weights.append(weigth_delta)
            self.outgoing_weights[i].value -= (weigth_delta * self.neural_network_properties.learning_rate)


    def do_back_propagation(self):
        self.bb_update_outgoing_weights()

    def do_reset(self):
        super().do_reset()
        self.grad_info = GradInfo(
            weights=[],
            biases=None
        )

        

class Entry_NN_Layer(NN_Layer):
    nodes: list[NN_Entry_Node]

    def __init__(self, node_size:int, neural_network_properties:VanillaNeuralNetworkProps):
        super().__init__(neural_network_properties)
        self.nodes = [NN_Entry_Node(neural_network_properties) for _ in range(node_size)]

    def do_forward_propagation(self, entry_values:list[float]):
        if len(entry_values) != len(self.nodes):
            raise ValueError("The entry values do note match the required input nodes")
        
        for i, e in enumerate(entry_values):
            self.nodes[i].do_forward_propagation(e)

    def do_back_propagation(self):
        for n in self.nodes:
            n.do_back_propagation()

    def do_reset(self):
        for n in self.nodes:
            n.do_reset()

    def get_grad_info(self):
        return convert_to_grad_weigths(self.nodes)

