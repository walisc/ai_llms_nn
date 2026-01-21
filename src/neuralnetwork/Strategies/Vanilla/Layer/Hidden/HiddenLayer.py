from neuralnetwork.Strategies.Vanilla.Utilities.Definitions import VanillaNeuralNetworkProps
from neuralnetwork.Strategies.Vanilla.Layer.Entry.EntryLayer import NN_Entry_Node
from neuralnetwork.Strategies.Vanilla.Layer.LayerBase import NN_Layer
from neuralnetwork.Strategies.Vanilla.Utilities.Utilities import convert_to_grad_weigths


class NN_Hidden_Node(NN_Entry_Node):

    # Updated during back propagation
    current_derivate_delta: float
    accumulated_derivate_deltas: list[float]

    bias: float

    def __init__(self, bais, neural_network_properties):
        super().__init__(neural_network_properties)
        self.bias = bais


    def bb_update_bias(self):
        self.grad_info.biases = self.current_derivate_delta
        self.bias -= (self.current_derivate_delta  * self.neural_network_properties.learning_rate)


    def bb_update_derivate_deltas(self):
   
        # Taking into account multiple run i.e mini-batches
        for i, a in enumerate(self.accumulated_activation_values):
            total_weight_activation_detla = 0.00
            for w in self.outgoing_weights:
                total_weight_activation_detla += w.value * w.node.accumulated_derivate_deltas[i]

            node_derivate_delta = total_weight_activation_detla * self.neural_network_properties.activation_function_strategy.get_derivative_value(a)
            self.accumulated_derivate_deltas.append(node_derivate_delta)

        self.current_derivate_delta = sum(self.accumulated_derivate_deltas)/len(self.accumulated_activation_values)

        
    def do_reset(self):
        super().do_reset()
        self.current_derivate_delta = None
        self.accumulated_derivate_deltas = []

    
    def do_back_propagation(self):
        self.bb_update_derivate_deltas()
        self.bb_update_outgoing_weights()
        self.bb_update_bias()


    def do_forward_propagation(self):
        total_child_value = 0.00
        for i, cn in enumerate(self.child_nodes):
            for w in cn.outgoing_weights:
                if w.node == self:
                    total_child_value += w.value * cn.current_activation_value
                    break
            
        new_node_value = total_child_value + self.bias
        self.set_value(new_node_value)


    def set_value(self, value):
        self.current_value = value
        self.accumulated_values.append(value)

        self.current_activation_value = self.neural_network_properties.activation_function_strategy.get_value(value)
        self.accumulated_activation_values.append(self.current_activation_value)
        return self
    

class Hidden_NN_Layer(NN_Layer):
    nodes: list[NN_Hidden_Node]

    def __init__(self, node_size:int, baises:list[float]=None, neural_network_properties:VanillaNeuralNetworkProps=None):
        super().__init__(neural_network_properties)
        self.set_nodes(node_size, baises, neural_network_properties)
    
    def set_nodes(self, node_size:int, baises:list[float]=None, neural_network_properties:VanillaNeuralNetworkProps=None):
        if baises and len(baises) != node_size:
            raise ValueError("The number of baises must match the number of nodes specified")
        self.nodes = [NN_Hidden_Node(bais, neural_network_properties) for bais in baises] if baises else [NN_Hidden_Node(0.00, neural_network_properties) for _ in range(node_size)]

    def do_forward_propagation(self,):
        for n in self.nodes:
            n.do_forward_propagation()

    def do_back_propagation(self):
        for n in self.nodes:
            n.do_back_propagation()

    def do_reset(self):
        for n in self.nodes:
            n.do_reset()

    def get_grad_info(self):
        return convert_to_grad_weigths(self.nodes)