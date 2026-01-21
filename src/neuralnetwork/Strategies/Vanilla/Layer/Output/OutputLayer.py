from neuralnetwork.Strategies.Vanilla.Layer.Hidden.HiddenLayer import Hidden_NN_Layer, NN_Hidden_Node
from neuralnetwork.Strategies.Vanilla.Utilities.Utilities import convert_to_grad_weigths


class NN_Output_Node(NN_Hidden_Node):
    def __init__(self, bais, neural_network_properties):
        super().__init__(bais, neural_network_properties)

    def do_forward_propagation(self, expected_value):
        super().do_forward_propagation()
        self.accumulated_cost_derivates.append(self.neural_network_properties.cost_derivative_function_strategy.get_derivative_value(self.current_activation_value, expected_value))
        self.accumulated_cost.append(self.neural_network_properties.cost_derivative_function_strategy.get_error_cost_value(self.current_activation_value, expected_value))

    def bb_update_derivate_deltas(self):
        for i, a in enumerate(self.accumulated_activation_values):
            current_delta = self.neural_network_properties.activation_function_strategy.get_derivative_value(a) * self.accumulated_cost_derivates[i]
            self.accumulated_derivate_deltas.append(current_delta)

        self.current_derivate_delta = sum(self.accumulated_derivate_deltas)/len(self.accumulated_derivate_deltas)

    def do_back_propagation(self):
        self.bb_update_derivate_deltas()
        self.bb_update_bias()

    def do_reset(self):
        super().do_reset()
        self.accumulated_cost_derivates = []
        self.accumulated_cost = []

    
    
class Output_NN_Layer(Hidden_NN_Layer):
    nodes: list[NN_Output_Node]

    def __init__(self, node_size:int, baises:list[float]=None, neural_network_properties=None):
        super().__init__(node_size, baises, neural_network_properties)
        self.nodes = [NN_Output_Node(bais, neural_network_properties) for bais in baises] if baises else [NN_Output_Node(0.0, neural_network_properties) for _ in range(node_size)]

    def do_forward_propagation(self, expect_values:list[float]):
        for i, n in enumerate(self.nodes):
            n.do_forward_propagation(expect_values[i])

    def do_back_propagation(self):
        for n in self.nodes:
            n.do_back_propagation()

    def do_reset(self):
        for n in self.nodes:
            n.do_reset()    


    def get_current_cost(self):
        node_error_costs = []
        total_error_cost = 0.0

        for r in self.nodes:
            node_error_costs_value = sum(r.accumulated_cost)/len(r.accumulated_cost)
            node_error_costs.append(node_error_costs_value)
            total_error_cost += node_error_costs_value

        total_error_cost /= len(node_error_costs)

        return total_error_cost, node_error_costs
    
    def get_grad_info(self):
        return convert_to_grad_weigths(self.nodes)