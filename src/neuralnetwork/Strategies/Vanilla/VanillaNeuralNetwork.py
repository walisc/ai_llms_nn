
from neuralnetwork.Strategies.NNBaseStrategy import NNBaseStrategy
from neuralnetwork.Strategies.Vanilla.Layer.Hidden.HiddenLayer import Hidden_NN_Layer
from neuralnetwork.Strategies.Vanilla.Layer.Output.OutputLayer import Output_NN_Layer
from neuralnetwork.Strategies.Vanilla.Utilities.ActivationStrategies.ActionFunctionStrategyRegistry import ActionFunctionStrategyRegistry
from neuralnetwork.Strategies.Vanilla.Layer.Entry.EntryLayer import Entry_NN_Layer
from neuralnetwork.Strategies.Vanilla.Utilities.CostDerivativeStrategies.CostDerivateStrategyRegistry import CostDerivateStrategyRegistry
from neuralnetwork.Strategies.Vanilla.Utilities.Definitions import VanillaNeuralNetworkProps
from neuralnetwork.Utilities.Definitions import NeuralNetworkProps


class VanillaNeuralNetwork(NNBaseStrategy):

    entry_layer: Entry_NN_Layer
    hidden_layers: list[Hidden_NN_Layer]
    output_layer: Output_NN_Layer

    def __init__(self, properties:NeuralNetworkProps):
        super().__init__(properties=properties)
        self.properties = properties    
        self.load_nueral_network_properties()
        self.hidden_layers = []
        self.output_layer = None
        self.entry_layer = None

    def load_nueral_network_properties(self):
        self.properties = VanillaNeuralNetworkProps(
            learning_rate = self.properties.learning_rate,
            activation_function_strategy = ActionFunctionStrategyRegistry.get_strategy(self.properties.activation_function_strategy),
            cost_derivative_function_strategy = CostDerivateStrategyRegistry.get_strategy(self.properties.cost_derivative_function_strategy)
        )

    def add_entry_layer(self, entry_nodes: int):
        self.entry_layer = Entry_NN_Layer(entry_nodes, self.properties)
        return self.entry_layer

    def _set_parent_layer(self, layer, incoming_weights:list[list[float]]=None):
        if (len(self.hidden_layers) == 0):
            self.entry_layer.set_parent_layer(layer, incoming_weights)
        else:
            self.hidden_layers[-1].set_parent_layer(layer, incoming_weights)

    def _ensure_entry_layer_exists(self):
        if not self.entry_layer:
            raise ValueError("Entry layer needs to be specified first")


    def add_hidden_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        self._ensure_entry_layer_exists()
        
        hl = Hidden_NN_Layer(nodes, baises, self.properties)
        self._set_parent_layer(hl, incoming_weights)

        self.hidden_layers.append(hl)

        return hl

    def add_output_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        self._ensure_entry_layer_exists()

        self.output_layer = Output_NN_Layer(nodes, baises, self.properties)
        self._set_parent_layer(self.output_layer , incoming_weights)

        return self.output_layer

    
    def do_forward_propagation(self, entry_nodes_values: list[list[float]], expect_values:list[list[float]]):

        if (len(entry_nodes_values) != len(expect_values)):
            raise ValueError(f"Expected entry values and expect values to be of same size")
        
        if not self.entry_layer or not self.output_layer:
            raise ValueError("Entry and output layers must be set")
        
    
        for i, en_value in enumerate(entry_nodes_values):
            self.entry_layer.do_forward_propagation(en_value)


            for hl in self.hidden_layers:
                hl.do_forward_propagation()
            
            self.output_layer.do_forward_propagation(expect_values[i])

        return self.output_layer.get_current_cost()

    
    def do_backpropagation(self):

        self.output_layer.do_back_propagation()

        for hl in self.hidden_layers:
            hl.do_back_propagation()

        
        self.entry_layer.do_back_propagation()


        print(self.output_layer.get_grad_info())
        for hl in self.hidden_layers:
            print(hl.get_grad_info())

        print(self.entry_layer.get_grad_info())

        print("\n\nUpdated Parameters:\m")
        print("Out Layer Biases:")
        print(self.output_layer.nodes[0].bias)
        print("Hidden Layer 0 Weights:")
        print([[v.value for v in n.outgoing_weights] for n in self.hidden_layers[0].nodes])
        print("Hidden Layer 0 Biases:")
        print([n.bias for n in self.hidden_layers[0].nodes])
        print("Entry Layer 0 Weights:")
        print([[v.value for v in n.outgoing_weights] for n in self.entry_layer.nodes])

        self.do_reset()


    def do_reset(self):
        self.entry_layer.do_reset()
        for hl in self.hidden_layers:
            hl.do_reset()
        self.output_layer.do_reset()
        