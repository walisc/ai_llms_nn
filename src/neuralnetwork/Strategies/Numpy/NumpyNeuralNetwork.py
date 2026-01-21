from dataclasses import dataclass
from neuralnetwork.Strategies.Numpy.Utilities.ActivationStrategies.ActionFunctionStrategyRegistry import ActionFunctionStrategyRegistry
from neuralnetwork.Strategies.Numpy.Utilities.CostDerivativeStrategies.CostDerivateStrategyRegistry import CostDerivateStrategyRegistry
from neuralnetwork.Strategies.Numpy.Utilities.Definitions import NumpyNeuralNetworkProps
from neuralnetwork.Utilities.Definitions import NeuralNetworkProps
from neuralnetwork.Strategies.NNBaseStrategy import NNBaseStrategy
import numpy as np

rng = np.random.default_rng()

class NodeWrapper:
    pass

class LayerWrapper:
    def __init__(self, nodes, baises=None, weights=None):
        self.layer = None
        self.baises = baises
        self.weights = weights
        self.nodes = nodes 
        
@dataclass
class NumpyLayer:
    weights: list[list[float]]
    baises: list[float]
    activations: list[float]


class NumpyNeuralNetworkImpl:

    numpy_layers: list[NumpyLayer]

    def __init__(self, layers: list[LayerWrapper], properties:NumpyNeuralNetworkProps):
        super(NumpyNeuralNetworkImpl, self).__init__()
        self.numpy_layers = []
        self.properties = properties
        self.do_reset()
        self.build_network(layers)

    def build_network(self, layers):
        for i, item in enumerate(layers):
            if i == 0:
                self.numpy_layers.append(NumpyLayer(
                        weights=None,
                        baises=None,
                        activations=[]
                    ))
                continue

      
            if not item.weights:
                weights = rng.random(size=(layers[i-1].nodes, item.nodes))
            else:

                if len(item.nodes) != len(item.weights):
                    raise "The weigths matrix does not the node (for the incoming weights)"
                
                if len(layers[i-1].nodes) != len(item.weights[0]):
                    raise "The weigths matrix does not the node (for the outgoing weights)"
                
                weights = item.weights

            if not item.baises:
                baises = rng.random(size=item.nodes)
            else:
                baises = item.baises
            
            self.numpy_layers.append(NumpyLayer(
                    weights=np.array(weights),
                    baises=np.array(baises),
                    activations=[]
                ))
    

    def do_entry_forward(self, x):
        in_value = np.array(x)

     
        for i, layer in enumerate(self.numpy_layers):
            if i == 0:
                layer.activations.append(in_value)
                continue

            in_value = (layer.weights @ in_value) + layer.baises
            in_value = self.properties.activation_function_strategy.get_value(in_value)
            layer.activations.append(in_value)
        return in_value
    

    def do_forward_propagation(self, entry_nodes_values: list[list[float]], expect_values:list[list[float]]):

        self.forward_pp_activations = []
        for e in entry_nodes_values:
            self.forward_pp_activations.append(self.do_entry_forward(e))

        cost_difference = self.properties.cost_derivative_function_strategy.get_cost_difference(self.forward_pp_activations, np.array(expect_values))

        self.cost_derivates = self.properties.cost_derivative_function_strategy.get_derivative_value_from_cost_differece(cost_difference)

        cost_array = np.mean(self.properties.cost_derivative_function_strategy.get_error_cost_value_from_cost_differece(cost_difference), axis=0)

        return np.mean(cost_array), cost_array


    
    def do_back_propagation(self):
  
       
        layer_derivate_delta_array = self.properties.activation_function_strategy.get_derivative_value(np.array(self.forward_pp_activations)) * self.cost_derivates
    
        layers_reversed = [l for l in reversed(self.numpy_layers)]

        for i_layer, l in enumerate(layers_reversed):
           

            if i_layer < len(layers_reversed) - 1:

                 # Update current baise
                layer_derivate_delta = np.mean(layer_derivate_delta_array, axis=0)
                l.baises = l.baises - (layer_derivate_delta * self.properties.learning_rate)
    
                # l-1 acttivation baises
                next_deruvates = []
                for i, a in enumerate(layers_reversed[i_layer+1].activations):
                    next_deruvates.append(np.multiply((l.weights.T @ layer_derivate_delta_array[i]), self.properties.activation_function_strategy.get_derivative_value(a)))


                # Current Weight
                all_weight_deltas = []

                for i, a in enumerate(layers_reversed[i_layer+1].activations):

                    in_nodes = np.array(a).reshape(1, len(a))
                    out_nodes = np.array(layer_derivate_delta_array[i]).reshape(len(layer_derivate_delta_array[i]), 1)
                    
     
                    all_weight_deltas.append( out_nodes @ in_nodes)

        
                l.weights = l.weights - (np.mean(np.stack(all_weight_deltas), axis=0) * self.properties.learning_rate)

                layer_derivate_delta_array = next_deruvates

        

        
    def do_reset(self):
        self.cost_derivates = None
        self.forward_pp_activations = []
        for l in self.numpy_layers:
            l.activations = []
       


class NumpyNeuralNetwork(NNBaseStrategy):

    layers: list[LayerWrapper]
    numpy_nn: NumpyNeuralNetworkImpl


    def __init__(self, properties):
        super().__init__(properties=properties)
        self.layers = []
        self.numpy_nn = None
        self.current_loss = None
        

    def add_output_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        self._ensure_can_add("output")
        
        self.layers.append(LayerWrapper(nodes=[NodeWrapper() for _ in range(nodes)], baises=baises, weights=incoming_weights))
        return self.layers[-1]
    


    def add_hidden_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        self._ensure_can_add("hidden")
        
        self.layers.append(LayerWrapper(nodes=[NodeWrapper() for _ in range(nodes)], baises=baises, weights=incoming_weights))
        return self.layers[-1]

    def _ensure_can_add(self, layer):
        if (len(self.layers) == 0):
            raise ValueError(f"Entry layer must be added before {layer} layer")


    def add_entry_layer(self, entry_nodes: int):
        self.layers.append(LayerWrapper(nodes=[NodeWrapper() for _ in range(entry_nodes)]))
        return self.layers[-1]
    

    def do_forward_propagation(self, entry_nodes_values: list[list[float]], expect_values:list[list[float]]):

        if (len(entry_nodes_values) != len(expect_values)):
            raise ValueError(f"Expected entry values and expect values to be of same size")
        
        self.ensure_network_built()
        
        return self.numpy_nn.do_forward_propagation(entry_nodes_values, expect_values)

        

    def ensure_network_built(self):
        if not self.numpy_nn:
            self.numpy_nn = NumpyNeuralNetworkImpl(self.layers, NumpyNeuralNetworkProps(
                learning_rate=self.properties.learning_rate,
                cost_derivative_function_strategy=CostDerivateStrategyRegistry.get_strategy(self.properties.cost_derivative_function_strategy),
                activation_function_strategy=ActionFunctionStrategyRegistry.get_strategy(self.properties.activation_function_strategy)
            ))


    def do_backpropagation(self):
        self.ensure_network_built()
        self.numpy_nn.do_back_propagation()
        self.numpy_nn.do_reset()


    
        



        
