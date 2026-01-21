from neuralnetwork.Utilities.Definitions import NeuralNetworkProps
from neuralnetwork.Strategies.NNBaseStrategy import NNBaseStrategy
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda) 

class NodeWrapper:
    pass

class LayerWrapper:
    def __init__(self, nodes, baises=None, weights=None):
        self.layer = None
        self.baises = baises
        self.weights = weights
        self.nodes = nodes 
        

class PyTorchNeuralNetworkImpl(nn.Module):
    layers: list[LayerWrapper]
    def __init__(self, layers: list[LayerWrapper], properties:NeuralNetworkProps):
        super(PyTorchNeuralNetworkImpl, self).__init__()
        print(f"Using device: {device}")
        self.layers = layers
        self.properties = properties
        self.build_network()

    def build_network(self):
        for i, item in enumerate(self.layers):
            if i < len(self.layers) - 1:
                self.layers[i+1].layer = nn.Linear(len(self.layers[i].nodes), len(self.layers[i+1].nodes))
                setattr(self, f"nn_layer_{i}", self.layers[i+1].layer)

       # TODO: set properly
       #  TODO: Note better values for propeties
        self.activation = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.properties.learning_rate)

        with torch.no_grad():

            for i, item in enumerate(self.layers):

                if item.layer is None:
                    continue

                if item.weights:
                    item.layer.weight.copy_(torch.tensor(item.weights))
                else: 
                    item.layer.weight.copy_(torch.randn_like(item.layer.weight))

               
                if  item.baises is not None:
                    item.layer.bias.copy_(torch.tensor(item.baises))


    def forward(self, x):
        output_tensor = x
        for layer in self.layers:
            if layer.layer is None:
                continue
     
            output_tensor = self.activation(layer.layer(output_tensor))

        return output_tensor



class PyTorchNeuralNetwork(NNBaseStrategy):

    layers: list[LayerWrapper]

    def __init__(self, properties):
        super().__init__(properties=properties)
        self.layers = []
        self.pytorch_nn = None
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
        result = self.pytorch_nn(torch.tensor(entry_nodes_values, dtype=torch.float32).to(device))

        self.current_loss = self.pytorch_nn.loss_fn(result, torch.tensor(expect_values, dtype=torch.float32).to(device))
        return self.current_loss, None
        

    def ensure_network_built(self):
        if not self.pytorch_nn:
            self.pytorch_nn = PyTorchNeuralNetworkImpl(self.layers, self.properties)
            self.pytorch_nn.to(device) 
            

    def do_backpropagation(self):
        self.pytorch_nn.optimizer.zero_grad()
        self.current_loss.backward()
        

        # for i, item in enumerate(self.layers):
        #     if item.layer:
        #         print(f"Layer{i} weights grad:\n", item.layer.weight.grad)
        #         print(f"Layer{i} bias grad:\n", item.layer.bias.grad)

        self.pytorch_nn.optimizer.step()

        # print("Model Parameters:")
        
        # for name, param in self.pytorch_nn.named_parameters():
        #     print(f"Layer: {name}")
        #     print(f"Shape: {param.shape}")
        #     print(f"Values:\n{param.data}\n") 


    
        



        
