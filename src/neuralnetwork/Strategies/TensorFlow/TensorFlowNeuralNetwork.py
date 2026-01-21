from typing import Any
from networkx import nodes
from neuralnetwork.Utilities.Definitions import NeuralNetworkProps
from neuralnetwork.Strategies.NNBaseStrategy import NNBaseStrategy
import tensorflow as tf
import os
import numpy as np

'''
TODO: Comment
model.fit(data['training data'], data['labels'], verbose=False, epochs=500)
'''

# def set_cpu_option():
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""


# def set_gpu_option(which_gpu, fraction_memory):
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
#     config.gpu_options.visible_device_list = which_gpu
#     set_session(tf.Session(config=config))
#     return

class TensorNeuralNetworkImpl:
    layers: list[Any]
    
    def __init__(self, layers: list[Any], properties:NeuralNetworkProps):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.layers = layers
        self.properties = properties
        self.build_network()

    def build_network(self):
        self.model = tf.keras.Sequential(self.layers)

        self.loss_fn = tf.keras.losses.MeanSquaredError() #TODO
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.properties.learning_rate) #TODO

        print(self.layers)
        weights_list = self.model.get_weights()
        for i, weights in enumerate(weights_list):
            print(f"Layer {i} weights shape: {weights}")

    
    def do_forward_propagation(self, entry_nodes_values: list[list[float]], expect_values:list[list[float]]):
        with tf.GradientTape() as tape:
            predictions = self.model(tf.constant(entry_nodes_values), training=True)            
            self.current_loss_value = self.loss_fn(tf.constant(expect_values), predictions)

        self.current_gradient_tape = tape

        return self.current_loss_value, None
    
    def do_backpropagation(self):
        gradients = self.current_gradient_tape.gradient(self.current_loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.do_reset()


    def do_reset(self):
        self.current_loss_value = None
        self.current_gradient_tape = None

class TensorFlowNeuralNetwork(NNBaseStrategy):

    layers: list[Any]

    def __init__(self, properties):
        super().__init__(properties=properties)
        self.layers = []
        self.tensorflow_nn = None
        

    def add_output_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        return self._add_none_entry_layers('output', nodes, baises, incoming_weights)
    

    def add_hidden_layer(self, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        return self._add_none_entry_layers('hidden', nodes, baises, incoming_weights)
        

    def add_entry_layer(self, entry_nodes: int):
        # TODO: use can also do layers.Dense(nodes, activation='sigmoid', input_shape=(entry_nodes,)), in the next layer, and skip this
        self.layers.append(tf.keras.layers.InputLayer(input_shape=(entry_nodes,)))
        return self.layers[-1]
    
    def _ensure_can_add(self, layer):
        if (len(self.layers) == 0):
            raise ValueError(f"Entry layer must be added before {layer} layer")
        
    def _add_none_entry_layers(self, layer_name:str, nodes: int, baises:list[float]=None, incoming_weights:list[list[float]]=None):
        self._ensure_can_add(layer_name)

        #TODO

        kernel_initializer = 'random_normal'
        bias_initializer = 'zeros'

        if incoming_weights:
            kernel_initializer = tf.keras.initializers.Constant(np.array(incoming_weights).T)

        if baises:
            bias_initializer = tf.keras.initializers.Constant(baises)

        print("kernel_initializer:", kernel_initializer)
        print("bias_initializer:", bias_initializer)
        layer = tf.keras.layers.Dense(units=nodes,  
                                      activation='sigmoid', # TODO: update
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)

        self.layers.append(layer)
        return self.layers[-1]
    

    def ensure_network_built(self):
        if not self.tensorflow_nn:
            self.tensorflow_nn =  TensorNeuralNetworkImpl(self.layers, self.properties)


    def do_forward_propagation(self, entry_nodes_values: list[list[float]], expect_values:list[list[float]]):
        if (len(entry_nodes_values) != len(expect_values)):
            raise ValueError(f"Expected entry values and expect values to be of same size")
        
        self.ensure_network_built()
        loss_value = self.tensorflow_nn.do_forward_propagation(entry_nodes_values, expect_values)
        return loss_value


    def do_backpropagation(self):
        self.ensure_network_built()
        self.tensorflow_nn.do_backpropagation()
        


    
        



        
