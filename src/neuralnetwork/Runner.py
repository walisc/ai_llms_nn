from neuralnetwork.Datasets.DatasetRegistry import DataSetRegistry
from neuralnetwork.Strategies import NNBaseStrategy
from neuralnetwork.Strategies.NNStrategiesRegistry import NNStrategiesRegistry
from neuralnetwork.Utilities.Definitions import NeuralNetworkProps, RunProps


def do_run(props: RunProps):

    nnCls = NNStrategiesRegistry.get_strategy(props.strategy)
    nn: NNBaseStrategy = nnCls(
        NeuralNetworkProps(
            learning_rate=0.1,
            activation_function_strategy=props.activation_function,
            cost_derivative_function_strategy=props.cost_derivative
        )
    )

    dataset = DataSetRegistry.get_dataset(props.dataset)()

    nn.add_entry_layer(dataset.get_input_size())

    for hld in dataset.get_preferred_hidden_layer_details():
        nn.add_hidden_layer(hld.neuron_count, hld.biases, hld.weights)

    output_details = dataset.get_output_details()
    nn.add_output_layer(output_details.neuron_count, output_details.biases, output_details.weights)

    data = dataset.get_data()

    train_data_input = data.input[:int(len(data.input) * props.train_percentage)]
    train_data_output = data.output[:int(len(data.input) * props.train_percentage)]


    def do_runthrough(input, expected_output):
        output = nn.do_forward_propagation(input, expected_output)

        print(f"Current Cost:- {output}")
        nn.do_backpropagation() 


    for epoch in range(props.epochs):
        if props.batch_size > 0:

            for i in range(0, len(train_data_input), props.batch_size):
                do_runthrough(train_data_input[i:i + props.batch_size],  train_data_output[i:i + props.batch_size])

        else:
            do_runthrough(train_data_input, train_data_output)

    
    test_data_input = data.input[int(len(data.input) * props.train_percentage):]
    test_data_output = data.output[int(len(data.input) * props.train_percentage):]  
    for i in range(len(test_data_input)):
        output = nn.do_forward_propagation([test_data_input[i]], [test_data_output[i]])
        print(f"Evaluation Cost:- {output}")
            



