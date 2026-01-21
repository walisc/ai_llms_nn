import argparse

from neuralnetwork.Datasets.DatasetRegistry import DataSetRegistry
from neuralnetwork.Strategies.NNStrategiesRegistry import NNStrategiesRegistry
from neuralnetwork.Utilities.ActivationStrategies import ACTIVATION_FUNCTION_STRATEGY_NAMES
from neuralnetwork.Utilities.CostDerivativeStrategies import COST_DERIVATIVE_STRATEGY_NAMES
from neuralnetwork.Utilities.Definitions import RunProps

import neuralnetwork.Runner as Runner



parser = argparse.ArgumentParser()
parser.add_argument("-s", "--strategy", type=str, required=True, choices=[o for o in NNStrategiesRegistry._strategies.keys()] )
parser.add_argument("-d", "--dataset", type=str, required=False, choices=[o for o in DataSetRegistry._datasets.keys()])

parser.add_argument("-af", "--activation_function", type=str, required=False, choices=[ACTIVATION_FUNCTION_STRATEGY_NAMES.SIGMOID, 
                                                                                       ACTIVATION_FUNCTION_STRATEGY_NAMES.RELU, 
                                                                                       ACTIVATION_FUNCTION_STRATEGY_NAMES.TANH], default=ACTIVATION_FUNCTION_STRATEGY_NAMES.SIGMOID )

parser.add_argument("-cd", "--cost_derivative", type=str, required=False, choices=[COST_DERIVATIVE_STRATEGY_NAMES.MSE, COST_DERIVATIVE_STRATEGY_NAMES.CROSS_ENTROPY], default=COST_DERIVATIVE_STRATEGY_NAMES.MSE )

parser.add_argument("-lr", "--learning_rate", type=float, required=False, default=0.1 )
parser.add_argument("-e", "--epochs", type=int, required=False, default=1000 )
parser.add_argument("-bs", "--batch_size", type=int, required=False, default=-1 )
parser.add_argument("-tp", "--train_percentage", type=float, required=False, default=0.7 )
parser.add_argument("-ls", "--log_steps", type=int, required=False, default=10 )    
parser.add_argument("-g", "--use_gpu", action='store_true' )


args = parser.parse_args()

def run():
    Runner.do_run(RunProps(**vars(args)))