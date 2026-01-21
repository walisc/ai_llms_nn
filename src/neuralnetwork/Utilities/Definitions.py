from dataclasses import dataclass


    



@dataclass(slots=True)
class NeuralNetworkProps:
    learning_rate: float
    activation_function_strategy: str
    cost_derivative_function_strategy: str

@dataclass(slots=True)
class GradInfo:
    weights: list[list[float]]
    biases: list[float]

@dataclass
class RunProps:
    strategy: str
    dataset: str
    activation_function: str
    cost_derivative: str
    learning_rate: float
    epochs: int
    batch_size: int
    train_percentage: float
    log_steps: int
    use_gpu: bool