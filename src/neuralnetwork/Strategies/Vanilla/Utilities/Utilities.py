from typing import Any
from neuralnetwork.Utilities.Definitions import GradInfo


def convert_to_grad_weigths(nodes:list[Any]) -> GradInfo:
    all_weights = []
    baises = []

    for ni, n in enumerate(nodes):
        for wi, w in enumerate(n.grad_info.weights):
            if len(all_weights) <= wi:
                all_weights.append([])

            all_weights[wi].append(w)
    
        baises.append(n.grad_info.biases)

    return GradInfo(
        weights=all_weights,
        biases=baises
    )