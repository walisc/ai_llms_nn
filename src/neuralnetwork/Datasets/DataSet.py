from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class PreferredLayerDetails:
    def __init__(self,  neuron_count: int, biases: list[float]=None, weights: list[list[float]]=None):
        self.neuron_count = neuron_count
        self.biases = biases
        self.weights = weights


class NNData:
    def __init__(self, input, ouput):
        if len(input) != len(ouput):
            raise ValueError("Input and output should be the same length")
        self.input = input
        self.output = ouput

        
class DataSet(ABC):

    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_data(self, id=None, properties=None) -> NNData:
        pass

    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def get_output_details(self) -> PreferredLayerDetails:
        pass

    def get_preferred_hidden_layer_details(self) -> list[PreferredLayerDetails]:
        return []

    def get_data_path(self):
        data_path = Path(__file__).parent / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path
    

class GeneratableDataSet(DataSet):
    def _get_saved_data_id(self, id=None):
        return f"{self.get_id()}_{id}" if id else self.get_id()
    
    @abstractmethod
    def is_saveable(self):
        pass

    @abstractmethod
    def load_saved_data(self, id):
        pass

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def save_data(self, save_id, data):
        pass

    def get_data(self, id=None, properties:Any=None) -> NNData:
        
        save_id = self._get_saved_data_id(id)
        if self.is_saveable():
            savedData = self.load_saved_data(save_id)

            if savedData:
                return self.before_data_send(savedData)
        
            data = self.generate_data(properties or {})
            self.save_data(save_id, data)
        else:
            data = self.generate_data(properties or {})
        
        return self.before_data_send(data)
    
    def before_data_send(self, data):
        return data
    
    def get_saved_data_path(self, id_path):
        generated_dir = self.get_data_path() / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / id_path
