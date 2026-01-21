from abc import ABC, abstractmethod
from os import path
from pathlib import Path
from typing import Any
import json

class PreferredLayerDetails:
    def __init__(self,  neuron_count: int, biases: list[float]=None, weights: list[list[float]]=None):
        self.neuron_count = neuron_count
        self.biases = biases
        self.weights = weights


class NNData:
    def __init__(self, input, output):
        if len(input) != len(output):
            raise ValueError("Input and output should be the same length")
        self.input = input
        self.output = output

    def get_save_dict(self):
        return {
            "input": self.input,
            "output": self.output
        }

        
class DataSet(ABC):

    @abstractmethod
    def get_id(self) -> str:
        pass


    @abstractmethod
    def do_get_data(self, id=None, properties=None) -> Any:
        pass

    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def get_output_details(self) -> PreferredLayerDetails:
        pass

    def get_preferred_hidden_layer_details(self) -> list[PreferredLayerDetails]:
        return []


    def get_data_id(self, id=None):
        return f"{self.get_id()}_{id}" if id else self.get_id()
    
    @abstractmethod
    def prepare_data(self, data:Any) -> NNData: 
        pass
    
    def is_cacheable(self):
        False

    def cache_data(self, data_id, data:NNData):
        with open(f"{self.get_cached_data_path(data_id)}.json", 'w') as f:
            json.dump(data.get_save_dict(), f)
        return data


    def get_cached_data(self, data_id):
        cached_path = f"{self.get_cached_data_path(data_id)}.json"
        if not path.exists(cached_path):
            return None
        
        with open(cached_path, "r") as f:
            return NNData(**json.load(f))


    def get_data(self, id=None, properties=None) -> NNData:
        data_id = self.get_data_id(id)
        if self.is_cacheable():
            data = self.get_cached_data(data_id)
            if data:
                return data
            
            return self.cache_data(data_id, self.prepare_data(self.do_get_data(id, properties)))
        
        return self.prepare_data(self.do_get_data(id, properties))

    def get_data_path(self):
        data_path = Path(__file__).parent / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path
    
    def get_cached_data_path(self, id_path):
        generated_dir = self.get_data_path() / "cache"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / id_path
    

class GeneratableDataSet(DataSet):
    
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

    def do_get_data(self, id=None, properties:Any=None) -> NNData:
        
        save_id = self.get_data_id(id)
        if self.is_saveable():
            data = self.load_saved_data(save_id)

            if not data:
                data = self.generate_data(properties or {})
                self.save_data(save_id, data)
        else:
            data = self.generate_data(properties or {})
        
        return data
    
    def before_data_send(self, data):
        return data
    
    def get_saved_data_path(self, id_path):
        generated_dir = self.get_data_path() / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir / id_path
