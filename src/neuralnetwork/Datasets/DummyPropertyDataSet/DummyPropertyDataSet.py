import random
import csv
from typing import Any
from neuralnetwork.Datasets.DataSet import GeneratableDataSet, NNData, PreferredLayerDetails
import statistics


class DummyPropertyDataSet(GeneratableDataSet):

    @classmethod
    def get_id(cls) -> str:
        return "DummyPropertyDataSet"
    
    def is_cacheable(self):
        return True
    
    def is_saveable(self):
        return True
    
    def load_saved_data(self, id):
        try:
            with open(f"{self.get_saved_data_path(id)}.csv", 'r') as f:
                reader = csv.DictReader(f)
                return [ {k: int(v) for k, v in row.items()} for row in reader ]
        except FileNotFoundError:
            return None
        
    
    def save_data(self, id, data):
        keys = data[0].keys()
        with open(f"{self.get_saved_data_path(id)}.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)


    def generate_data(self, properties):
        all_data = []

        random.seed(properties.get("seed_id", None))
        total_samples = properties.get("count", 1000)
        good_percentage = properties.get("percentage_good", 0.4)

        num_good = int(total_samples * good_percentage)
        num_bad = total_samples - num_good

        properties = {
            "price": {
                "good": lambda: random.randint(100000, 699999),
                "bad": lambda: random.randint(700000, 2000000)
            },
            "amenities_score": {
                "good": lambda: random.randint(9, 10),
                "bad": lambda: random.randint(1, 8)
            },
            "aesthetic_score": {
                "good": lambda: random.randint(8, 10),
                "bad": lambda: random.randint(1, 7) 
            }
        }

        for _ in range(num_good):
            price = properties["price"]["good"]()
            amenities = properties["amenities_score"]["good"]()
            aesthetic = properties["aesthetic_score"]["good"]()
            label = 1 
            
            
            all_data.append({
                "price": price,
                "amenities_score": amenities,
                "aesthetic_score": aesthetic,
                "is_good_investment": label
            })

        for _ in range(num_bad):
            bad_property = [p for p in properties.keys()][random.randint(0,2)]
            all_data.append({k: (properties[k]["bad"]() if k == bad_property else properties[k]["good"]()) for k in properties.keys()})
            all_data[-1]["is_good_investment"] = 0

        random.shuffle(all_data)

        return all_data

        
    def prepare_data(self, data:Any) -> NNData: 
        cols_ver = {}

        for p in data:
            for k,v in p.items():
                if k not in cols_ver:
                    cols_ver[k] = []
                cols_ver[k].append(v)

        cols_ver_props = {}

        for k, v in cols_ver.items():
            cols_ver_props[k] = {
                "mean": statistics.mean(v),
                "std": statistics.pstdev(v) 
            }

        def get_standardised_value(row):
            return [(v - cols_ver_props[k]["mean"]) / cols_ver_props[k]["std"] for k,v in row.items() if k != "is_good_investment"]


        return NNData([get_standardised_value(p) for p in data], [[p["is_good_investment"]] for p in data])


    def get_input_size(self) -> int:
        return 3
    
    def get_output_details(self) -> PreferredLayerDetails:
        return PreferredLayerDetails(1, [0.0], [[0.2, 0.4]])

    def get_preferred_hidden_layer_details(self) -> list[PreferredLayerDetails]:
        return [
            PreferredLayerDetails(2, [0.0, 0.0], [
                                                    [0.2, 0.3, 0.5],
                                                    [0.4, 0.6, 0.6]
                                                    ])
        ]
   

   
    
    