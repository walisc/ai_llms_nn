from neuralnetwork.Datasets.DummyDataSet.DummyDataSet import DummyDataSet
from neuralnetwork.Datasets.Property.PropertyData import PropertyDataSet


class DataSetRegistry:
    _datasets = {
        PropertyDataSet.get_id(): PropertyDataSet,
        DummyDataSet.get_id(): DummyDataSet
    }

    @classmethod
    def get_dataset(cls, dataset_name) :
        dataset = cls._datasets.get(dataset_name)
        if not dataset:
            raise ValueError(f"The Dataset {dataset_name} not found in registry.")
        return dataset
    