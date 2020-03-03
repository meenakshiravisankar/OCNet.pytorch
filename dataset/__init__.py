from .cityscapes import CitySegmentationTrain, CitySegmentationTest, CitySegmentationTrainWpath
from .ade20k import Ade20kSegmentationTrain, Ade20kSegmentationTest, Ade20kSegmentationTrainWpath
datasets = {
	'cityscapes_train': CitySegmentationTrain,
	'cityscapes_test': CitySegmentationTest,
	'cityscapes_train_w_path': CitySegmentationTrainWpath,
    'ade20k_train' : Ade20kSegmentationTrain,
    'ade20k_test' : Ade20kSegmentationTest,
    'ade20k_train_w_path' : Ade20kSegmentationTrainWpath
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
