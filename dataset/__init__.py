from .cityscapes import CitySegmentationTrain, CitySegmentationTest, CitySegmentationTrainWpath
from .ade20k import Ade20kSegmentationTrain, Ade20kSegmentationTest, Ade20kSegmentationTrainWpath
from .idd import IddSegmentationTrain, IddSegmentationTest, IddSegmentationTrainWpath
from .lite import LiteSegmentationTrain, LiteSegmentationTest, LiteSegmentationTrainWpath



datasets = {
	'cityscapes_train': CitySegmentationTrain,
	'cityscapes_test': CitySegmentationTest,
	'cityscapes_train_w_path': CitySegmentationTrainWpath,
    'ade20k_train' : Ade20kSegmentationTrain,
    'ade20k_test' : Ade20kSegmentationTest,
    'ade20k_train_w_path' : Ade20kSegmentationTrainWpath,
    'idd_train' : IddSegmentationTrain,
    'idd_test' : IddSegmentationTest,
    'idd_train_w_path' : IddSegmentationTrainWpath,
    'lite_train' : LiteSegmentationTrain,
    'lite_test' : LiteSegmentationTest,
    'lite_train_w_path' : LiteSegmentationTrainWpath
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
