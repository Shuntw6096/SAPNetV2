from detectron2.data.datasets import register_pascal_voc, register_coco_instances
from pathlib import Path

dataset_base_dir = Path(__file__).parent.parent.parent / 'datasets'

json_file_path = dataset_base_dir/'Cityscapes-coco'/'instancesonly_filtered_gtFine_train.json'
image_root =  dataset_base_dir/'Cityscapes-coco'/'train'
split = 'train'
meta_name = 'cityscapes_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

dataset_dir = str(dataset_base_dir/ 'Cityscapes-coco'/'VOC2007-val-original')
split = 'val' # "train", "test", "val", "trainval"
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
years = 2007
meta_name = 'cityscapes_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)


json_file_path = dataset_base_dir/'Foggy-cityscapes-coco'/'instancesonly_filtered_gtFine_train.json'
image_root =  dataset_base_dir/'Foggy-cityscapes-coco'/'train'
split = 'train'
meta_name = 'foggy-cityscapes_{}'.format(split)
register_coco_instances(meta_name, {}, json_file_path, image_root)

dataset_dir = str(dataset_base_dir/ 'Foggy-cityscapes-coco'/'VOC2007')
split = 'val' # "train", "test", "val", "trainval"
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
years = 2007
meta_name = 'foggy-cityscapes_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'Foggy-cityscapes-coco'/'VOC2007')
split = 'test' # "train", "test", "val", "trainval"
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
years = 2007
meta_name = 'foggy-cityscapes_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'sim10k')
split = 'train' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'sim10k_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'sim10k')
split = 'val' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'sim10k_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'Cityscapes-coco'/'VOC2007-car-train')
split = 'train' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'cityscapes-car_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'Cityscapes-coco'/'VOC2007-car2')
split = 'val' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'cityscapes-car2_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'Cityscapes-coco'/'VOC2007-car2')
split = 'test' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'cityscapes-car2_test1'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'kitti-VOCdevkit2007')
split = 'train' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'kitti-car_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'Cityscapes-coco'/'VOC2007-car2')
split = 'trainval' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'cityscapes-car2_test2'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'cityscapes2foggy_cycleGAN')
split = 'train' # "train", "test", "val", "trainval"
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
years = 2007
meta_name = 'cityscapes2fog_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'sim10k2city_cycleGAN')
split = 'train' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'sim10k2city_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)

dataset_dir = str(dataset_base_dir/ 'kitti2city_cycleGAN')
split = 'train' # "train", "test", "val", "trainval"
classes = ('car',)
years = 2007
meta_name = 'kitti2city_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)