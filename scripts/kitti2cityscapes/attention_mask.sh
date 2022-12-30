# This script is to visualize attention mask, output directory is under output
# DATASETS.TEST can have multiple test set, is source and target domain test sets, batch size is 1
# SOLVER.MAX_ITER determine how many image to visualize, if the number is greater than dataset size, just visualize entire dataset

python tools/train_net.py --config-file "configs/kitti2cityscapes/sapnetV2_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/kitti2cityscapes-best/output-kitti2city-sapv2-medm-mscam-cycle2-l0.1-e1.0-d0.05-22-10-30_23-16/model_0006399.pth" \
DATASETS.TEST "('kitti-car_train', 'cityscapes-car2_val')" SOLVER.MAX_ITER 999