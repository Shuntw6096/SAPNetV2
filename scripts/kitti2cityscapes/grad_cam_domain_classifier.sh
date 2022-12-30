# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where the domain classifier focuses on the image based on the features we choose
# --grad-cam-target-domain and --grad-cam-target-domain is to visualize source or target domain evidence
# --backbone-feature and --attention-mask is to select which feature to visualize
# DATASETS.TEST can have multiple test set, is source and target domain test sets, batch size is 1
# because of pytorch issue, please use build_resnet_backbone_ to create resnet instead of default resnet function


# sapnet v2
# target domain evidence, backbone feature 
python tools/train_net.py --config-file "configs/kitti2cityscapes/sapnetV2_R_50_C4.yaml" --grad-cam-target-doamin --backbone-feature \
MODEL.WEIGHTS "pretrained/kitti2cityscapes-best/output-kitti2city-sapv2-medm-mscam-cycle2-l0.1-e1.0-d0.05-22-10-30_23-16/model_0006399.pth" \
DATASETS.TEST "('kitti-car_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 478


# source domain evidence, backbone feature 
python tools/train_net.py --config-file "configs/kitti2cityscapes/sapnetV2_R_50_C4.yaml" --grad-cam-source-doamin --backbone-feature \
MODEL.WEIGHTS "pretrained/kitti2cityscapes-best/output-kitti2city-sapv2-medm-mscam-cycle2-l0.1-e1.0-d0.05-22-10-30_23-16/model_0006399.pth" \
DATASETS.TEST "('kitti-car_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 478
