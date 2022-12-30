# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where location on image is object detector focus
# --grad-cam-object-detection is to visualize grad cam of object detector

python tools/train_net.py --config-file "configs/kitti2cityscapes/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 SOLVER.MAX_ITER 478 DATASETS.TEST "('cityscapes-car2_test2',)" \
MODEL.WEIGHTS "pretrained/kitti2cityscapes-best/output-kitti2city-sapv2-medm-mscam-cycle2-l0.1-e1.0-d0.05-22-10-30_23-16/model_0006399.pth"
