# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where location on image is object detector focus
# --grad-cam-object-detection is to visualize grad cam of object detector
python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_"  SOLVER.MAX_ITER 478 DATASETS.TEST "('cityscapes-car2_test1',)" \
MODEL.WEIGHTS "pretrained/sim10k2cityscapes-best/output-sim10k2city-sapv2-medm-mscam-cycle2-l0.1-e0.8-d0.3-22-10-15_00-30_49.48/model_0005599.pth" \
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75


python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 478 DATASETS.TEST "('cityscapes-car2_test1',)" \
MODEL.WEIGHTS "pretrained/sim10k-sapnet/model_0002199.pth" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75
