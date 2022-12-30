# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where location on image is object detector focus
# --grad-cam-object-detection is to visualize grad cam of object detector


python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.85 SOLVER.MAX_ITER 478 \
MODEL.WEIGHTS "pretrained/cityscapes2foggy-best/output-city2foggy-sapv0-medm-mscam-cycle2-l0.6-e1.0-d0.2-22-10-11_20-10-49.63/model_0011199.pth"


python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.85 SOLVER.MAX_ITER 478 \
MODEL.WEIGHTS "pretrained/city2foggy-sapnet/model_0006799.pth"
