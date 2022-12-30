# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where the domain classifier focuses on the image based on the features we choose
# --grad-cam-target-domain and --grad-cam-target-domain is to visualize source or target domain evidence
# --backbone-feature and --attention-mask is to select which feature to visualize
# DATASETS.TEST can have multiple test set, is source and target domain test sets, batch size is 1
# because of pytorch issue, please use build_resnet_backbone_ to create resnet instead of default resnet function

# sapnet v2
# target domain evidence, backbone feature 
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --grad-cam-target-doamin --backbone-feature \
MODEL.WEIGHTS "pretrained/cityscapes2foggy-best/output-city2foggy-sapv0-medm-mscam-cycle2-l0.6-e1.0-d0.2-22-10-11_20-10-49.63/model_0011199.pth" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 999


# source domain evidence, backbone feature 
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --grad-cam-source-doamin --backbone-feature \
MODEL.WEIGHTS "pretrained/cityscapes2foggy-best/output-city2foggy-sapv0-medm-mscam-cycle2-l0.6-e1.0-d0.2-22-10-11_20-10-49.63/model_0011199.pth" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 999


# sapnet
# target domain evidence, backbone feature 
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --grad-cam-target-doamin --backbone-feature \
MODEL.WEIGHTS "pretrained/city2foggy-sapnet/model_0006799.pth" MODEL.DA_HEAD.NAME "SAPNet" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 999


# source domain evidence, backbone feature 
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --grad-cam-source-doamin --backbone-feature \
MODEL.WEIGHTS "pretrained/city2foggy-sapnet/model_0006799.pth" MODEL.DA_HEAD.NAME "SAPNet" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" SOLVER.MAX_ITER 999
