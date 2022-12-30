# This script is to visualize attention mask, output directory is under output
# DATASETS.TEST can have multiple test set, is source and target domain test sets, batch size is 1
# SOLVER.MAX_ITER determine how many image to visualize, if the number is greater than dataset size, just visualize entire dataset

python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/cityscapes2foggy-best/output-city2foggy-sapv0-medm-mscam-cycle2-l0.6-e1.0-d0.2-22-10-11_20-10-49.63/model_0011199.pth" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val',)" SOLVER.MAX_ITER 999


python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/city2foggy-sapnet/model_0006799.pth" MODEL.DA_HEAD.NAME "SAPNet" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val',)" SOLVER.MAX_ITER 999