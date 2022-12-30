# This script is to visualize attention mask, output directory is under output
# DATASETS.TEST can have multiple test set, is source and target domain test sets, batch size is 1
# SOLVER.MAX_ITER determine how many image to visualize, if the number is greater than dataset size, just visualize entire dataset

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV2_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/sim10k2cityscapes-best/output-sim10k2city-sapv2-medm-mscam-cycle2-l0.1-e0.8-d0.3-22-10-15_00-30_49.48/model_0005599.pth" \
DATASETS.TEST "('sim10k_train', 'cityscapes-car2_val')" SOLVER.MAX_ITER 478


python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV2_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/sim10k-sapnet/model_0002199.pth" MODEL.DA_HEAD.NAME "SAPNet" \
DATASETS.TEST "('sim10k_train', 'cityscapes-car2_val')" SOLVER.MAX_ITER 478