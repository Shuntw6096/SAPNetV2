source_model_path="pretrained/sim10k-baseline/model_0039999.pth"
# SAPNet + MEDM
weight_DA=0.1
weight_entropy=0.8
weight_diversity=0.3

python tools/train_net.py --config-file "configs/sim10k2cityscapes/ablation/MEDM_R_50_C4.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnet-medm-l${weight_DA}-e${weight_entropy}-d${weight_diversity}" MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT ${weight_entropy} MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT -${weight_diversity} \
MODEL.WEIGHTS ${source_model_path}
