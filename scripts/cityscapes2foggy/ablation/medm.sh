source_model_path="pretrained/cityscapes-baseline/model_0034999.pth"
# SAPNet + MEDM
weight_DA=0.6
weight_entropy=1.0
weight_diversity=0.2

python tools/train_net.py --config-file "configs/cityscapes2foggy/ablation/MEDM_R_50_C4.yaml" --num-gpus 1 \
--setting-token "city2foggy-sapnet-medm-l${weight_DA}-e${weight_entropy}-d${weight_diversity}" MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT ${weight_entropy} MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT -${weight_diversity} \
MODEL.WEIGHTS ${source_model_path}
