source_model_path="pretrained/sim10k-baseline/model_0039999.pth"
# SAPNet + Cycle GAN
weight_DA=0.1

python tools/train_net.py --config-file "configs/sim10k2cityscapes/ablation/CycleGAN_R_50_C4.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnet-cyclegan-l${weight_DA}" MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.WEIGHTS ${source_model_path}
