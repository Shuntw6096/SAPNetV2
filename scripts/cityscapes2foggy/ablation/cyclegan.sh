source_model_path="pretrained/cityscapes-baseline/model_0034999.pth"
# SAPNet + Cycle GAN 
weight_DA=0.6

python tools/train_net.py --config-file "configs/cityscapes2foggy/ablation/CycleGAN_R_50_C4.yaml" --num-gpus 1 \
--setting-token "city2foggy-sapnet-cyclegan-l${weight_DA}" MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.WEIGHTS ${source_model_path}
