source_model_path="pretrained/kitti-baseline/model_0027999.pth"
# SAPNet + Cycle GAN
L_DA=0.1

python tools/train_net.py --config-file "configs/kitti2cityscapes/ablation/CycleGAN_R_50_C4.yaml" --num-gpus 1 \
--setting-token "kitti2city-sapnet-cyclegan-l${L_DA}" MODEL.DA_HEAD.LOSS_WEIGHT ${L_DA} \
MODEL.WEIGHTS ${source_model_path}
