source_model_path="pretrained/kitti-baseline/model_0027999.pth"
L_DA=0.1
E_MEDM=1.0
D_MEDM=0.05
python tools/train_net.py --config-file "configs/kitti2cityscapes/sapnetV2_R_50_C4.yaml" --num-gpus 1 \
--setting-token "kitti2city-sapnetv2-l${L_DA}-e${E_MEDM}-d${D_MEDM}" MODEL.DA_HEAD.LOSS_WEIGHT ${L_DA} \
MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT ${E_MEDM} MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT -${D_MEDM} \
MODEL.WEIGHTS ${source_model_path}
