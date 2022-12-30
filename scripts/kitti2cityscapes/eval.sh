# evaluate source only object detector
echo "start to evaluate source only object detector"
model_path="pretrained/kitti-baseline/model_0027999.pth"
python tools/train_net.py --config-file "configs/kitti2cityscapes/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}

# evaluate object detector with DA
echo "start to evaluate object detector with DA"
model_path="pretrained/kitti2cityscapes-best/output-kitti2city-sapv2-medm-mscam-cycle2-l0.1-e1.0-d0.05-22-10-30_23-16/model_0006399.pth"
python tools/train_net.py --config-file "configs/kitti2cityscapes/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}
