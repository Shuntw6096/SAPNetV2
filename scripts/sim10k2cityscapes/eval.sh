# evaluate source only object detector
echo "start to evaluate source only object detector"
model_path="pretrained/sim10k-baseline/model_0039999.pth"
python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}

# evaluate object detector with DA
echo "start to evaluate object detector with DA"
model_path="pretrained/sim10k2cityscapes-best/output-sim10k2city-sapv2-medm-mscam-cycle2-l0.1-e0.8-d0.3-22-10-15_00-30_49.48/model_0005599.pth"
python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}