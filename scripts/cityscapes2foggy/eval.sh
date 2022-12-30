# evaluate source only object detector
echo "start to evaluate source only object detector"
model_path="pretrained/cityscapes-baseline/model_0034999.pth"
python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}

# evaluate object detector with DA
echo "start to evaluate object detector with DA"
model_path="pretrained/cityscapes2foggy-best/output-city2foggy-sapv0-medm-mscam-cycle2-l0.6-e1.0-d0.2-22-10-11_20-10-49.63/model_0011199.pth"
python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}