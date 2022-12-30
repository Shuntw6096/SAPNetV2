python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 \
MODEL.WEIGHTS "pretrained/sim10k2cityscapes-best/output-sim10k2city-sapv2-medm-mscam-cycle2-l0.1-e0.8-d0.3-22-10-15_00-30_49.48/model_0005599.pth" DATASETS.TEST "('cityscapes-car2_val',)"

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 \
# MODEL.WEIGHTS "pretrained/sim10k-baseline/model_0039999.pth" DATASETS.TEST "('cityscapes-car2_val',)"