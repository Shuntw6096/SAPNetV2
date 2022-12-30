python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.85 \
MODEL.WEIGHTS "pretrained/cityscapes2foggy-best/output-city2foggy-sapv0-medm-mscam-cycle2-l0.6-e1.0-d0.2-22-10-11_20-10-49.63/model_0011199.pth" DATASETS.TEST "('foggy-cityscapes_val',)"

# python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.7 \
# MODEL.WEIGHTS "pretrained/cityscapes-baseline/model_0034999.pth" DATASETS.TEST "('foggy-cityscapes_val',)"

