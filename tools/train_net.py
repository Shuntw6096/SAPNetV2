from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import verify_results
from detectron2.utils.events import EventStorage, TensorboardXWriter
import os
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(os.getcwd())

from detection.trainer import (
    DATrainer, 
    DefaultTrainer_, 
    SpatialAttentionVisualHelper, 
    GramCamForDomainClassfier,
    GramCamForObjectDetection,
)

from detection.hooks import EvalHook_

# register datasets
import detection.data.register

# register compoments
import detection.modeling
from detection.meta_arch.sap_rcnn import SAPRCNN
from detection.da_heads import build_DAHead
from detection.modeling.rpn import SAPRPN




def add_saprcnn_config(cfg):
    from detectron2.config import CfgNode as CN
    _C = cfg
    _C.MODEL.DOMAIN_ADAPTATION_ON = False
    _C.MODEL.DA_HEAD = CN()
    _C.MODEL.DA_HEAD.IN_FEATURE = "res4"
    _C.MODEL.DA_HEAD.IN_CHANNELS = 256
    _C.MODEL.DA_HEAD.NUM_ANCHOR_IN_IMG = 5
    _C.MODEL.DA_HEAD.EMBEDDING_KERNEL_SIZE = 3
    _C.MODEL.DA_HEAD.EMBEDDING_NORM = True
    _C.MODEL.DA_HEAD.EMBEDDING_DROPOUT = True
    _C.MODEL.DA_HEAD.FUNC_NAME = 'cross_entropy'
    _C.MODEL.DA_HEAD.POOL_TYPE = 'avg'
    _C.MODEL.DA_HEAD.LOSS_WEIGHT = 1.0
    _C.MODEL.DA_HEAD.WINDOW_STRIDES = [2, 2, 2]
    _C.MODEL.DA_HEAD.WINDOW_SIZES = [3, 6, 9]
    _C.MODEL.PROPOSAL_GENERATOR.NAME = "SAPRPN"

    _C.MODEL.DA_HEAD.NAME = 'SAPNetMSCAM'
    _C.MODEL.DA_HEAD.RPN_MEDM_ON = False
    _C.MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT = 0.
    _C.MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT = 0.

    _C.DATASETS.SOURCE_DOMAIN = CN()
    _C.DATASETS.SOURCE_DOMAIN.TRAIN = ()
    _C.DATASETS.TARGET_DOMAIN = CN()
    _C.DATASETS.TARGET_DOMAIN.TRAIN = ()
    _C.SOLVER.NAME = "default"

def check_cfg(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        assert cfg.MODEL.DA_HEAD.LOSS_WEIGHT > 0,  'MODEL.DA_HEAD.LOSS_WEIGHT must be greater than 0'
    
    if cfg.MODEL.DA_HEAD.RPN_MEDM_ON:
        assert cfg.MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT > 0, 'MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT must be greater than 0'
        assert cfg.MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT < 0, 'MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT must be smaller than 0'
    

def setup(args):
    cfg = get_cfg()
    add_saprcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    now = datetime.now()

    if args.resume and args.eval_all:
        cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / 'eval')
    if not args.resume:
        cfg.OUTPUT_DIR = './outputs/output-{}'.format(now.strftime("%y-%m-%d_%H-%M"))
        if args.setting_token:
            cfg.OUTPUT_DIR = './outputs/output-{}-{}'.format(args.setting_token, now.strftime("%y-%m-%d_%H-%M"))
    cfg.freeze()
    check_cfg(cfg)
    if not (args.test_images or args.visualize_attention_mask or args.gcs or args.gct or args.gco):
        default_setup(cfg, args)
    elif args.visualize_attention_mask or args.gcs or args.gct or args.gco or args.test_images:
        if args.gcs or args.gct: 
            assert args.attention_mask or args.backbone_feature, 'please determine which feature to visualize'
            assert cfg.MODEL.DOMAIN_ADAPTATION_ON, 'domain classfier is used, cfg.MODEL.DOMAIN_ADAPTATION_ON should be True'
        # set random seed
        rank = comm.get_rank()
        seed = cfg.SEED
        seed_all_rng(None if seed < 0 else seed + rank)
    return cfg

def test_images(cfg):
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import load_voc_instances
    import cv2
    predictor = DefaultPredictor(cfg)

    for dataset_name in cfg.DATASETS.TEST:
        now = datetime.now()
        output_dir = Path(__file__).parent.parent/ 'test_images'/ (dataset_name + '-' + now.strftime("%y-%m-%d_%H-%M"))
        output_dir.mkdir(parents=True, exist_ok=True)
        dirname = MetadataCatalog.get(dataset_name).get('dirname')
        split = MetadataCatalog.get(dataset_name).get('split')
        thing_classes = MetadataCatalog.get(dataset_name).get('thing_classes')
        for d in iter(load_voc_instances(dirname, split, thing_classes)):
            im = cv2.imread(d.get('file_name'), cv2.IMREAD_COLOR)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=0.6, instance_mode=ColorMode.IMAGE)
            outputs = predictor(im)
            cv2.imwrite(str(output_dir/'{}.jpg').format(Path(d.get('file_name')).stem), v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1])

def visualize_attention_mask(cfg):
    '''
    helper function for visualizing spattal attention mask of domain classfier
    '''
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from collections import defaultdict
    now = datetime.now()
    fig = plt.figure() # memory comsuming
    for test_set in cfg.DATASETS.TEST:
        output_dir = Path(__file__).parent.parent/ 'test_images'/ ('attention-mask' + '-' + cfg.MODEL.DA_HEAD.NAME + '-' + now.strftime("%y-%m-%d_%H-%M-%S")) / test_set
        output_dir.mkdir(parents=True, exist_ok=True)
        predictor = SpatialAttentionVisualHelper(cfg, test_set)
        max_iter = min(cfg.SOLVER.MAX_ITER, predictor.data_size)
        for _ in range(max_iter):
            attention_masks = predictor()
            n = 0
            # track attention mask's verage max value and average value
            record = defaultdict(lambda : {'average max value':0, 'average':0})
            for name in attention_masks:
                n += 1
                print(name)
                idx = name.split('-')[-1]
                dir_name = output_dir/name[:-len(idx)-1]
                dir_name.mkdir(exist_ok=True)
                mask = attention_masks[name].cpu().numpy()
                maxV = round(np.max(mask), 2)
                record[dir_name.name]['average max value'] += maxV
                record[dir_name.name]['average'] +=  mask.mean()
                maxV = str(maxV)
                mask = cv2.resize(mask, (2048,1024))
                max_ = np.max(mask)
                mask[mask < 0] = 0
                if max_ > 0:
                    mask = mask / max_
                mask = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[..., ::-1]
                plt.imshow(mask)
                fname = dir_name/(idx + '-' + maxV + '.jpg')
                plt.axis('off')
                plt.savefig(str(fname), bbox_inches='tight', pad_inches=0.0)
                plt.clf() # clear plot
        
            num_window = n / len(record)
            with open(output_dir/'track.txt', 'a') as track:
                for name, d in record.items():
                    print(name, file=track)
                    print(f'average max value:{d["average max value"]/num_window}, average:{d["average"]/num_window}', file=track)
        plt.close(fig)

def grad_cam_domain_classsifier(cfg, domain:str, target_feature:str):
    '''
    helper function for visualizing grad cam of domain classfier
    '''
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    now = datetime.now()
    fig = plt.figure() # memory comsuming
    for test_set in cfg.DATASETS.TEST:
        output_dir = Path(__file__).parent.parent/ 'test_images'/ ('grad-cam' + '-' + cfg.MODEL.DA_HEAD.NAME + '-' + domain + '-' + target_feature.replace(' ','-') + '-' + now.strftime("%y-%m-%d_%H-%M-%S")) / test_set
        output_dir.mkdir(parents=True, exist_ok=True)
        cam = GramCamForDomainClassfier(cfg, domain, target_feature, test_set)
        max_iter = min(cfg.SOLVER.MAX_ITER, cam.data_size)
        sub = n = 0
        for _ in range(max_iter):
            mask, im, im_name, prob = cam()
            prob = round(prob.item(), 2)
            sub += prob
            n += 1
            im = im.permute(1,2,0).numpy()[:,:,::-1]
            mask = mask.cpu().numpy()
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]  # bgr to rgb
            im = im / 255.+ heatmap
            im -= np.max(np.min(im), 0)
            im /= np.max(im)
            im *= 255.
            im = np.uint8(im)
            fname = output_dir / (Path(im_name).stem + f'-{prob}' + '.jpg')
            plt.imshow(im)
            plt.axis('off')
            plt.savefig(str(fname), bbox_inches='tight', pad_inches=0.0)
            plt.clf() # clear plot
            print(fname.stem)
        with open(str(output_dir.parent/'track.txt'), 'a') as tr:
            print('mean accuracy [0, 1]:', test_set, round(sub/n, 3), file=tr)      
    plt.close(fig)

def grad_cam_object_detection(cfg):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    now = datetime.now()

    for test_set in cfg.DATASETS.TEST:
        output_dir = Path(__file__).parent.parent/ 'test_images'/ ('grad-cam-object-detection-' + now.strftime("%y-%m-%d_%H-%M-%S")) / test_set
        output_dir.mkdir(parents=True, exist_ok=True)
        cam = GramCamForObjectDetection(cfg, test_set)
        max_iter = min(cfg.SOLVER.MAX_ITER, cam.data_size)
        for _ in range(max_iter):
            mask, im_name = cam()
            im = cv2.imread(im_name)[..., ::-1] # bgr to rgb
            mask = mask.cpu().numpy()
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]  # bgr to rgb
            im = im / 255. + heatmap
            im -= np.max(np.min(im), 0)
            im /= np.max(im)
            im *= 255.
            im = np.uint8(im)[...,::-1]
            fname = output_dir / (Path(im_name).stem + '.jpg')
            cv2.imwrite(str(fname), im)
            print(fname.stem)


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DATrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DATrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if args.eval_all:
        assert args.resume, "add resume flag"
        model = DATrainer.build_model(cfg)
        pth_list = DetectionCheckpointer(model, save_dir=str(Path(cfg.OUTPUT_DIR).parent)).get_all_checkpoint_files()
        def test_():
            res = DATrainer.test(cfg, model)
            return res
        writer = TensorboardXWriter(cfg.OUTPUT_DIR)
        pth_list.sort() # sorted model name by iteration, model_001999.pth, ...
        with EventStorage() as storage:
            for p in pth_list:
                d = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                    p, resume=args.resume
                )
                storage._iter = d['iteration']
                EvalHook_(0, test_)._do_eval() # 0, eval every iteration
                writer.write()
            writer.close()
        return

    if args.visualize_attention_mask:
        visualize_attention_mask(cfg)
        return

    if args.gcs or args.gct:
        if args.gcs:
            if args.backbone_feature:
                grad_cam_domain_classsifier(cfg, 'source', 'backbone')
            else:
                grad_cam_domain_classsifier(cfg, 'source', 'attention mask')
        elif args.gct:
            if args.backbone_feature:
                grad_cam_domain_classsifier(cfg, 'target', 'backbone')
            else:
                grad_cam_domain_classsifier(cfg, 'target', 'attention mask')
        return

    if args.gco:
        grad_cam_object_detection(cfg)
        return

    if args.test_images:
        test_images(cfg)
        return

    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        trainer = DATrainer(cfg)
    else:
        trainer = DefaultTrainer_(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--test-images", action="store_true", help="output predicted bbox to test images")
    parser.add_argument("--setting-token", help="add some simple profile about this experiment to output directory name")
    parser.add_argument("--eval-all", action="store_true", help="eval all checkpoint under the cfg.OUTPUT_DIR, and put result to its sub dir")
    parser.add_argument("--visualize-attention-mask", action="store_true", help="visualize attention mask, output directory is under test_images")
    # args.gcs and args.gct are conflicting options
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--grad-cam-source-doamin", dest='gcs',action="store_true", help="visualize source domain, use --attention-mask or --backbone-feature to select the feature to visualize, store visualization at the cfg.OUTPUT_DIR")
    group1.add_argument("--grad-cam-target-doamin", dest='gct', action="store_true", help="visualize target domain using grad cam, use --attention-mask or --backbone-feature to select the feature to visualize, store visualization at the cfg.OUTPUT_DIR")
    # args.attention_mask and args.backbone_feature are conflicting options
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--attention-mask",action="store_true", help="visualize attention mask using grad cam")
    group2.add_argument("--backbone-feature",action="store_true", help="visualize backbone feature using grad cam")
    parser.add_argument("--grad-cam-object-detection", dest='gco', action="store_true", help="visualize grad cam of object detector,  output directory is under test_images")
    args = parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )