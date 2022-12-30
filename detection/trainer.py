# All the trainer support mixed precision training
import logging
import weakref
import time
import torch
import torch.nn.functional as F
from functools import partial
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, create_ddp_model, SimpleTrainer, hooks
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from torch.nn.parallel import DataParallel, DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
from .evaluation.pascal_voc import PascalVOCDetectionEvaluator_
from .data.build import build_DA_detection_train_loader


class _DATrainer(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super(SimpleTrainer).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.source_domain_data_loader = source_domain_data_loader
        self.target_domain_data_loader = target_domain_data_loader
        self._source_domain_data_loader_iter = iter(source_domain_data_loader)
        self._target_domain_data_loader_iter = iter(target_domain_data_loader)
        self.loss_weight = loss_weight
        self.optimizer = optimizer

    def run_step(self):
        assert self.model.training, "[_DATrainer] model was changed to eval mode!"

        start = time.perf_counter()
        s_data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        t_data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time

        loss_dict = self.model(s_data, t_data)

        loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()


class _DAAMPTrainer(_DATrainer):
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, grad_scaler=None):
        
        unsupported = "_DAAMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported
        super().__init__(model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer)
        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[_DAAMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[_DAAMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        s_data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        t_data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time

        with autocast():
            loss_dict = self.model(s_data, t_data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

class DATrainer(DefaultTrainer):
    # one2one domain adpatation trainer
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        source_domain_data_loader = self.build_train_loader(cfg, 'source')
        target_domain_data_loader = self.build_train_loader(cfg, 'target')

        model = create_ddp_model(model, broadcast_buffers=False)

        loss_weight = {'loss_cls': 1, 'loss_box_reg': 1, 'loss_rpn_cls': 1, 'loss_rpn_loc': 1,\
        'loss_sap_source_domain': cfg.MODEL.DA_HEAD.LOSS_WEIGHT, 'loss_sap_target_domain': cfg.MODEL.DA_HEAD.LOSS_WEIGHT}

        if cfg.MODEL.DA_HEAD.RPN_MEDM_ON:
            loss_weight.update({'loss_target_entropy': cfg.MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT, 'loss_target_diversity': cfg.MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT})

        self._trainer = (_DAAMPTrainer if cfg.SOLVER.AMP.ENABLED else _DATrainer)(
            model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
    
    @classmethod
    def build_train_loader(cls, cfg, dataset_domain):
        return build_DA_detection_train_loader(cfg, dataset_domain=dataset_domain)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.LRScheduler(),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        return DefaultTrainer_.build_optimizer(cfg, model)

class DefaultTrainer_(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.SOLVER.NAME == 'default':
            return super().build_optimizer(cfg, model)

        elif cfg.SOLVER.NAME == 'adam':
            return torch.optim.Adam(
                [p for name, p in model.named_parameters() if p.requires_grad], 
                lr=cfg.SOLVER.BASE_LR,
                betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.NAME == 'adamw':
            return torch.optim.AdamW(
                [p for name, p in model.named_parameters() if p.requires_grad], 
                lr=cfg.SOLVER.BASE_LR,
                betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError('Not support {}'.format(cfg.SOLVER.NAME))

class SpatialAttentionVisualHelper:
    def __init__(self, cfg, test_set):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        data_loader = DATrainer.build_test_loader(cfg, test_set)
        self.data_size = len(data_loader) # for `visualize_attention_mask()` in train_net.py to track dataloader's size
        self.dataloader = iter(data_loader)

    def __call__(self):
        """
        Returns:
            predictions (dict):
                the output of the model
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            data = next(self.dataloader)
            predictions = self.model.visualize_spatial_attention_mask(data)
            return predictions


class GramCamForDomainClassfier:
    '''
    Both domain run grad cam seperately, target layer is spatial attention mask or backbone feature
    target layer name of spatial attention mask:
        da_heads.semantic_list.0.4 Conv2d
        da_heads.semantic_list.1.4 Conv2d
        da_heads.semantic_list.2.4 Conv2d
        da_heads.semantic_list.3.4 Conv2d
        ..., the numer is the same as number of window
    target layer name of backbone feature:
        the last convolution layer of backbone
    '''
    def __init__(self, cfg, domain, target_feature, test_set):
        assert domain in ['source', 'target']
        assert target_feature in ['backbone', 'attention mask']
        self.target_feature = target_feature
        self.domain = domain
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        data_loader = DATrainer.build_test_loader(cfg, test_set)
        self.data_size = len(data_loader) # for `grad_cam_domain_classsifier` in train_net.py to track dataloader's size
        self.dataloader = iter(data_loader)
        self.handlers = []
        if target_feature == 'attention mask':
            self.target_layer_name_list = self.attention_mask_target_layer_name()
        else:
            self.target_layer_name_list = self.backbone_target_layer_name()
        self.features = [None] * len(self.target_layer_name_list)
        self.gradients = [None] * len(self.target_layer_name_list)
        self._register_hook()
        # source: https://stackoverflow.com/questions/57323023/pytorch-loss-backward-and-optimizer-step-in-eval-mode-with-batch-norm-laye
        # eval mode does not block parameters to be updated, 
        # it only changes the behaviour of some layers (batch norm and dropout) during the forward pass
        self.model.eval()

    def _get_features_hook(self, module, input, output, key=0):
        self.features[key] = output.detach()

    def _get_grads_hook(self, module, grad_in, grad_out, key=0):
        self.gradients[key] = grad_out[0].detach()

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def _register_hook(self):
        if self.target_feature == 'attention mask':
            target_module = self.model.da_heads
            for (name, module) in target_module.named_modules():
                if name in self.target_layer_name_list:
                    key = int(name.split('.')[-2])
                    self.handlers.append(module.register_forward_hook(partial(self._get_features_hook, key=key)))
                    self.handlers.append(module.register_full_backward_hook(partial(self._get_grads_hook, key=key)))

        else:
            target_module = self.model.backbone
            for (name, module) in target_module.named_modules():
                if name in self.target_layer_name_list:
                    self.handlers.append(module.register_forward_hook((self._get_features_hook)))
                    self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))


    def attention_mask_target_layer_name(self):
        '''
        Return: list[str], taget layer name in domain classifier
        '''
        name_list = set()
        for name, m in self.model.da_heads.named_modules():
            if 'semantic_list' in name and name.split('.')[-1] == '4' and isinstance(m, torch.nn.Conv2d):
                name_list.add(name)
        return name_list

    def backbone_target_layer_name(self):
        '''
        Return: list[str], taget layer name in domain classifier
        '''
        layer_name = None
        for name, m in self.model.backbone.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                layer_name = name
        return [layer_name]

    def __call__(self):
        data = next(self.dataloader) # batch size is 1
        _, im_h, im_w = data[0]['image'].size()
        self.model.zero_grad()
        domain_vector:torch.Tensor = self.model.get_domain_vector(data)
        domain_vector = domain_vector.softmax(dim=1)[0] # [N, 2], where N is batch size, is 1
        if self.domain == 'target':
            domain_vector[1].backward()
            prob = domain_vector[1]
        else:
            domain_vector[0].backward()
            prob = domain_vector[0]
        final_cam = 0.
        for g, f in zip(self.gradients, self.features):
            weight = g.mean(dim=(-2,-1), keepdim=True)
            cam:torch.Tensor = weight * f
            cam = cam.sum(dim=(0, 1)) # [H, W]
            cam = F.relu(cam) # [H, W]
            cam -= cam.min()
            if cam.max() != 0: cam /= cam.max()
            cam = F.interpolate(cam.view(1, 1, cam.size(0), cam.size(1)), size=(im_h, im_w), mode='bicubic', align_corners=True)
            cam = cam.view(cam.size(-2), cam.size(-1))
            cam = cam.clamp(min=0., max=1.)
            final_cam += cam
        final_cam = final_cam / len(self.target_layer_name_list)
        final_cam = final_cam / final_cam.max()
        return final_cam, data[0]['image'], data[0]['file_name'], prob


class GramCamForObjectDetection:
    '''
    Grad cam for object detector, target layer is the last convolution layer of roi_head
    '''
    def __init__(self, cfg, test_set):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.target_layer:str = self.target_layer_name()
        print(self.target_layer)
        data_loader = DATrainer.build_test_loader(cfg, test_set)
        self.data_size = len(data_loader) # for `grad_cam_object_detection` in train_net.py to track dataloader's size
        self.dataloader = iter(data_loader)
        self.handlers = []
        self.feature = None
        self.gradient = None
        self._register_hook()
        self.model.eval()

    def _get_features_hook(self, module, input, output):
        self.feature = output.detach()

    def _get_grads_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def target_layer_name(self):
        '''
        Return: str, taget layer name in domain classifier
        '''
        layer_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                layer_name = name
        return layer_name

    def _register_hook(self):
        for (name, module) in self.model.named_modules():
            if name == self.target_layer:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))
                break

    def __call__(self):
        data = next(self.dataloader) # batch size is 1
        im_h, im_w = data[0]['height'], data[0]['width']
        self.model.zero_grad()
        output = self.model.inference(data)
        output = output[0]['instances']
        final_cam = torch.zeros((im_h, im_w), device=next(self.model.parameters()).device)
        for score, box, ind in zip(output.scores, output.pred_boxes, output.indices):
            score.backward(retain_graph=True)
            gradient = self.gradient[ind]
            feature:torch.Tensor = self.feature[ind]
            weight = gradient.mean(dim=(1,2), keepdim=True)
            weight = weight.sum(dim=(1,2), keepdim=True)
            cam = F.relu((weight * feature).sum(dim=0))
            cam -= cam.min()
            ma = cam.max()
            if ma > 0:
                cam /= ma
            x1, y1, x2, y2 = box.detach()
            x1, y1, x2, y2 = x1.int(), y1.int(), x2.int(), y2.int()
            cam *= 255
            cam = F.interpolate(cam.view(1, 1, cam.size(0), cam.size(1)), size=(int((y2-y1).item()), int((x2-x1).item())), mode='bicubic', align_corners=True)
            cam = cam.view(cam.size(-2), cam.size(-1)) / 255
            cam = cam.clamp(min=0., max=1.)
            final_cam[y1:y2, x1:x2] += cam
            self.model.zero_grad()
        final_cam = final_cam.clamp(min=0., max=1.)
        return final_cam, data[0]['file_name']