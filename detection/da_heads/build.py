from detectron2.utils.registry import Registry

DA_HEAD_REGISTRY = Registry("DA_HEAD")

def build_DAHead(cfg):
    name = cfg.MODEL.DA_HEAD.NAME
    return DA_HEAD_REGISTRY.get(name)(cfg)

