import numpy as np
import torch

from ..interface import FeatureExtractor
from object_representation.openpoints.utils import EasyConfig
from object_representation.openpoints.models.build import build_model_from_cfg
from object_representation.openpoints.utils import load_checkpoint

#Todo: add Rotation of KOS in Config and create transformation
class Pc_Mae(FeatureExtractor):

    def __init__(self):
        super().__init__()
        self.cfg_path = '/home/i53/student/jandl/repos/DIGNet/object_representation/models/pc_mae/cfgs/pretrain.yaml'
        self.ckpt = '/home/i53/student/jandl/repos/DIGNet/object_representation/models/pc_mae/checkpoints/pretrain.pth'
        self.cfg = EasyConfig()
        self.cfg.load(self.cfg_path)
        self.model = None
        self.num_points = 1024  # Used in Paper
        self._load_model()

    def _load_model(self):
        self.model = build_model_from_cfg(self.cfg.model)
        load_checkpoint(self.model, pretrained_path=self.ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _call_model(self):
        neighborhood, center = self.model.group_divider(torch.unsqueeze(self.pc['pos'], dim=0))
        group_input_tokens = self.model.MAE_encoder.encoder(neighborhood)
        pos = self.model.MAE_encoder.pos_embed(center)
        x_vis = self.model.MAE_encoder.blocks(group_input_tokens, pos)
        return self.model.MAE_encoder.norm(x_vis)

    def extract(self):
        self._preprocess()
        print(self.__dict__)
        return self._call_model()

    def _obj_transform(self):
        self.obj.rotate((-np.pi / 2, 0, 0))




