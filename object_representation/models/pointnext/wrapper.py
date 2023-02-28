import torch

from ..interface import FeatureExtractor
from object_representation.openpoints.utils import EasyConfig
from object_representation.openpoints.models import build_model_from_cfg
from object_representation.openpoints.utils import load_checkpoint

class Pointnext(FeatureExtractor):

    def __init__(self):
        super().__init__()
        self.cfg_path = '/home/i53/student/jandl/repos/DIGNet/object_representation/models/pointnext/cfgs/shapenetpart/pointnext-s.yaml'
        self.ckpt = '/home/i53/student/jandl/repos/DIGNet/object_representation/models/pointnext/checkpoints/shapenetpart-train-pointnext-s-ngpus4-seed5011-20220821-170334-J6Ez964eYwHHPZP4xNGcT9_ckpt_best.pth'
        self.cfg = EasyConfig()
        self.cfg.load(self.cfg_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        self.model = build_model_from_cfg(self.cfg.model)
        load_checkpoint(self.model, pretrained_path=self.ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _call_model(self):
        inp = torch.unsqueeze(torch.cat((self.pc['pos'], self.pc['colors'], self.pc['heights']), 1), 0).transpose(1,2).contiguous()
        inp = inp.to(self.device)
        return self.model.encoder.forward(torch.unsqueeze(self.pc['pos'], 0), inp)  # just encode data

    def extract(self):
        self._preprocess()
        print(self.__dict__)
        return self._call_model()

