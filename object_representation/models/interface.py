"""
Feature Extractor is interface for using feature extractors for point clouds.
If you want to use features from a new model you have to define a folder like the others and create the extractor
using this interface.

"""

import torch
import object_representation.openpoints.transforms.point_transformer_gpu as transforms
from helper import Visualizer


class FeatureExtractor(object):

    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.type = torch.float
        self.obj = None
        self.pc = {}
        self.transforms = []

    def _load_model(self):
        pass

    # def __call_model(self):
    #     pass

    def _preprocess(self):
        self._obj2pc()
        self._transform()

    def extract(self):
        pass

    def _add_transforms(self, trans):
        self.transforms += trans

    def add_object(self, obj):
        self.obj = obj

    def _obj2pc(self):
        self.pc['pos'] = torch.tensor(self.obj.mesh.vertices, dtype=self.type, device=self.device)
        self.pc['colors'] = torch.tensor(self.obj.mesh.visual.vertex_colors, dtype=self.type,
                                         device=self.device)[:, 0:3]/255

    def _transform(self):
        Visualizer.o3d_tensor(self.pc)
        for t in eval(f'self.cfg.{self.__class__.__name__}.datatransforms'):
            func = getattr(transforms, t)()
            self.pc = func(self.pc)
            Visualizer.o3d_tensor(self.pc)

    def clear(self):
        self.obj = None
        self.pc = {}

