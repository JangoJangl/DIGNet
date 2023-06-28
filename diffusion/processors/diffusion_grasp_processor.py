import numpy as np
from helper.representation import Representation
from helper.sampler import Sampler
from helper.transform import Transform
from helper.scaler import Scaler
from helper.visualize import Visualizer


# TODO: add noise for data augmentation
# TODO: add outlier removal or morphological operators

class DiffusionGraspProcessor(object):
    def __init__(
            self,
            representation: str = 'rot_6d_representation',
            sampler: str = 'vertices_sampler',
            transform: str = 'center_transform',
            scaler: str = 'var_scale',
            num_points: int = 1024,
            scale_factor: float = 1.0,
            rotation_factor: float = 2 * np.pi,
            use_normals: bool = False,
            use_colors: bool = False,
            device: str = 'cuda'):

        self.num_points = num_points
        self.scale_factor = scale_factor
        self.use_normals = use_normals
        self.use_colors = use_colors
        self.device = device
        self.current_obj = None

        self.__get_representation(representation)        # representation of grasp transformation
        self.__get_sampler(sampler)                      # sampler used to sample points from mesh
        self.__get_transform(transform)                  # COS transform applied to the object
        self.__get_scaler(scaler)                        # scale the object and grasp

    def __get_representation(self, representation):
        try:
            self.representation = eval(f"Representation.{representation}")
            self.representation_name = representation
        except AttributeError as e:
            print(e)

    def __get_sampler(self, sampler):
        try:
            self.sampler = eval(f"Sampler.{sampler}")
        except AttributeError as e:
            print(e)

    def __get_transform(self, transform):
        try:
            self.transform = eval(f"Transform.{transform}")
        except AttributeError as e:
            print(e)

    def __get_scaler(self, scaler):
        try:
            self.scaler = eval(f"Scaler().{scaler}")
        except AttributeError as e:
            print(e)

    # todo: create postprocess function with composition of inverse functions
    def preprocess(self, obj, grasps):
        # Visualizer.visualize(obj, grasps)
        obj, grasps = Transform.start_transform(obj, grasps)
        # Visualizer.visualize(obj, grasps)
        self.current_obj = obj

        obj = self.sampler(obj, num_points=self.num_points, use_normals=self.use_normals, use_colors=self.use_colors, device=self.device)
        # obj, grasps, self.inverse_transform_function = self.transform(obj, grasps)  #Todo fix transofrm of grasps
        # Visualizer.visualize(obj, grasps)
        grasps, self.inverse_representation_function = self.representation(grasps, device=self.device)
        # Visualizer.visualize(obj, grasps, inverse=self.inverse_representation_function)
        obj, grasps, self.inverse_scaler = self.scaler(obj, grasps, self.representation_name)
        # Visualizer.visualize(obj, grasps, inverse=self.inverse_representation_function)
        return obj, grasps

    def postprocess(self, grasps):
        grasps = self.inverse_scaler(grasps)
        grasps = self.inverse_representation_function(grasps)
        # grasps = self.inverse_transform_function(grasps)
        return grasps

    def __call__(self, obj, grasps):
        return self.preprocess(obj, grasps)
