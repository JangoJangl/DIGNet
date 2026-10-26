import numpy as np
import torch
from helper.representation import Representation
from helper.sampler import Sampler
from helper.transform import Transform
from helper.scaler import Scaler
from helper.visualize import Visualizer
from helper.augmentation import Augmentation
from helper.rendering import DepthRenderer


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
            device: str = 'cuda',
            root_directory: str = '/home/temp_store/jandl/Dataset/acronym',
            random_rotation: bool = False,
            partial_view_prob: float = 0.0,
            partial_oversample: int = 4,
            jitter_sigma: float = 0.0,
            partial_mode: str = 'hpr',
            eval_partial_view: bool = False,
    ):

        print(f"Using {device} for processing")
        self.num_points = num_points
        self.scale_factor = scale_factor
        self.use_normals = use_normals
        self.use_colors = use_colors
        self.device = device
        self.current_obj = None
        self.root_directory = root_directory
        self.representation_name = representation
        # augmentation, only applied to training samples (see __call__)
        self.random_rotation = random_rotation          # rotate object and grasps by a uniform random rotation
        self.partial_view_prob = partial_view_prob      # probability to replace the full cloud by a single view
        self.partial_oversample = partial_oversample    # dense cloud size factor used before hidden point removal
        self.jitter_sigma = jitter_sigma                # std of the gaussian noise on the points in metre
        # 'hpr': hidden point removal on a dense surface sample, 'render': ray cast depth image with sensor noise
        assert partial_mode in ('hpr', 'render'), f"unknown partial_mode {partial_mode}"
        self.partial_mode = partial_mode
        self.eval_partial_view = eval_partial_view      # also show partial views to the test set (no other augmentation)
        self.renderer = DepthRenderer() if partial_mode == 'render' else None
        self.__get_representation(representation)        # representation of grasp transformation
        self.__get_sampler(sampler)                      # sampler used to sample points from mesh
        self.__get_transform(transform)                  # COS transform applied to the object
        self.__get_scaler(scaler)                        # scale the object and grasp
        print("Preprocessor initialized")

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
            self.scaler = eval(f"Scaler(root_directory='{self.root_directory}').{scaler}")
        except AttributeError as e:
            print(e)

    # todo: create postprocess function with composition of inverse functions
    def preprocess(self, obj, grasps, augment=False):
        # Visualizer.visualize(obj, grasps)
        obj, grasps = Transform.start_transform(obj, grasps)
        # Visualizer.visualize(obj, grasps)
        self.current_obj = obj

        partial = (augment and np.random.rand() < self.partial_view_prob) or (not augment and self.eval_partial_view)
        points = None
        if partial and self.partial_mode == 'render':
            points = self.renderer.partial_cloud(obj, self.num_points, device=self.device, dtype=grasps.dtype)
            if points is not None:
                # centre on the visible points, as it has to be done for a real observation
                points, grasps = Augmentation.translate(points, grasps, -points.mean(dim=0))
        if points is None:
            num_points = self.num_points * self.partial_oversample if partial else self.num_points
            points = self.sampler(obj, num_points=num_points, use_normals=self.use_normals, use_colors=self.use_colors, device=self.device)
            if partial:
                points, grasps = Augmentation.partial_view(points, grasps, self.num_points)
        obj = points
        if augment:
            obj, grasps = self.augment(obj, grasps)
        # obj, grasps, self.inverse_transform_function = self.transform(obj, grasps)  #Todo fix transofrm of grasps
        # Visualizer.visualize(obj, grasps)
        grasps, self.inverse_representation_function = self.representation(grasps, device=self.device)
        # Visualizer.visualize(obj, grasps, inverse=self.inverse_representation_function)
        obj, grasps, self.inverse_scaler = self.scaler(obj, grasps, self.representation_name)
        # Visualizer.visualize(obj, grasps, inverse=self.inverse_representation_function)
        return obj, grasps

    def preprocess_cloud(self, points, grasps, tcp_depth=0.11217, augment=False):
        """
        Like preprocess, for precomputed clouds of the unified datasets (dataset_tools/): points [N, 3] and
        gripper base poses [K, 4, 4] in the same frame. Centres on the observed points (as for a real
        observation) and moves the grasp origin to the finger centre like start_transform does for ACRONYM.
        """
        points, grasps = Augmentation.translate(points, grasps, -points[:, :3].mean(dim=0))
        to_finger_centre = torch.eye(4, dtype=grasps.dtype, device=grasps.device)
        to_finger_centre[2, 3] = tcp_depth
        grasps = grasps @ to_finger_centre
        self.current_obj = points
        if augment:
            points, grasps = self.augment(points, grasps)
        grasps, self.inverse_representation_function = self.representation(grasps, device=self.device)
        points, grasps, self.inverse_scaler = self.scaler(points, grasps, self.representation_name)
        return points, grasps

    def augment(self, obj, grasps):
        if self.random_rotation:
            obj, grasps = Augmentation.rotate(obj, grasps, Augmentation.random_rotation(dtype=obj.dtype))
        if self.jitter_sigma > 0:
            obj = Augmentation.jitter(obj, sigma=self.jitter_sigma, clip=5 * self.jitter_sigma)
        return obj, grasps

    def postprocess(self, grasps):
        grasps = self.inverse_scaler(grasps)
        grasps = self.inverse_representation_function(grasps)
        # grasps = self.inverse_transform_function(grasps)
        return grasps

    def __call__(self, obj, grasps, augment=False):
        return self.preprocess(obj, grasps, augment=augment)
