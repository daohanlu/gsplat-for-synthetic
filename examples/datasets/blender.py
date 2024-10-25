import json
import os
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, NamedTuple, Union, Any
from PIL import Image

import numpy as np
import open3d as o3d

from .colmap import Dataset, similarity_from_cameras
from .normalize import align_principle_axes, transform_cameras, transform_points


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

CameraInfo = NamedTuple("CameraInfo", [("camera_ids", List[int]), ("image_names", List[str]), ("image_paths", List[str]),
                                       ("camtoworlds", np.ndarray), ("Ks_dict", Dict[int, np.ndarray]), ("params_dict", Dict[int, Dict[int, np.ndarray]]), ("imsize_dict", Dict[int, Tuple[int, int]]), ("mask_dict", Dict[int, Any])])
def readCamerasFromTransforms(dataset_dir: str, transformsfile: str, extension=".png", factor: int = 1) -> CameraInfo:
    # Built from INRIA's Gaussian Splatting Code
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/8a70a8cd6f0d9c0a14f564844ead2d1147d5a7ac/scene/dataset_readers.py#L179
    camera_ids = []
    image_names = []
    image_paths = []
    camtoworlds = []
    Ks_dict: Dict[int, np.ndarray] = {}
    params_dict: Dict[int, np.ndarray] = {}
    imsize_dict: Dict[int, Tuple[int, int]] = {}
    mask_dict: Dict[int, Any] = {}

    with open(os.path.join(dataset_dir, transformsfile)) as json_file:
        contents = json.load(json_file)
        assert 'camera_angle_x' in contents or 'focal_length_px' in contents, 'Either camera_angle_x or focal_length_px must be in the JSON file.'
        FOVX = contents["camera_angle_x"] if 'camera_angle_x' in contents else None
        FX = contents["focal_length_px"] if 'focal_length_px' in contents else None

        frames = contents["frames"]

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(dataset_dir, frame["file_path"] + extension)
            camera_id = idx

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # Note: I believe only c2w is needed in GSplat, though INRIA's code uses R and T from w2c.

            image_path = os.path.join(dataset_dir, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if FX is not None:
                fx = FX
            else:
                fx = fov2focal(FOVX, image.width)
            fy = fx
            cx, cy = 0.5 * image.width, 0.5 * image.height
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            camera_ids.append(camera_id)
            image_names.append(image_name)
            image_paths.append(image_path)
            camtoworlds.append(c2w)
            # Assume SIMPLE_PINHOLE camera model from Blender (no distortion)
            params_dict[camera_id] = np.empty(0, dtype=np.float32)
            imsize_dict[camera_id] = (image.width // factor, image.height // factor)
            mask_dict[camera_id] = None
    camtoworlds = np.array(camtoworlds)
    return CameraInfo(camera_ids, image_names, image_paths, camtoworlds, Ks_dict, params_dict, imsize_dict, mask_dict)


def readPointsFromPLY(plyfile: str) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
    pcd = o3d.io.read_point_cloud(plyfile)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    assert points.shape[0] > 0, "No points in the PLY file."
    assert points.shape[1] == 3, "Points must be 3D."
    if colors.shape[0] == 0:
        colors = None
    return points, colors


class AbstractParser(ABC):
    def __init__(
        self,
        data_dir: str,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir: str = data_dir
        self.normalize: bool = normalize
        self.test_every: int = test_every

        self.image_names: List[str] = []
        self.image_paths: List[str] = []
        self.camtoworlds: np.ndarray = None  # np.ndarray, (num_images, 4, 4)
        self.camera_ids: List[int] = None  # List[int], (num_images,)
        self.Ks_dict: Dict[int, np.ndarray] = None  # Dict of camera_id -> K
        self.params_dict: Dict[int, np.ndarray] = None  # Dict of camera_id -> params
        self.imsize_dict: Dict[int, Tuple[int, int]] = None  # Dict of camera_id -> (width, height)
        self.mask_dict: Union[None, Dict] = None  # Dict of camera_id -> mask
        self.points: Union[None, np.ndarray] = None  # np.ndarray, (num_points, 3)
        self.points_err: Union[None, np.ndarray] = None  # np.ndarray, (num_points,)
        self.points_rgb: Union[None, np.ndarray] = None  # np.ndarray, (num_points, 3)
        self.point_indices: Union[None, Dict[str, np.ndarray]] = None  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform: np.ndarray = np.eye(4)  # np.ndarray, (4, 4): transformation matrix for normalization
        self.scene_scale: float = None  # float

    def post_init_normalize(self):
        # size of the scene measured by cameras
        if len(self.camtoworlds) > 1:
            camera_locations = self.camtoworlds[:, :3, 3]
            scene_center = np.mean(camera_locations, axis=0)
            dists = np.linalg.norm(camera_locations - scene_center, axis=1)
            self.scene_scale = np.max(dists)
        else:
            self.scene_scale = 1.0  # dummy scene scale when it's not available
        if self.normalize:
            if self.points is None:
                raise ValueError("Points are not loaded. Cannot normalize without points.")
            else:
                # Normalize the world space.
                points = self.points
                T1 = similarity_from_cameras(self.camtoworlds)
                camtoworlds = transform_cameras(T1, self.camtoworlds)
                points = transform_points(T1, points)

                T2 = align_principle_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)

                self.camtoworlds = camtoworlds
                self.transform = T2 @ T1
                self.points = points

    @abstractmethod
    def parse(self):
        raise NotImplementedError()


class BlenderParser(AbstractParser):
    def __init__(
        self,
        data_dir: str,
        normalize: bool = False,
        test_every: int = 8,
    ):
        super().__init__(data_dir, normalize, test_every)
        self.parse()
        self.post_init_normalize()

    def parse(self):
        # Load NeRF Synthetic dataset (generated by Blender)
        cams = readCamerasFromTransforms(self.data_dir, "transforms_train.json")
        if os.path.exists(os.path.join(self.data_dir, "point_cloud.ply")):
            self.points, self.points_rgb = readPointsFromPLY(os.path.join(self.data_dir, "point_cloud.ply"))
            self.points_err = None
            self.point_indices = None
        else:
            self.points = None
            self.points_rgb = None
            self.points_err = None
            self.point_indices = None
        if self.points is not None and self.points_rgb is None:
            self.points_rgb = np.random.random(self.points.shape) * 255.0
        self.image_names = cams.image_names
        self.image_paths = cams.image_paths
        self.camtoworlds = cams.camtoworlds
        self.camera_ids = cams.camera_ids
        self.Ks_dict = cams.Ks_dict
        self.params_dict = cams.params_dict
        self.imsize_dict = cams.imsize_dict
        self.mask_dict = cams.mask_dict
