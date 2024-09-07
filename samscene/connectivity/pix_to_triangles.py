import os
import copy

import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from trimesh import Trimesh
from functools import partial
from tqdm.contrib.concurrent import process_map

from samscene.renderer import PyTorch3DRenderer
from samscene.utils import load_camera_from_trajectory


def process_batch(cameras, mesh, output_dir):
    renderer = PyTorch3DRenderer(mesh=mesh)
    for ci, cam in tqdm(enumerate(cameras), total=len(cameras)):
        name = cam.name
        renderer.set_camera(cam)
        fragments = renderer.rasterize()
        pix_to_face = fragments.pix_to_face.cpu().numpy()[0, :, :, 0]

        np.save(os.path.join(output_dir, f"{name}_pix_to_face.npy"), pix_to_face)


class Pix2Triangles:
    def __init__(self, mesh_path, cameras_dir, transform=None):
        assert os.path.isfile(mesh_path), "Mesh file not found"
        assert os.path.isdir(cameras_dir), "Cameras directory not found"

        plydata = PlyData.read(mesh_path)
        x = np.asarray(plydata["vertex"]["x"])
        y = np.asarray(plydata["vertex"]["y"])
        z = np.asarray(plydata["vertex"]["z"])
        vertices = np.column_stack((x, y, z))
        triangles = np.vstack(plydata["face"].data["vertex_indices"])
        self.mesh = Trimesh(vertices=vertices, faces=triangles, process=False)
        if transform is not None:
            self.mesh.apply_transform(transform)

        self.cameras = load_camera_from_trajectory(cameras_dir)
        self.num_cameras = len(self.cameras)

    def compute_visible_triangles(
        self, output_dir="outputs", batch_size=100, num_workers=1
    ):
        os.makedirs(output_dir, exist_ok=True)
        if num_workers <= 1:
            process_batch(
                self.cameras,
                self.mesh,
                output_dir,
            )
        else:
            # split cameras into batches
            batch_cameras = np.split(
                self.cameras, np.arange(batch_size, len(self.cameras), batch_size)
            )
            process_map(
                partial(
                    process_batch,
                    mesh=copy.deepcopy(self.mesh),
                    output_dir=output_dir,
                ),
                batch_cameras,
                max_workers=num_workers,
            )
