import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.renderer.blending import BlendParams


class PyTorch3DRenderer:
    def __init__(self, mesh=None, camera=None, device="cuda"):
        self.device = torch.device(device)
        self.trimesh = None
        self.mesh = None
        self.camera = None

        self.width = 640
        self.height = 480

        if mesh is not None:
            self.set_trimesh(mesh)
            self.set_mesh(mesh)

        if camera is not None:
            self.set_camera(camera)

        self.set_device(device)
        self.raster_settings = RasterizationSettings(
            image_size=[self.height, self.width],
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
        )

    def set_camera(self, camera):
        self.camera = camera

    def set_trimesh(self, mesh):
        self.trimesh = mesh

    def set_mesh(self, mesh):
        self.mesh = Meshes(
            verts=[torch.from_numpy(mesh.vertices).float().to(self.device)],
            faces=[torch.from_numpy(mesh.faces).float().to(self.device)],
        )
        return self.mesh

    def set_device(self, device="cuda:0"):
        self.device = torch.device(device)

    def rasterize(self):
        pose = self.camera.extrinsics
        tmp_mesh = self.trimesh.copy()
        tmp_mesh.apply_transform(pose)
        self.set_mesh(tmp_mesh)
        # pose = self.camera.extrinsics
        R = (
            torch.from_numpy(np.eye(3))
            .permute(1, 0)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        T = torch.from_numpy(np.zeros((3, 1))).permute(1, 0).float().to(self.device)
        focal_length = [
            self.camera.intrinsics[0, 0],
            self.camera.intrinsics[1, 1],
        ]
        principal_point = [
            self.camera.intrinsics[0, 2],
            self.camera.intrinsics[1, 2],
        ]
        resolutoin = [self.height, self.width]
        cameras = PerspectiveCameras(
            focal_length=-torch.Tensor(focal_length)
            .float()
            .unsqueeze(0)
            .to(self.device),
            principal_point=torch.Tensor(principal_point)
            .float()
            .unsqueeze(0)
            .to(self.device),
            device=self.device,
            R=R,
            T=T,
            image_size=torch.Tensor(resolutoin).float().unsqueeze(0).to(self.device),
            in_ndc=False,
        )
        self.raster_settings.image_size = resolutoin

        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )

        fragments = rasterizer(self.mesh.to(self.device))
        return fragments
