import os

import hydra
import numpy as np
from omegaconf import DictConfig

import samscene
from samscene.utils import log, read_json
from samscene.connectivity import Pix2Triangles, TriangleTracker


@hydra.main(version_base="1.2", config_path="../configs", config_name="masks3d")
def main(cfg: DictConfig):
    samscene.attach_to_log()

    mesh_path = os.path.join(cfg.dataset_dir, cfg.scan_id, f"{cfg.scan_id}.ply")
    mesh_transformation_path = os.path.join(
        cfg.dataset_dir, cfg.scan_id, f"{cfg.scan_id}.align.json"
    )
    mesh_transformation = read_json(mesh_transformation_path)["coordinate_transform"]
    mesh_transformation = np.array(mesh_transformation).reshape(4, 4).transpose()

    camera_dir = os.path.join(cfg.output, cfg.scan_id, "camera")

    pix2triangles_dir = os.path.join(cfg.output, cfg.scan_id, "pix_to_triangles")
    os.makedirs(pix2triangles_dir, exist_ok=True)
    log.info(f"Computing visible triangles for {cfg.scan_id}...")
    pix2triangles = Pix2Triangles(mesh_path, camera_dir, transform=mesh_transformation)
    pix2triangles.compute_visible_triangles(
        pix2triangles_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    log.info(f"Visible triangles are computed and saved to {pix2triangles_dir}")

    log.info(f"Tracking triangles for {cfg.scan_id}...")
    masks3d_dir = os.path.join(cfg.output, cfg.scan_id, "masks3d")
    triangle_tracker = TriangleTracker(
        mesh_path, camera_dir, pix2triangles_dir, cfg.sam_mask
    )
    face_to_object_path = os.path.join(
        cfg.output, cfg.scan_id, "masks3d", "face_to_object.npy"
    )
    triangle_tracker.track(face_to_object_path)
    triangle_tracker.export_colored_mesh(
        os.path.join(cfg.output, cfg.scan_id, "masks3d", "colored_mesh.ply")
    )
    log.info(f"3D triangle masks are tracked and saved to {masks3d_dir}")


if __name__ == "__main__":
    main()
