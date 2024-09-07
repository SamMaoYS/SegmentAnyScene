import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData
from trimesh import Trimesh
from functools import partial
from tqdm.contrib.concurrent import thread_map

from samscene.utils import load_camera_from_trajectory


class IDTracker:
    def __init__(self):
        self.max_vote = 0
        self.best_id = -1
        self.vote = {}

    def add(self, id):
        if id not in self.vote:
            self.vote[id] = 1
        else:
            self.vote[id] += 1
        if self.vote[id] > self.max_vote:
            self.max_vote = self.vote[id]
            self.best_id = id
        return self.best_id

    def reset(self):
        self.max_vote = 0
        self.best_id = -1
        self.vote = {}


class TriangleTracker:
    def __init__(self, mesh_path, cameras_dir, pix_to_face_dir, sam_mask_dir):
        assert os.path.isfile(mesh_path), "Mesh file not found"
        assert os.path.isdir(cameras_dir), "Cameras directory not found"

        plydata = PlyData.read(mesh_path)
        x = np.asarray(plydata["vertex"]["x"])
        y = np.asarray(plydata["vertex"]["y"])
        z = np.asarray(plydata["vertex"]["z"])
        vertices = np.column_stack((x, y, z))
        triangles = np.vstack(plydata["face"].data["vertex_indices"])
        self.mesh = Trimesh(vertices=vertices, faces=triangles, process=False)
        self.cameras = load_camera_from_trajectory(cameras_dir)
        self.pix_to_face_dir = pix_to_face_dir
        self.sam_mask_dir = sam_mask_dir
        self.num_triangles = len(triangles)
        self.faces = self.mesh.faces
        self.vertices = self.mesh.vertices

        self.triangles_tracker = [None] * self.num_triangles
        for i in range(self.num_triangles):
            self.triangles_tracker[i] = IDTracker()
        self.triangles_tracker = np.asarray(self.triangles_tracker)

        self.h = 480
        self.w = 640

        self.face_to_object = np.zeros(self.num_triangles, dtype=int)

    def _load_sam_masks(self, name, level):
        sam_mask = np.full((self.h, self.w), -1, dtype=int)
        instance_id = 0
        for l in level:
            seg_path = os.path.join(self.sam_mask_dir + f"_level_{l}", name + ".npy")
            masks = np.load(seg_path, allow_pickle=True)
            masks = sorted(masks, key=lambda x: x["area"], reverse=True)

            num_masks = len(masks)
            for i in range(num_masks):
                valid_mask_pil = Image.fromarray(masks[i]["segmentation"]).convert("L")
                valid_mask_pil = valid_mask_pil.resize((self.w, self.h))
                valid_mask = np.array(valid_mask_pil).astype(bool)
                sam_mask[valid_mask] = instance_id
                instance_id += 1
        return sam_mask

    def compute_intance_size(self, cam):
        name = cam.name
        # load pix triangle indices
        pix_to_face = np.load(
            os.path.join(self.pix_to_face_dir, f"{name}_pix_to_face.npy")
        )
        valid_pix_mask = pix_to_face >= 0
        points = self.vertices[self.faces[pix_to_face[valid_pix_mask]]].reshape(-1, 3)
        bounds = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        volume = np.prod(bounds[1] - bounds[0])
        return volume

    def sort_masks(self):
        mask_face_size = {}
        partial_compute_intance_size = partial(self.compute_intance_size)
        instance_sizes = thread_map(
            partial_compute_intance_size,
            self.cameras,
        )
        mask_face_size = {ci: size for ci, size in enumerate(instance_sizes)}
        # sort mask face size by the values
        mask_face_size = dict(sorted(mask_face_size.items(), key=lambda item: item[1]))
        return mask_face_size

    def set_image_resolution(self, h, w):
        self.h = h
        self.w = w

    def track(self, level, output_path=None):
        def track_add(tri, tri_id):
            tri.add(tri_id)

        vec_track_add = np.vectorize(track_add)

        def track_best(tri):
            return tri.best_id

        vec_track_best = np.vectorize(track_best)

        global_object_index = 0
        h, w = self.h, self.w

        mask_face_size = self.sort_masks()
        camera_indices = list(mask_face_size.keys())
        for ci in tqdm(camera_indices):
            if not isinstance(ci, int):
                ci = int(ci)
            cam = self.cameras[ci]
            name = cam.name

            # load sam masks
            sam_mask = self._load_sam_masks(name, level)

            # load pix triangle indices
            pix_to_face = np.load(
                os.path.join(self.pix_to_face_dir, f"{name}_pix_to_face.npy")
            )
            valid_pix_mask = pix_to_face >= 0
            sam_mask[~valid_pix_mask] = -1

            unique_sam_mask, mask_areas = np.unique(
                sam_mask[sam_mask != -1], return_counts=True
            )
            # sort by mask area
            unique_sam_mask = unique_sam_mask[np.argsort(mask_areas)[::-1]]

            pix_object_ids = np.full((h, w), -1, dtype=int)
            tmp_indices = np.where(pix_to_face >= 0)
            pix_object_ids[tmp_indices] = vec_track_best(
                self.triangles_tracker[pix_to_face[tmp_indices]]
            )

            history_best = {}
            for i, mask_id in enumerate(unique_sam_mask):
                tmp_mask = sam_mask == mask_id
                # find the best object id by majority voting
                unique_best_ids, counts = np.unique(
                    pix_object_ids[tmp_mask],
                    return_counts=True,
                )
                counts_indices = np.argsort(counts)[::-1]
                best_id = unique_best_ids[counts_indices[0]]
                if best_id >= 0:
                    history_best[mask_id] = best_id
            # mask id with the same best id
            mask_ids = np.array(list(history_best.keys()))
            best_ids = np.array(list(history_best.values()))
            unique_best_ids, indices = np.unique(best_ids, return_index=True)
            for i in indices:
                mask_id = mask_ids[i]
                tmp_mask = sam_mask == mask_id
                faces = pix_to_face[tmp_mask]
                best_id = best_ids[i]
                vec_track_add(self.triangles_tracker[faces], best_id)

            # assign new ids for the rest of the masks
            tmp_masks = np.setdiff1d(unique_sam_mask, mask_ids[indices])
            for mask_id in tmp_masks:
                tmp_mask = sam_mask == mask_id
                faces = pix_to_face[tmp_mask]
                vec_track_add(self.triangles_tracker[faces], global_object_index)
                global_object_index += 1

        self.face_to_object = np.zeros(self.num_triangles, dtype=int)
        for face_id in range(self.num_triangles):
            tracker = self.triangles_tracker[face_id]
            self.face_to_object[face_id] = tracker.best_id

        if output_path is not None:
            os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
            np.save(output_path, self.face_to_object)

        return self.face_to_object

    def export_colored_mesh(self, output_path: str, seed: int = 42):
        np.random.seed(seed)
        unique_objects = np.unique(self.face_to_object)
        object_colors = {}
        for object_id in unique_objects:
            object_colors[object_id] = (np.random.rand(3) * 255).astype(np.uint8)

        output_mesh = self.mesh.copy()
        face_colors = np.zeros((self.num_triangles, 3), dtype=int)
        for face_id in range(self.num_triangles):
            object_id = self.face_to_object[face_id]
            face_colors[face_id] = object_colors[object_id]
        output_mesh.visual.face_colors = face_colors
        os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
        output_mesh.export(output_path)
