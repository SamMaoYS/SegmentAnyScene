import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from semantic_sam import (
    build_semantic_sam,
    SemanticSamAutomaticMaskGenerator,
)


class SemanticSAM:
    def __init__(self, model_type="L", ckpt=None, level=[1, 2, 3, 4, 5], device="cuda"):
        self.sam = build_semantic_sam(model_type=model_type, ckpt=ckpt)
        self.sam.to(device=device)
        self.level = level
        self.mask_generator = SemanticSamAutomaticMaskGenerator(self.sam, level=level)
        self.transform = transforms.Compose(
            [
                transforms.Resize(480, interpolation=Image.BICUBIC),
            ]
        )

    def prepare_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Invalid image type")

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_ori = self.transform(image)

        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

        return image_ori, images

    def generate_masks(self, image):
        original_image, input_image = self.prepare_image(image)
        masks = self.mask_generator.generate(input_image)
        return original_image, masks

    @staticmethod
    def visualize_annotations(anns, alpah=0.35):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [alpah]])
            img[m] = color_mask
        return img

    def generate_masks_batch(self, image_files, output_dir, visualize=False):
        os.makedirs(output_dir, exist_ok=True)
        for image_file in tqdm(image_files):
            frame_id = os.path.splitext(os.path.basename(image_file))[0]
            original_image, masks = self.generate_masks(image_file)
            np.save(
                os.path.join(output_dir, frame_id + ".npy"),
                np.array(masks, dtype=object),
                allow_pickle=True,
            )

            if visualize:
                colorred_mask = self.visualize_annotations(masks)
                if colorred_mask is None:
                    continue
                alpha = colorred_mask[:, :, 3:]
                foreground = alpha * colorred_mask[:, :, :3]
                background = (1.0 - alpha) * (original_image.astype(np.float32) / 255.0)
                output = (foreground + background) * 255.0
                output = output.astype(np.uint8)

                output_pil = Image.fromarray(output)
                output_pil.save(os.path.join(output_dir, frame_id + ".jpg"))
