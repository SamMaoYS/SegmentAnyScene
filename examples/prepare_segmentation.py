import os

import hydra
from glob import glob
from omegaconf import DictConfig

import samscene
from samscene.segmentation import SemanticSAM


@hydra.main(version_base="1.2", config_path="../configs", config_name="semantic_sam")
def main(cfg: DictConfig):
    samscene.attach_to_log()

    # load semantic SAM model, default uses swinl backbone
    semantic_sam = SemanticSAM(
        model_type=cfg.model_type,
        ckpt=cfg.ckpt,
        level=cfg.level,
        device=cfg.device,
    )
    image_files = glob.glob(os.path.join(cfg.input_dir, "*.png"))
    # npy format predicted SAM files will be saved to the output directory
    # jpg images will be saved to the output directory if visualize is True
    semantic_sam.generate_masks_batch(
        image_files, cfg.output_dir, visualize=cfg.visualize
    )


if __name__ == "__main__":
    main()
