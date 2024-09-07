import os

import hydra
from omegaconf import DictConfig

import samscene
from samscene.dataset.multiscan import Decoder


@hydra.main(version_base="1.2", config_path="../configs", config_name="multiscan")
def main(cfg: DictConfig):
    samscene.attach_to_log()

    data_types = {
        "rgb": {"ext": "mp4", "fmt": "png"},
        "camera": {"ext": "jsonl", "fmt": "json"},
        # "depth": {"ext": "depth.zlib", "fmt": "png"},
        # "confidence": {"ext": "confidence.zlib", "fmt": "png"},
    }

    for key, value in data_types.items():
        input_path = os.path.join(
            cfg.dataset_dir, cfg.scan_id, f"{cfg.scan_id}.{value['ext']}"
        )
        output_path = os.path.join(cfg.output, cfg.scan_id, key)
        os.makedirs(output_path, exist_ok=True)
        tmp_path = os.path.join(cfg.output, cfg.tmp, cfg.scan_id)
        os.makedirs(tmp_path, exist_ok=True)

        decoder = Decoder(input_path, tmp_dir=tmp_path)
        decoder.set_frame_indices(cfg.start, cfg.stop, cfg.step)
        frame_param = {
            "width": cfg.width,
            "height": cfg.height,
            "scale": cfg.scale,
            "crop": cfg.crop,
        }
        decoder.export(
            output_path,
            format=value["fmt"],
            frame_param=frame_param,
            num_workers=cfg.num_workers,
        )


if __name__ == "__main__":
    main()
