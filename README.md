# SegmentAnyScene

Segment-Anything on 3D scenes meshes.

## Introduction

Segment Any Scene extends the capabilities of the Segment Anything Model, originally designed for 2D images, to work with 3D scene meshes.

## Installation

1. Create a new conda environment and activate it.
    ```bash
    conda create -n samscene python=3.9 -y
    conda activate samscene
    ```

2. Install [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) by following their instructions [here](https://github.com/UX-Decoder/Semantic-SAM?tab=readme-ov-file#unicorn-getting-started).

    Note: you need to add the folder path to your PYTHONPATH variable manually.
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/Semantic-SAM
    ```

3. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d), please refer to their [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details or using conda.
    ```bash
    conda install pytorch3d::pytorch3d
    ```

4. Install this package.
    ```bash
    pip install -e .
    ```

### TODO
    - [ ] add a standalone semantic-sam inference package for easier installation
    - [ ] add a installation script for easier installation all packages


## Dataset

Download multiscan example dataset from [HuggingFace](https://huggingface.co/datasets/ysmao/multiscan_example).
```bash
huggingface-cli download --resume-download ysmao/multiscan_example --local-dir ./data/multiscan/scene_00021_00 --local-dir-use-symlinks False --repo-type dataset
```

For downloading the entire multiscan dataset, please refer to [Multiscan Dataset](https://github.com/smartscenes/multiscan?tab=readme-ov-file#multiscan-dataset).

### TODO
- [ ] Add support for ScanNet dataset
- [ ] Add support for Arbitrary 3D scene meshes with/without camera trajectories

