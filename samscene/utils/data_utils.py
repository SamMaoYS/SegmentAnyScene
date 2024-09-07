import numpy as np
from dataclasses import dataclass


@dataclass
class CameraParameter:
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    name: str = "0"
