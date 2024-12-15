import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class RunningMean:
    def __init__(self, mean=0.0, n=0):
        self.mean = mean
        self.n = n

    def update(self, new_value):
        self.mean = self.running_mean(self.mean, self.n, new_value)
        self.n += 1

    def running_mean(self, mean, n, new_value):
        return (mean * n + new_value) / (n + 1)

    def get(self):
        return self.mean

    def reset(self):
        self.mean = 0.0
        self.n = 0


class RunningMeanDict:
    def __init__(self):
        self.means = {}

    def set_params(self, idx: int, mean: float, n: int):
        self.means[idx] = RunningMean(mean=mean, n=n)

    def update(self, new_value, idx: int):
        if idx not in self.means:
            self.means[idx] = RunningMean()
        self.means[idx].update(new_value)

    def get(self):
        return {k: v.get() for k, v in self.means.items()}


class RunningMeanList:
    def __init__(self, n: int):
        self.means = [RunningMean() for _ in range(n)]

    def set_params(self, idx: int, mean: float, n: int):
        self.means[idx] = RunningMean(mean=mean, n=n)

    def update_all(self, new_values: List[float]):
        for i, v in enumerate(new_values):
            if v is not None:
                self.means[i].update(v)

    def update_with_idx(self, new_value: float, idx: int):
        self.means[idx].update(new_value)

    def get(self):
        return [v.get() for v in self.means]


def pil_to_torch(images):
    return torch.stack(
        [
            torch.tensor(np.array(im)).permute(2, 0, 1).to(torch.float32) / 255.0
            for im in images
        ]
    )


def load_images(files_to_load, tar=None):
    if tar is not None:
        images = []
        for f in files_to_load:
            im = tar.extractfile(f)
            image = Image.open(io.BytesIO(im.read()))
            images.append(image)
    else:
        images = []
        for f in files_to_load:
            images.append(Image.open(f))
    return pil_to_torch(images)


def get_images_from_json(
    prompt: str, json_data: Dict
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """
    Get images and seeds for a prompt from JSON data.

    Args:
        prompt: str - The prompt to search for
        json_data: Dict - JSON with {prompt: [paths]} or {prompt: {seed: path}}

    Returns:
        Tuple[Optional[List[str]], Optional[List[int]]] - (image_paths, seeds) or (None, None)
    """
    if prompt not in json_data:
        return None, None

    value = json_data[prompt]
    processed_prompt = "_".join(prompt.split())

    # Handle dict format (seed: path)
    if isinstance(value, dict):
        sorted_items = sorted(value.items(), key=lambda x: int(x[0]))
        images = load_images([path for _, path in sorted_items])
        seeds = [int(seed) for seed, _ in sorted_items]
        return images, seeds

    # Handle list format
    if isinstance(value, list):
        images = []
        seeds = []

        # Try to extract seeds from paths
        for path in value:
            path_obj = Path(path)
            seed_match = re.search(rf".*{processed_prompt}_(\d+).*", path_obj.stem)

            images.append(path)
            if seed_match:
                seeds.append(int(seed_match.group(1)))
            else:
                seeds = None
                break
        images = load_images(images)
        return images, seeds

    return None, None


def get_images(prompt, all_file_names, tar=None):
    """
    Get images matching a prompt from either a tar file or directory.

    Args:
        prompt: str - The prompt to search for in filenames
        all_file_names: List[Path/str] - List of filenames from tar or directory
        tar: Optional[TarFile] - Tar archive containing images

    Returns:
        Tuple[List[Image], Optional[List[int]]] - List of loaded images and their seeds if available
    """
    processed_prompt = "_".join(prompt.split())

    # Find files containing the full processed prompt

    files_with_prompt = [
        f
        for f in all_file_names
        if processed_prompt + "_" in str(f if isinstance(f, Path) else Path(f).stem)
    ]

    if not files_with_prompt:
        return None, None

    # Check if files follow seed naming convention
    avaiable_seed = all(
        re.match(
            rf".*{processed_prompt}_\d+.*",
            str(f.stem if isinstance(f, Path) else Path(f).stem),
        )
        for f in files_with_prompt
    )

    if avaiable_seed:
        # Extract seeds and sort files
        seeds = [
            int(
                re.search(
                    rf".*{processed_prompt}_(\d+).*",
                    str(f.stem if isinstance(f, Path) else Path(f).stem),
                ).group(1)
            )
            for f in files_with_prompt
        ]
        files_to_load = [f for _, f in sorted(zip(seeds, files_with_prompt))]
    else:
        logger.warning(
            f"Some files do not respect the required format '{processed_prompt}_<seed_number>'. "
            "Fake seed will be used, do not consider the score per seed."
        )
        seeds = None
        files_to_load = files_with_prompt

    images = load_images(files_to_load, tar)
    return images, seeds
