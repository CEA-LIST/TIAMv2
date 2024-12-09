from typing import List
import torch
import numpy as np
import re
import logging
from PIL import Image
import io
from pathlib import Path

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


def get_images(prompt, all_file_names, save_dir_images=None, tar=None):
    """
    Get images matching a prompt from either a tar file or directory.

    Args:
        prompt: str - The prompt to search for in filenames
        all_file_names: List[Path/str] - List of filenames from tar or directory
        save_dir_images: Optional[Path] - Directory to save extracted images
        tar: Optional[TarFile] - Tar archive containing images

    Returns:
        Tuple[List[Image], Optional[List[int]]] - List of loaded images and their seeds if available
    """
    processed_prompt = "_".join(prompt.split())

    # Find files containing the full processed prompt

    files_with_prompt = [
        f
        for f in all_file_names
        if processed_prompt in str(f if isinstance(f, Path) else Path(f).stem)
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


# def get_images(prompt, all_file_names, save_dir_images=None, tar=None):

#     processed_prompt = "_".join(prompt.split())
#     seeds = None
#     if processed_prompt in all_file_names:
#         files_with_prompt = [
#             f for f in all_file_names if f.stem.startswith(processed_prompt)
#         ]
#         avaiable_seed = True
#         for file in files_with_prompt:
#             if not re.match(rf"{processed_prompt}_\d+", file.stem):
#                 avaiable_seed = False
#                 logger.warning(
#                     f"File '{file}' does not respect the required format. "
#                     f"The format should be '{processed_prompt}_<seed_number>'. "
#                     f"Fake seed will be used, do not consider the score per seed."
#                 )
#                 break
#         if avaiable_seed:
#             seeds = [int(f.stem.split("_")[-1]) for f in files_with_prompt]
#             # sort files by seed
#             files_to_load = [f for _, f in sorted(zip(seeds, files_with_prompt))]
#         else:
#             seeds = None
#             files_to_load = files_with_prompt

#         images = load_images(files_to_load, save_dir_images, tar)
#         return images, seeds


# def cure_data(self, row, batch_size):
#     #!todo faire evluer ca que pour les objets pour l'instant
#     if self.multi_template_style_prompt and batch_size > 1:
#         raise ValueError("batch size must be 1 if multi_template_style_prompt dataset")
#     if "object3" in row["labels_params"] and row["labels_params"]["object3"] == [""]:
#         row["labels_params"].pop("object3")
#         row["params"].pop("object3")
#     return row


# if self.multi_template_style_prompt:
#     batch = self.cure_data(batch, batch_size)
