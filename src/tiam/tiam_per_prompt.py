from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from pandas import DataFrame
from ultralytics import YOLO

from .attribute_binding import eval_distance_to_colors
from .coco import getlabels2numbers
from .eval import eval
from .utils import RunningMean, RunningMeanDict, RunningMeanList


class TIAM_per_prompt:
    def __init__(
        self,
        model_path: str,
        save_dir: str,
        batch_size: int = 32,
        open_vocabulary: bool = False,
        min_conf_threshold: float = 0.25,
        confs_for_score: Optional[Union[float, List[float]]] = None,
        threshold_colors: float = 0.4,
    ):
        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        self.detect_color = False
        self.min_conf_threshold = min_conf_threshold
        if confs_for_score is None:
            self.confs_for_score = [self.min_conf_threshold]
        elif isinstance(confs_for_score, float):
            self.confs_for_score = [confs_for_score]
        elif isinstance(confs_for_score, list):
            self.confs_for_score = confs_for_score
        elif isinstance(confs_for_score, ListConfig):
            self.confs_for_score = OmegaConf.to_container(confs_for_score)
        else:
            raise ValueError(
                f"confs_for_score must be a float or a list of float, get {type(confs_for_score)}"
            )
        self.confs_for_score = sorted(self.confs_for_score)
        if self.confs_for_score[0] < 0 or self.confs_for_score[-1] > 1:
            raise ValueError("confs_for_score must be between 0 and 1")
        self.reset()

        self.model: YOLO = YOLO(
            model_path,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.labels2idx = getlabels2numbers()
        self.dtype = torch.float32
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_colors = threshold_colors

    def stream_batch_size(self, batch: torch.Tensor, seeds: List[int]):
        """return a generator that yield batch of images
        Args:
            batch (torch.Tensor): NCHW tensor
        """
        n = batch.size(0)
        for i in range(0, n, self.batch_size):
            yield batch[i : min(i + self.batch_size, n)], seeds[
                i : min(i + self.batch_size, n)
            ]

    def cast_to_floatdevice(self, tensor: torch.tensor):
        """Cast a tensor to float and device"""
        return tensor.to(self.dtype)

    @torch.no_grad()
    def predict(
        self,
        images: Union[torch.Tensor, np.ndarray],
        classes: List[str],
        prompt: str,
        color_classes: Dict[str, str] = None,
        seeds_used: Optional[List[int]] = None,
    ):
        """Predict the classes in the images

        Args:
            images (Union[torch.Tensor, np.array]): images to predict, images values must be between 0 and 1
            classes (List[str]): classes to predict
            color_classes (Dict[str, str], optional): dict of classes with associated color. Defaults to None.
            seeds_used (Optional[List[int]], optional): seeds used to generate the images. Defaults to None.
        """

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, torch.Tensor):
            pass
        else:
            raise ValueError("img must be a numpy array or a torch tensor")

        if color_classes is not None:
            self.detect_color = True
            self.reset()
        else:
            self.detect_color = False
            self.reset()

        # images = self.cast_to_floatdevice(images)
        images = images.to(self.device)

        if seeds_used is None:
            seeds_used = list([-1] * images.shape[0])
        elif len(seeds_used) != images.shape[0]:
            raise ValueError(
                f"seeds must have the same length as images, get {len(seeds_used)} and {images.shape[0]}"
            )

        if isinstance(seeds_used, torch.Tensor):
            seeds_used = seeds_used.tolist()

        for batch, seeds in self.stream_batch_size(images, seeds_used):

            classes2detect = [self.labels2idx[p] for p in classes]
            results = self.model.predict(
                source=batch,
                classes=classes2detect,
                conf=self.min_conf_threshold,
                imgsz=batch.shape[-2:],
                device=self.device,
            )

            for (idx_img, r), s in zip(enumerate(results), seeds):
                boxes = r.boxes.xyxyn.tolist()
                detected_classes = r.boxes.cls.tolist()
                conf = r.boxes.conf.tolist()
                if self.detect_color:
                    detected_colors = []
                    percentages_colors = []
                    if r.masks is not None:
                        masks = r.masks.xy
                        img_np = batch[idx_img].cpu().numpy().transpose(1, 2, 0)
                        img_np = (img_np * 255).astype(np.uint8)

                        for m in masks:
                            l, p = eval_distance_to_colors(
                                image=img_np,
                                mask=m,
                            )
                            detected_colors.append(l.tolist())
                            percentages_colors.append(p.tolist())
                else:
                    detected_colors = None
                    percentages_colors = None

                tiam_scores, count_order = eval(
                    bbox=boxes,
                    detected_classes=detected_classes,
                    conf=conf,
                    classes_from_prompt=classes,
                    conf_min=self.confs_for_score,
                    binding=self.detect_color,
                    detected_colors=detected_colors,
                    percentage_colors=percentages_colors,
                    colors_from_prompt=color_classes,
                    threshold_colors=self.threshold_colors,
                )
                self.save_scores(tiam_scores, count_order, s)

        # save the results
        self.save_prompt_scores_to_json(prompt)
        score_return = self.get_scores()
        self.reset()
        return score_return[self.confs_for_score[0]]["tiam"], self.confs_for_score[0]

    def save_prompt_scores_to_json(self, prompt):
        file_name = self.save_dir / f"{'_'.join(prompt.split())}.json"
        scores = self.get_scores()
        index = list(scores.keys())

        columns = {
            "conf": index,
            "tiam": [v["tiam"] for v in scores.values()],
            "n_class_detected": [v["n_class_detected"] for v in scores.values()],
            "tiam_per_seed": [v["tiam_per_seed"] for v in scores.values()],
            "prompt": prompt,
        }

        columns["count_order"] = [v["count_order"] for v in scores.values()]

        if self.detect_color:
            columns["tiam_gt_color"] = [v["tiam_gt_color"] for v in scores.values()]
            columns["tiam_gt_color_per_seed"] = [
                v["tiam_gt_color_per_seed"] for v in scores.values()
            ]
            columns["count_order_binding"] = [
                v["count_order_binding"] for v in scores.values()
            ]

        DataFrame(columns).to_json(file_name, indent=4)

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def reset(self):
        self.scores = {}
        for c in self.confs_for_score:
            self.scores[c] = {
                "tiam": RunningMean(),
                "n_class_detected": RunningMean(),
                "tiam_per_seed": RunningMeanDict(),
                "count_order": None,
            }
            if self.detect_color:
                self.scores[c]["tiam_gt_color"] = RunningMean()
                self.scores[c]["tiam_gt_color_per_seed"] = RunningMeanDict()
                self.scores[c][
                    "count_order_binding"
                ] = None  # proportion de bonne attribution *quand la classe est détecté* par rapport à position dans le prompt

    def get_scores(self):
        computes_scores = {}
        for c in self.confs_for_score:
            computes_scores[c] = {}
            computes_scores[c]["tiam"] = self.scores[c]["tiam"].get()

            computes_scores[c]["n_class_detected"] = self.scores[c][
                "n_class_detected"
            ].get()
            computes_scores[c]["count_order"] = self.scores[c]["count_order"].get()

            computes_scores[c]["tiam_per_seed"] = self.scores[c]["tiam_per_seed"].get()

            if self.detect_color:
                computes_scores[c]["tiam_gt_color"] = self.scores[c][
                    "tiam_gt_color"
                ].get()
                computes_scores[c]["tiam_gt_color_per_seed"] = self.scores[c][
                    "tiam_gt_color_per_seed"
                ].get()

                computes_scores[c]["count_order_binding"] = self.scores[c][
                    "count_order_binding"
                ].get()

        return dict(sorted(computes_scores.items()))

    def save_scores(self, tiam_score, count_order, seed):
        if self.detect_color:
            tiam_score, tiam_score_binding = tiam_score
            count_order, count_order_binding = count_order
        else:
            tiam_score_binding = None
            count_order_binding = None

        n_classes = len(count_order[self.confs_for_score[0]])
        for c in self.confs_for_score:
            self.scores[c]["tiam"].update(tiam_score[c][0])
            self.scores[c]["n_class_detected"].update(tiam_score[c][1] / n_classes)
            self.scores[c]["tiam_per_seed"].update(tiam_score[c][0], seed)
            if self.scores[c]["count_order"] is None:
                self.scores[c]["count_order"] = RunningMeanList(n_classes)
            self.scores[c]["count_order"].update_all(count_order[c])
            if self.detect_color:
                self.scores[c]["tiam_gt_color"].update(tiam_score_binding[c])
                self.scores[c]["tiam_gt_color_per_seed"].update(
                    tiam_score_binding[c], seed
                )
                if self.scores[c]["count_order_binding"] is None:
                    self.scores[c]["count_order_binding"] = RunningMeanList(n_classes)
                self.scores[c]["count_order_binding"].update_all(count_order_binding[c])
