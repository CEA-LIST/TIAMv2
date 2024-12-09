from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import ListConfig, OmegaConf
from pandas import DataFrame
from ultralytics import YOLO
import torch

from .utils import RunningMean, RunningMeanDict, RunningMeanList
from .attribute_binding import eval_distance_to_colors
from .coco import getlabels2numbers
from .eval import eval


class TIAMScore:
    def __init__(
        self,
        confs_for_score: Optional[Union[float, List[float]]] = None,
        min_conf_threshold: float = 0.25,
        multi_template_style_prompt: bool = False,
    ):
        self.seg = False
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
        self.multi_template_style_prompt = multi_template_style_prompt

    def reset(self):
        self.scores = {}
        for c in self.confs_for_score:
            self.scores[c] = {
                "tiam": RunningMean(),
                "n_class_detected": RunningMean(),
                "tiam_per_seed": RunningMeanDict(),
                "count_order": None,
            }
            if self.seg:
                self.scores[c]["tiam_gt_color"] = RunningMean()
                self.scores[c]["tiam_gt_color_per_seed"] = RunningMeanDict()
                self.scores[c][
                    "count_order_binding"
                ] = None  # proportion de bonne attribution *quand la classe est détecté* par rapport à position dans le prompt

    def set_values_from_df(self, df: DataFrame):
        conf = df["conf"].unique()
        # sort conf
        conf = sorted(conf)
        self.confs_for_score = conf
        # todo a remettre
        if "tiam_gt_color" in df.columns:
            self.seg = True
        seeds = list(df["tiam_per_seed"].iloc[0].keys())
        n_seeds = len(seeds)
        self.scores = {}
        for c in conf:
            n_prompt = len(df[df["conf"] == c])
            n_images = n_prompt * n_seeds

            self.scores[c] = {
                "tiam": RunningMean(
                    mean=df[df["conf"] == c]["tiam"].mean(), n=n_images
                ),
            }

            self.scores[c]["tiam_per_seed"] = RunningMeanDict()
            for s in seeds:
                self.scores[c]["tiam_per_seed"].set_params(
                    idx=s,
                    mean=df[df["conf"] == c]["tiam_per_seed"]
                    .apply(lambda x: x[s])
                    .mean(),
                    n=n_prompt,
                )
            n_classes = len(df[df["conf"] == c]["count_order"].iloc[0])
            if not self.multi_template_style_prompt:
                self.scores[c]["count_order"] = RunningMeanList(n_classes)
                for i in range(n_classes):
                    self.scores[c]["count_order"].set_params(
                        idx=i,
                        mean=df[df["conf"] == c]["count_order"]
                        .apply(lambda x: x[i])
                        .mean(),
                        n=n_images,
                    )
                self.scores[c]["n_class_detected"] = RunningMean(
                    mean=df[df["conf"] == c]["n_class_detected"].mean(), n=n_images
                )
            if self.seg:
                self.scores[c]["tiam_gt_color"] = RunningMean(
                    mean=df[df["conf"] == c]["tiam_gt_color"].mean(), n=n_images
                )
                self.scores[c]["tiam_gt_color_per_seed"] = RunningMeanDict()
                for s in seeds:
                    self.scores[c]["tiam_gt_color_per_seed"].set_params(
                        idx=s,
                        mean=df[df["conf"] == c]["tiam_gt_color_per_seed"]
                        .apply(lambda x: x[s])
                        .mean(),
                        n=n_prompt,
                    )
                if not self.multi_template_style_prompt:
                    self.scores[c]["count_order_binding"] = RunningMeanList(n_classes)
                    for i in range(n_classes):
                        self.scores[c]["count_order_binding"].set_params(
                            idx=i,
                            mean=df[df["conf"] == c]["count_order_binding"]
                            .apply(lambda x: x[i])
                            .mean(),
                            n=n_images,
                        )

    def get_scores(self):
        computes_scores = {}
        for c in self.confs_for_score:
            computes_scores[c] = {}
            computes_scores[c]["tiam"] = self.scores[c]["tiam"].get()
            if not self.multi_template_style_prompt:
                computes_scores[c]["n_class_detected"] = self.scores[c][
                    "n_class_detected"
                ].get()
                computes_scores[c]["count_order"] = self.scores[c]["count_order"].get()

            computes_scores[c]["tiam_per_seed"] = self.scores[c]["tiam_per_seed"].get()

            if self.seg:
                computes_scores[c]["tiam_gt_color"] = self.scores[c][
                    "tiam_gt_color"
                ].get()
                computes_scores[c]["tiam_gt_color_per_seed"] = self.scores[c][
                    "tiam_gt_color_per_seed"
                ].get()
                if not self.multi_template_style_prompt:
                    computes_scores[c]["count_order_binding"] = self.scores[c][
                        "count_order_binding"
                    ].get()

        # sort per conf
        return dict(sorted(computes_scores.items()))

    def save_scores(self, tiam_score, count_order, seed):
        if self.seg:
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
            if self.seg:
                self.scores[c]["tiam_gt_color"].update(tiam_score_binding[c])
                self.scores[c]["tiam_gt_color_per_seed"].update(
                    tiam_score_binding[c], seed
                )
                if self.scores[c]["count_order_binding"] is None:
                    self.scores[c]["count_order_binding"] = RunningMeanList(n_classes)
                self.scores[c]["count_order_binding"].update_all(count_order_binding[c])

    def get_tiam_score(self, attribute_binding: bool = False):
        """Return the TIAM score"""

        key = "tiam_gt_color" if self.seg and attribute_binding else "tiam"
        scores = self.get_scores()
        return {f"{c}": scores[c][key] for c in self.confs_for_score}

    def get_boxplot_seeds(self, attribute_binding: bool = False):
        """Boxplot of the TIAM score per seeds, axis x is the confidence threshold"""
        scores = self.get_scores()
        key = (
            "tiam_gt_color_per_seed"
            if self.seg and attribute_binding
            else "tiam_per_seed"
        )
        all_data = {c: list(scores[c][key].values()) for c in self.confs_for_score}

        fig, ax = plt.subplots(figsize=(len(self.confs_for_score) * 2, 4))
        sns.boxplot(
            data=all_data,
            ax=ax,
            color="skyblue",
            showfliers=True,
            fliersize=3,
            linewidth=1,
            showmeans=True,
            meanprops={
                "marker": "+",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },
        )
        ax.set_ylabel("TIAM")
        ax.set_xlabel("Confidence threshold")
        ax.grid(axis="y", alpha=0.5)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig, ax

    def get_object_occurence_by_position(
        self, attribute_binding: bool = False, conf: Optional[float] = None
    ):
        """Return the plot for the firts confidence threshold registered. When attribute_binding is True, the plot is for the color binding and the meaning is different. It is among the detected classes, the proportion of classes that are well binded to the color. When attribute_binding is False, the plot is the proportion of classes detected by position in the prompt"""
        if conf is None or conf not in self.confs_for_score:
            conf = self.confs_for_score[0]
        key = "count_order_binding" if self.seg and attribute_binding else "count_order"

        scores = self.get_scores()
        data = scores[conf][key]
        len(data)

        # Fill missing values with NaN to make all lists of equal length
        data = {key: data}
        df_melted = (
            DataFrame(data)
            .reset_index()
            .melt(id_vars="index", var_name="N Class in the prompt", value_name="value")
        )
        df_melted.rename(columns={"index": "position"}, inplace=True)
        fig, ax = plt.subplots(figsize=(len(data) * 2, 4))
        sns.barplot(
            data=df_melted,
            x="N Class in the prompt",
            y="value",
            hue="position",
            ax=ax,
            palette="viridis",
        )
        if self.seg and attribute_binding:
            ax.set_ylabel("Binding success rate among detected object")
        else:
            ax.set_ylabel("Proportion of objects in images")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.5)
        return fig, ax

    def load_data_one_file(self, path_to_json_file):
        """Load the data from the json file

        Args:
            path_to_json_file (str): path to the json file

        Returns:
            DataFrame: dataframe with the scores
        """
        path_to_json_file = Path(path_to_json_file)
        if not path_to_json_file.is_file():
            raise ValueError("path_to_json_file must be a file")
        data = pd.read_json(path_to_json_file)
        self.reset()
        self.set_values_from_df(data)

    def load_data_from_multiple_files(
        self,
        path_to_json_files,
        save_dir: Optional[str] = None,
        multi_template_style_prompt: bool = False,
        files=None,
    ):
        """Load the data from the json files and save the results in a json file

        Args:
            path_to_json_files (str): path to the json files
            save_dir (Optional[str], optional): directory to save the results. Defaults to None.
        """
        self.multi_template_style_prompt = multi_template_style_prompt
        path_to_json_files = Path(path_to_json_files)
        if not path_to_json_files.is_dir():
            raise ValueError("path_to_json_files must be a directory")
        # get all the json files
        if files is None:
            files = list(path_to_json_files.glob("*.json"))
        # load and concat all the json files
        all_scores = []
        for f in files:
            try:
                all_scores.append(pd.read_json(f))
            except Exception as e:
                print(f"Error with {f}: {e}")
        all_scores = pd.concat(all_scores, ignore_index=True)
        if save_dir is not None:
            save_dir = Path(save_dir)
            if save_dir.is_dir():
                save_dir = save_dir / "tiam_score.json"
            if not save_dir.suffix == ".json":
                save_dir = save_dir.with_suffix(".json")
            all_scores.to_json(save_dir, indent=4)
        self.reset()
        self.set_values_from_df(all_scores)

    def get_current_tiam(self, conf: Optional[float] = None):
        """Return the current TIAM score of the first confidence threshold registered in the class

        Args:
            conf (Optional[float], optional): _description_. Defaults to None.

        Returns:
            float: tiam score
        """
        if conf is None or conf not in self.confs_for_score:
            conf = self.confs_for_score[0]
        return self.scores[conf]["tiam"].get(), conf

    def get_seeds_scores(
        self,
    ):
        """Return the TIAM score per seeds"""
        score = self.get_scores()
        tiam_per_seed = {}
        for c, v in score.items():
            tiam_per_seed_np = np.array(list(v["tiam_per_seed"].values()))
            tiam_per_seed[f"{c}"] = tiam_per_seed_np
            if self.seg:
                tiam_per_seed_np_binding = np.array(
                    list(v["tiam_gt_color_per_seed"].values())
                )
                tiam_per_seed[f"{c}" + "_binding"] = tiam_per_seed_np_binding
        return tiam_per_seed

    def log_scores(self):
        """Create 3 dataframes with the scores

        1. TIAM, n_class, TIAM gt colors
        2. TIAM for the seeds with min, q1, median, q3, max, mean, var
        3. Count order, count order binding
        """
        score = self.get_scores()

        index = self.confs_for_score.copy()
        if self.seg:
            index_w_binding = []
            for c in self.confs_for_score:
                index_w_binding.append(c)
                index_w_binding.append(f"{c}" + "_binding")
        else:
            index_w_binding = index

        if self.multi_template_style_prompt:
            tiam_df = {"tiam": []}
        else:
            tiam_df = {"tiam": [], "n_class_detected": []}
        if self.seg:
            tiam_df["tiam_gt_color"] = []

        seeds_df = {
            "min": [],
            "q1": [],
            "median": [],
            "q3": [],
            "max": [],
            "mean": [],
            "var": [],
        }
        count_order_df = []
        for c, v in score.items():
            # tiam
            tiam_df["tiam"].append(v["tiam"])
            if not self.multi_template_style_prompt:
                tiam_df["n_class_detected"].append(v["n_class_detected"])

            # seeds
            tiam_per_seed_np = np.array(list(v["tiam_per_seed"].values()))

            seeds_df["min"].append(np.min(tiam_per_seed_np))
            seeds_df["q1"].append(np.percentile(tiam_per_seed_np, 25))
            seeds_df["median"].append(np.median(tiam_per_seed_np))
            seeds_df["q3"].append(np.percentile(tiam_per_seed_np, 75))
            seeds_df["max"].append(np.max(tiam_per_seed_np))
            seeds_df["mean"].append(np.mean(tiam_per_seed_np))
            seeds_df["var"].append(np.var(tiam_per_seed_np))
            # colors
            if self.seg:
                tiam_df["tiam_gt_color"].append(v["tiam_gt_color"])
                tiam_per_seed_np_binding = np.array(
                    list(v["tiam_gt_color_per_seed"].values())
                )
                seeds_df["min"].append(np.min(tiam_per_seed_np_binding))
                seeds_df["q1"].append(np.percentile(tiam_per_seed_np_binding, 25))
                seeds_df["median"].append(np.median(tiam_per_seed_np_binding))
                seeds_df["q3"].append(np.percentile(tiam_per_seed_np_binding, 75))
                seeds_df["max"].append(np.max(tiam_per_seed_np_binding))
                seeds_df["mean"].append(np.mean(tiam_per_seed_np_binding))
                seeds_df["var"].append(np.var(tiam_per_seed_np_binding))
            if not self.multi_template_style_prompt:
                # count order
                count_order_df.append(v["count_order"])
                if self.seg:
                    count_order_df.append(v["count_order_binding"])
        tiam_df = DataFrame(tiam_df, index=index)
        seeds_df = DataFrame(seeds_df, index=index_w_binding)
        if not self.multi_template_style_prompt:
            count_order_df = DataFrame(count_order_df, index=index_w_binding)
        else:
            count_order_df = None
        return tiam_df, seeds_df, count_order_df


class TIAM_per_prompt(TIAMScore):
    # TODO: peut être faire évoluer vers une config ou on peu passer des images avec des prompts différents en input

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
        super().__init__(confs_for_score, min_conf_threshold)
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
        # self.open_vocabulary = open_vocabulary

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
            # print(f"{batch.shape=}; {seeds=};")
            # print(f"dtype={batch.dtype}; device={batch.device};")
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
                # print(f"{classes2detect=};")
                # print(color_classes)
                # print(f"{detected_classes=}; {detected_colors=}; {percentages_colors=}; {percentages_colors=}; ")
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

    # TODO: metrics if we conditionned the model with boxes
    def iou_score():
        """IOU conditionning boxe and generate boxe"""
        pass

    def mean_average_precision():
        """mAP . The mAP compares the ground-truth bounding box to the detected box and returns a score. The higher the score, the more accurate the model is in its detections.
        mAP @ X with X being the IOU threshold

        """
        pass

    def to(self, device):
        self.device = device
        self.model.to(self.device)
