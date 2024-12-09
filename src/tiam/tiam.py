import ast
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from rich.progress import track

from .tiam_per_prompt import TIAM_per_prompt
from .utils import get_images, get_images_from_json

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_FORMAT = [".png", ".jpg", ".jpeg"]


def get_adequate_processing(n_images_equal, seeds_equal, colored):
    processing = {
        "n_prompt": "count",
        "tiam": "mean",
        "n_class_detected": "mean",
        "count_order": aggregate_dicts,
    }
    if n_images_equal and seeds_equal:
        processing["tiam_per_seed"] = aggregate_dicts

    if colored:
        processing["tiam_gt_color"] = "mean"
        processing["count_order_binding"] = aggregate_dicts
        if n_images_equal and seeds_equal:
            processing["tiam_gt_color_per_seed"] = aggregate_dicts
    return processing


def aggregate_dicts(series):
    df_dict = pd.DataFrame(series.tolist())
    # Calculate mean for each seed
    means = df_dict.mean()
    return means.to_dict()


def load_data_from_multiple_files(
    path_to_json_files,
    save_dir: Optional[str] = None,
    files=None,
    precision: Optional[int]=3,
):
    """Load the data from the json files and save the results in a json file and markdown files

    Args:
        path_to_json_files (str): path to the json files
        save_dir (Optional[str], optional): directory to save the results. Defaults to None.
        precision (Optional[int]): round the final score (displayed) to 10^(-precision). Do not round if precision <=0. Default: precison = 3 (round to 1e-3)
    """

    path_to_json_files = Path(path_to_json_files)
    if not path_to_json_files.is_dir():
        raise ValueError("path_to_json_files must be a directory")
    if files is None:
        files = list(path_to_json_files.glob("*.json"))
    type_data = {}

    for f in files:
        try:
            colored = False
            df_temp = pd.read_json(f)
            if "tiam_gt_color" in df_temp.columns:
                colored = True
            n_entities = len(df_temp.loc[0, "count_order"])
            if type_data.get((colored, n_entities)) is None:
                type_data[(colored, n_entities)] = [df_temp]
            else:
                type_data[(colored, n_entities)].append(df_temp)

            # check si color ou non, il faudra calcuelr le score uniquement pour ceux ou il y a a couleur
            # concatenate the data
            # all_scores.append(pd.read_json(f))
        except Exception as e:
            print(f"Error with {f}: {e}")
    # create conctenate dataframe per type of prompt
    df_resume = {"color": [], "wo_color": []}

    for (colored, n_entities), dfs in type_data.items():
        # check if df same number of seeds
        # if not low execution [only tiam score]
        n_images = None
        available_seeds = None
        available_confidence = None
        seeds_equal = True
        n_images_equal = True
        for df in dfs:
            # check if tiam per seed is available
            if available_confidence is None:
                available_confidence = df["conf"].unique()
            else:
                # keep common confidence
                available_confidence = list(
                    set(available_confidence).intersection(df["conf"].unique())
                )

            if "tiam_per_seed" not in df.columns:
                seeds_equal = False
                n_images_equal = False
                break

            seeds = df.loc[0, "tiam_per_seed"].keys()
            if n_images is None:
                n_images = len(seeds)
            if available_seeds is None:
                available_seeds = sorted([int(s.split("_")[-1]) for s in seeds])
            if n_images != len(seeds):
                logger.warning(
                    "The number of images per prompt is not consistent. The score will be computed without considering the seeds."
                )
                n_images_equal = False
                break
            if available_seeds != sorted([int(s.split("_")[-1]) for s in seeds]):
                logger.warning(
                    "The seeds are not consistent. The score will be computed without considering the seeds."
                )
                seeds_equal = False
                break
        if n_images_equal and seeds_equal:
            df_concat = pd.concat(dfs, ignore_index=True)
        else:
            # keep colums
            keep_columns = [
                "prompt",
                "conf",
                "n_class_detected",
                "tiam",
                "count_order",
            ]
            if colored:
                keep_columns.append("tiam_gt_color")

            dfs = [df[keep_columns] for df in dfs]
            df_concat = pd.concat(dfs, ignore_index=True)
        df_concat["n_prompt"] = 1
        # filter on common confidence
        df_concat = df_concat[df_concat["conf"].isin(available_confidence)]
        df_concat = (
            df_concat.groupby("conf")
            .agg(get_adequate_processing(n_images_equal, seeds_equal, colored))
            .reset_index()
        )
        # average number of class detected per image
        df_concat["n_class_detected"] = df_concat["n_class_detected"] * n_entities
        # round the scores
        if precision>0:
            for i, r in df_concat.iterrows():
                df_concat.at[i,'tiam'] = round(r['tiam'], precision)
                df_concat.at[i,'count_order'] = {k : round(v,precision) for k,v in r['count_order'].items()}
                df_concat.at[i,'tiam_per_seed'] = {k : round(v,precision) for k,v in r['tiam_per_seed'].items()}
                if colored:
                    df_concat.at[i,'count_order_binding'] = {k : round(v,precision) for k,v in r['count_order_binding'].items()}
                    df_concat.at[i,'tiam_gt_color_per_seed'] = {k : round(v,precision) for k,v in r['tiam_gt_color_per_seed'].items()}

        
        # print the score for the number of entities and colors
        print(f"Score for {'colored ' if colored else ''}{n_entities} entities")
        print(df_concat.to_markdown(floatfmt="."+str(precision)+"f"))
        # save average score per data_type
        df_concat.to_json(
            save_dir
            / f"tiam_score_{n_entities}_{'colored_' if colored else ''}entities.json",
            indent=4,
        )
        if colored:
            df_resume["color"].append((df_concat, n_entities))
        else:
            df_resume["wo_color"].append((df_concat, n_entities))

    # Compute TIAM score without regards on the number of entities
    # weighted compute on the number of prompt

    # rechecker que les mêmes seeds et les mêmes confidences utilisés

    if len(df_resume["color"]) > 1:
        per_seed_available = True
        available_confidence = None
        available_seeds = None
        for df, n_entities in df_resume["color"]:
            if available_confidence is None:
                available_confidence = df["conf"].unique()
            else:
                # keep common confidence
                available_confidence = list(
                    set(available_confidence).intersection(df["conf"].unique())
                )
            if "tiam_per_seed" not in df.columns:
                per_seed_available = False
            else:
                if available_seeds is None:
                    available_seeds = sorted(
                        [
                            int(s.split("_")[-1])
                            for s in df.loc[0, "tiam_per_seed"].keys()
                        ]
                    )
                if available_seeds != sorted(
                    [int(s.split("_")[-1]) for s in df.loc[0, "tiam_per_seed"].keys()]
                ):
                    per_seed_available = False

        df_weighted = calculate_weighted_metrics(
            df, color=True, per_seed=per_seed_available
        )
        df_weighted.to_json(save_dir / "prompt_weighted_tiam_score_color.json")

    if len(df_resume["wo_color"]) > 1:
        available_confidence = None
        available_seeds = None
        per_seed_available = True
        for df, n_entities in df_resume["wo_color"]:
            if available_confidence is None:
                available_confidence = df["conf"].unique()
            else:
                # keep common confidence
                available_confidence = list(
                    set(available_confidence).intersection(df["conf"].unique())
                )
            if "tiam_per_seed" not in df.columns:
                per_seed_available = False
            else:
                if available_seeds is None:
                    available_seeds = sorted(
                        [
                            int(s.split("_")[-1])
                            for s in df.loc[0, "tiam_per_seed"].keys()
                        ]
                    )
                if available_seeds != sorted(
                    [int(s.split("_")[-1]) for s in df.loc[0, "tiam_per_seed"].keys()]
                ):
                    per_seed_available = False

            df_weighted = calculate_weighted_metrics(
                df, color=False, per_seed=per_seed_available
            )
            df_weighted.to_json(
                save_dir / f"prompt_weighted_tiam_score_{n_entities}_entities.json"
            )


def calculate_weighted_metrics(df, color=False, per_seed=True):
    """
    Calculate weighted metrics from DataFrame with configurable color and per-seed computations

    Args:
        df: pandas DataFrame with columns [tiam, n_prompt, tiam_per_seed]
            optional: [tiam_gt_color, tiam_gt_color_per_seed]
        color: bool, whether to compute color-related metrics
        per_seed: bool, whether to compute per-seed metrics
    Returns:
        DataFrame with weighted means per confidence
    """

    # Initialize weighted dict with base metrics
    weighted = {
        "weighted_tiam": df["tiam"] * df["n_prompt"],
        "n_prompt": df["n_prompt"],
    }

    # Handle color metrics if enabled
    if color:
        weighted["weighted_tiam_gt_color"] = df["tiam_gt_color"] * df["n_prompt"]

    # Handle per-seed metrics if enabled
    if per_seed:
        if isinstance(df["tiam_per_seed"].iloc[0], str):
            df["tiam_per_seed"] = df["tiam_per_seed"].apply(ast.literal_eval)
        weighted["weighted_tiam_per_seed"] = df.apply(
            lambda x: {
                seed: score * x["n_prompt"]
                for seed, score in x["tiam_per_seed"].items()
            },
            axis=1,
        )

        if color and "tiam_gt_color_per_seed" in df.columns:
            if isinstance(df["tiam_gt_color_per_seed"].iloc[0], str):
                df["tiam_gt_color_per_seed"] = df["tiam_gt_color_per_seed"].apply(
                    ast.literal_eval
                )
            weighted["weighted_tiam_gt_color_per_seed"] = df.apply(
                lambda x: {
                    seed: score * x["n_prompt"]
                    for seed, score in x["tiam_gt_color_per_seed"].items()
                },
                axis=1,
            )

    # Group and calculate metrics
    grouped = (
        pd.DataFrame(weighted)
        .groupby(df["conf"])
        .agg({k: "sum" for k in weighted.keys()})
    )

    # Prepare result dictionary
    result = {"weighted_mean_tiam": grouped["weighted_tiam"] / grouped["n_prompt"]}

    if color:
        result["weighted_mean_tiam_gt_color"] = (
            grouped["weighted_tiam_gt_color"] / grouped["n_prompt"]
        )

    if per_seed:
        result["weighted_mean_tiam_per_seed"] = grouped.apply(
            lambda x: {
                seed: score / x["n_prompt"]
                for seed, score in x["weighted_tiam_per_seed"].items()
            },
            axis=1,
        )
        if color and "weighted_tiam_gt_color_per_seed" in grouped:
            result["weighted_mean_tiam_gt_color_per_seed"] = grouped.apply(
                lambda x: {
                    seed: score / x["n_prompt"]
                    for seed, score in x["weighted_tiam_gt_color_per_seed"].items()
                },
                axis=1,
            )

    return pd.DataFrame(result)


def params_to_detect(row):
    entities = list(row["labels_params"].values())

    if "adjs_params" in row and len(row["adjs_params"]) > 0:
        adjs = list(row["adj_apply_on"].keys())
        color_classes = {}
        for adj in adjs:
            color_classes[row["labels_params"][row["adj_apply_on"][adj]]] = row[
                "adjs_params"
            ][adj]
    else:
        color_classes = None
    return entities, color_classes


def clean_data(row):
    keys = ["labels_params"]
    if "adjs_params" in row:
        keys.append("adjs_params")
        keys.append("adj_apply_on")
    for key in keys:
        row[key] = {k: v for k, v in row[key].items() if v}
    return row


def validate_row(row, entity_cols, adj_cols):
    # Check for gaps in entities
    has_entity = False
    for i, col in enumerate(entity_cols):
        if pd.notna(row[col]):
            has_entity = True
        elif (
            has_entity
            and i < len(entity_cols) - 1
            and pd.notna(row[entity_cols[i + 1]])
        ):
            raise ValueError(f"Gap detected in entities: {dict(row)}")

    # Check adjective correspondence
    for i, adj_col in enumerate(adj_cols):
        if pd.notna(row[adj_col]):
            if i >= len(entity_cols) or pd.isna(row[entity_cols[i]]):
                raise ValueError(f"Adjective without corresponding entity: {dict(row)}")

    return has_entity


def csv2dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Validate columns
    entity_cols = sorted([col for col in df.columns if col.startswith("entity")])
    adj_cols = sorted([col for col in df.columns if col.startswith("adj")])

    # Check for sequential numbering
    def validate_sequential_cols(cols):
        numbers = [
            int(re.findall(r"\d+", col)[0]) for col in cols if re.findall(r"\d+", col)
        ]
        if not numbers:
            return True
        return sorted(numbers) == list(range(1, len(numbers) + 1))

    if not validate_sequential_cols(entity_cols) or not validate_sequential_cols(
        adj_cols
    ):
        raise ValueError(
            "Columns must be sequentially numbered (entity1, entity2, etc.)"
        )

    # Filter and validate rows
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            if validate_row(row, entity_cols, adj_cols):
                valid_rows.append(row)
        except ValueError as e:
            raise ValueError(f"Error in row {idx}: {str(e)}")

    if not valid_rows:
        raise ValueError("No valid rows found in CSV")

    processed_data = []

    for row in valid_rows:
        # Extract entities
        entities = {
            f"entity{i+1}": str(row[col])
            for i, col in enumerate(entity_cols)
            if pd.notna(row[col])
        }

        # Extract adjectives
        adjs = {
            f"adj{i+1}": str(row[col])
            for i, col in enumerate(adj_cols)
            if pd.notna(row[col])
        }

        # Create adj_apply_on mapping with empty values if no adjective exists
        adj_apply_on = {}
        for i in range(len(entities)):
            adj_key = f"adj{i+1}"
            if adj_key in adjs and adjs[adj_key]:
                adj_apply_on[adj_key] = f"entity{i+1}"
            else:
                adj_apply_on[adj_key] = ""

        # Fill empty adjectives for remaining entities
        for i in range(len(entities)):
            adj_key = f"adj{i+1}"
            if adj_key not in adjs:
                adjs[adj_key] = ""

        processed_row = {
            "prompt": row["prompt"],
            "labels_params": entities,
            "adjs_params": adjs,
            "adj_apply_on": adj_apply_on,
        }

        processed_data.append(processed_row)
    return Dataset.from_list(processed_data)


def compute_tiam_score(
    save_dir,
    dataset_path,
    image_dir,
    model_path="yolov8x-seg.pt",
    batch_size=32,
    detect_only=False,
):

    images_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir_per_prompt = save_dir / "tiam_score_per_prompt"
    dataset_path = Path(dataset_path)

    json_files = None
    if images_dir.is_dir():
        all_file_names = [
            f for f in images_dir.iterdir() if f.suffix in ALLOWED_IMAGE_FORMAT
        ]
        tar = None
    elif images_dir.is_file() and images_dir.suffix == ".tar":
        with tarfile.open(images_dir, "r") as tar:
            tar = tarfile.open(images_dir, "r")
            all_file_names = tar.getnames()
    elif images_dir.suffix == ".json":
        with open(images_dir, "r") as f:
            json_files = json.load(f)
        # process the data

        tar = None

    if dataset_path.is_dir():
        dataset = load_from_disk(dataset_path)  # load from disk
    elif dataset_path.suffix == ".csv":
        dataset = csv2dataset(dataset_path)
    else:
        dataset = load_dataset(str(dataset_path))["train"]  # load from huggingface

    pipeline = TIAM_per_prompt(
        save_dir=save_dir_per_prompt,
        model_path=model_path,
        batch_size=batch_size,
        confs_for_score=[0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )

    unavailable_prompts = []
    for row in track(iter(dataset), total=len(dataset)):
        row = clean_data(row)

        prompt = row["prompt"]

        if json_files is not None:
            images, seeds = get_images_from_json(json_data=json_files, prompt=prompt)
        else:
            images, seeds = get_images(
                prompt=prompt, all_file_names=all_file_names, tar=tar
            )
        if images is None:
            unavailable_prompts.append(prompt)
            continue
        entities, classes_per_color = params_to_detect(row)
        pipeline.predict(
            images=images,
            seeds_used=seeds,
            classes=entities,
            color_classes=classes_per_color,
            prompt=prompt,
        )
    if len(unavailable_prompts) > 0:
        logger.info(
            f"{len(unavailable_prompts)} prompts were not found in the images folder. List of prompts :{unavailable_prompts}"
        )
    if not detect_only:
        load_data_from_multiple_files(
            path_to_json_files=save_dir_per_prompt,
            save_dir=save_dir,
        )
