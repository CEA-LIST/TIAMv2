from pathlib import Path

import typer
from typing_extensions import Annotated

from .prompts_dataset.utils import create_dataset as pipeline_dataset
from .tiam import compute_tiam_score, load_data_from_multiple_files

app = typer.Typer(
    name="tiam",
    help="Tool for computing the TIAM score for images and datasets.",
)


@app.command(
    help="Create a dataset based on the provided configuration file and save it to the specified path."
)
def create_dataset(
    config_path: Annotated[
        Path,
        typer.Option(help="Path to the configuration file for creating the dataset"),
    ],
    save_path: Annotated[
        Path, typer.Option(help="Path where the created dataset will be saved")
    ],
):
    dataset = pipeline_dataset(config_path)
    dataset.save_to_disk(save_path)


# todo prompt: seed : path image
# todo prompt: [path image]
# rodo: prompt cqv


@app.command(
    help="Compute the TIAM score for the given images and dataset using the specified model."
)
def score(
    save_dir: Annotated[
        Path, typer.Option(help="Directory where the results will be saved")
    ],
    image_dir: Annotated[
        Path, typer.Option(help="Directory containing the images to be scored")
    ] = ...,  # todo
    dataset_path_or_url: Annotated[
        Path, typer.Option(help="Path or URL to the dataset")
    ] = ...,  # todo
    model_path_or_url: Annotated[
        str, typer.Option(help="Path or URL to the model")
    ] = "yolov8x-seg.pt",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 32,
    detect_only: Annotated[
        bool,
        typer.Option(help="Flag to indicate if only detection should be performed"),
    ] = False,
):
    compute_tiam_score(
        save_dir=save_dir,
        dataset_path=dataset_path_or_url,
        image_dir=image_dir,
        model_path_or_url=model_path_or_url,
        batch_size=batch_size,
        detect_only=detect_only,
    )


@app.command(
    help="Load data from multiple JSON files, display it and save the results to the specified directory."
)
def load_score(
    save_dir: Annotated[
        Path, typer.Option(help="Directory where the results will be saved")
    ],
    path_to_json_files: Annotated[
        Path,
        typer.Option(help="Path to the directory containing JSON files to be loaded"),
    ],
):
    load_data_from_multiple_files(
        save_dir=save_dir,
        path_to_json_files=path_to_json_files,
    )
