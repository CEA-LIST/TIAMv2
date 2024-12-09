from pathlib import Path

import typer
from typing_extensions import Annotated

from .prompts_dataset.utils import create_dataset as pipeline_dataset
from .tiam import compute_tiam_score, csv2dataset, load_data_from_multiple_files

from datasets import load_dataset

app = typer.Typer(
    name="tiam",
    help="Tool for computing the TIAM score for images and datasets.",
)


@app.command(
    help="Create a dataset based on the provided configuration file and save it to the specified path."
)
def create_dataset(
    config_file: Annotated[
        Path,
        typer.Option(help="configuration file (xxx.yaml) to create the prompt dataset"),
    ],
    save_dir: Annotated[
        Path, typer.Option(help="directory to save the created prompt dataset")
    ],
):
    dataset = pipeline_dataset(config_file)
    dataset.save_to_disk(save_dir)

    # save explicit prompt in a text file
    with open(Path(save_dir).joinpath("prompts.txt"), "w") as f:
        for line in dataset["prompt"]:
            f.write(f"{line}\n")
    # save the corresponding CSV file
    with open(Path(save_dir).joinpath("prompts.csv"), "w") as f:
        first_line = "prompt,"
        for e in dataset["labels_params"][0].keys():
            first_line = first_line + e + ","
        for a in dataset["adjs_params"][0].keys():
            first_line = first_line + a + ","
        f.write(f"{first_line[:-1]}\n")
        for row in iter(dataset):

            line = f'"{row["prompt"]}",'
            line += f'{",".join(v for v in row["labels_params"].values())}'
            line += f'{",".join(v for v in row["adjs_params"].values())}'
            f.write(f"{line}\n")  # FIXME on garde dernière virgule?? sinon {line[:-1]}


@app.command(
    help="Compute the TIAM score for the given images and dataset using the specified model."
)
def score(
    save_dir: Annotated[
        Path, typer.Option(help="Directory where the results will be saved")
    ],
    image_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing the synthetic images or json files with prompts and path (and seeds)"
        ),
    ] = Path(
        "/dev/null"
    ),  # impossible default path
    dataset_path: Annotated[
        Path, typer.Option(help="Path or URL to the dataset or CSV file")
    ] = Path(
        "/dev/null"
    ),  # impossible default path
    model_detect_segment: Annotated[
        str,
        typer.Option(
            help="Path or URL to the model used for object detection and segmentation"
        ),
    ] = "yolov8x-seg.pt",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 32,
    detect_only: Annotated[
        bool,
        typer.Option(help="Flag to indicate if only detection should be performed"),
    ] = False,
):
    if image_dir == Path("/dev/null"):
        image_dir = Path(save_dir).joinpath("images/")
    if dataset_path == Path("/dev/null"):
        dataset_path = Path(save_dir).joinpath("dataset/")

    compute_tiam_score(
        save_dir=save_dir,
        dataset_path=dataset_path,
        image_dir=image_dir,
        model_path=model_detect_segment,
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
    disp_precision: Annotated[
        int,
        typer.Option(help="Round the displayed results to 1e-disp_precision. Do not round if disp_precision<=0. Default = 3"),
    ] = 3,
):
    load_data_from_multiple_files(
        save_dir=save_dir,
        path_to_json_files=path_to_json_files,
        precision=disp_precision,
    )


@app.command(help="CSV prompts to Dataset")
def csv_to_dataset(
    csv_path: Annotated[
        Path,
        typer.Option(help="Path to the CSV file containing the prompts"),
    ],
    save_dir: Annotated[
        Path, typer.Option(help="Path where the created dataset will be saved")
    ] = None,
):
    dataset = csv2dataset(csv_path)
    print(dataset)
    for row in iter(dataset):
        print(row)

@app.command(help="download prompt dataset from Huggingface")
def get_hub_dataset(
    dataset_name: Annotated[
        str,
        typer.Option(help="name of the prompt dataset on Huggingface"),
    ],
    save_dir: Annotated[
        Path, typer.Option(help="folder where the dataset will be saved")
    ] = None,
):
    if dataset_name in ['Paulgrim/2_entities',
                        'Paulgrim/3_entities',
                        'Paulgrim/2_colored_entities',
                        'Paulgrim/3_colored_entities']:
        dataset = load_dataset(dataset_name)
        dataset['dataset']=dataset.pop('train')
        dataset.save_to_disk(save_dir)

        with open(save_dir.joinpath("prompts.txt"), "w") as f:
            for line in dataset['dataset']["prompt"]:
                f.write(f"{line}\n")
    else:
        print(f'!!! unknown prompt dataset ({dataset_name})')
