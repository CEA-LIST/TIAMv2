import typer
from .prompts_dataset.utils import create_dataset as pipeline_dataset
from .tiam import load_data_from_multiple_files, compute_tiam_score
from pathlib import Path
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def create_dataset(
    config_path: Annotated[Path, typer.Option()],
    save_path: Annotated[Path, typer.Option()],
):

    dataset = pipeline_dataset(config_path)
    dataset.save_to_disk(save_path)


@app.command()
def score(
    save_dir: Annotated[Path, typer.Option()],
    image_dir: Annotated[Path, typer.Option()],
    dataset_path_or_url: Annotated[Path, typer.Option()],
    model_path_or_url: str = "yolov8x-seg.pt",
    batch_size: int = 32,
):
    compute_tiam_score(
        save_dir=save_dir,
        dataset_path=dataset_path_or_url,
        image_dir=image_dir,
        model_path_or_url=model_path_or_url,
        batch_size=batch_size,
    )


@app.command()
def concatenate_scores():
    pass
