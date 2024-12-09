# [WACV 2024] TIAM - A Metric for Evaluating Alignment in Text-to-Image Generation

TIAM is a metric to estimate the performance of a visual language model (VLM) in terms of alignment of the prompt with the generated images. It estimates to which extent the entites (objects) and their attributes specified in the prompt are actually visible in the synthetic images. The work was [published at WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Grimal_TIAM_-_A_Metric_for_Evaluating_Alignment_in_Text-to-Image_Generation_WACV_2024_paper.html) and in a previous preprint on [![arxiv](https://img.shields.io/badge/arXiv-2307.05134-b31b1b.svg)](https://arxiv.org/abs/2307.05134).

This repo is a refactoring that makes it easier to use, while the the [original code](https://github.com/grimalPaul/TIAM) focused on reproducind the experiments of the paper. If you find this program useful for your research, please [cite it](#citation)


# Install

Install with [uv](https://docs.astral.sh/uv//), that itself can be [installed with one line](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv build
source .venv/bin/activate
tiam --help
```

Future works will allow to install with `pip`

# Usage

To evaluate a VLM the general workflow consists to:

* create a dataset of prompts with TIAM or use [an existing one](doc/benchmarking.md)
* generate several images per prompt with a VLM 
* link the prompts to the images for TIAM
* evaluate with TIAM

You can use any model to generate images. We provide the tools to get the prompts in ASCII, CSV or [arrow](https://huggingface.co/docs/datasets/about_arrow) format.

To create a dataset, you can use one of the configuration files provided in `src/tiam/data/`. As a toy example, let create a dataset of prompts with 3 objects and 2 colors with [this configuration file](src/tiam/data/sample.yaml) and save it in `SAVE_DIR=tests/data/sample/dataset`

```
tiam create-dataset --config-file src/tiam/data/sample.yaml --save-dir tests/data/sample/dataset
```
The prompts are generated in 3 formats, including a human readable `SAVE_DIR/prompts.txt`. Let generate 4 images per prompt and save them in `tests/data/sample/images`. You can use any VLM of your choice by providing the prompts in `SAVE_DIR/prompts.txt` but the name of the image file must be `the_prompt_with_underscres_N.png`, where `N=0,1,2,3` (for 4 images per prompt).

Images generated for this toy example with Stable Diffusion 1.4 are in [tests/data/sample/images](tests/data/sample/images). Now your working directory looks like:

```bash
<SAVE_DIR>
 ├── images/
     ├── the_first_prompt_with_objects_and_attributes_0.png
     ├── the_first_prompt_with_objects_and_attributes_1.png
     ├── the_first_prompt_with_objects_and_attributes_2.png
     ├── the_first_prompt_with_objects_and_attributes_3.png
     └── the_second_prompt_with_objects_and_attributes.0.png
         (...)
         (...)
     └── the_last_prompt_with_objects_and_attributes_3.png
 └── dataset/
    ├── prompts.csv
    ├── prompts.txt
    └── [Dataset] = 2 'json' files and one or several 'arrow' file(s)
```

And get the perfomances with:

```
tiam score --save-dir tests/data/sample/ 
```
It uses the default directories `SAVE_DIR/images` and `SAVE_DIR/dataset` but these last can be changed. As explained below, it exists several methods to configure tiam, both to create the dataset and compute the score.

## Creating a Dataset

### From a configuration file
Create a dataset based on the provided configuration file and save it to the specified path.

```bash
tiam create-dataset --config-file <config.yaml> --save-dir <SAVE_DIR>/dataset
```
With:
* `--config-file`: configuration file similar to those in `src/tiam/data/`
* `--save-dir`: folder to save the created prompt dataset.

### From Huggingface
Some datasets are available on the hub, in particular [for benchmarking](doc/benchmarking.md)


```bash
tiam get-hub-dataset --dataset-name "Paulgrim/2_entities" --save-dir <SAVE_DIR>
tiam get-hub-dataset --dataset-name "Paulgrim/2_colored_entities" --save-dir <SAVE_DIR>
tiam get-hub-dataset --dataset-name "Paulgrim/3_entities" --save-dir <SAVE_DIR>
tiam get-hub-dataset --dataset-name "Paulgrim/3_colored_entities" --save-dir <SAVE_DIR>
```

## Compute TIAM Score
### Linking images to prompt
Several methods can be used to let TIAM know which synthetic images correspond to each prompt of the dataset.

#### Unique repository and constrained image filename
You can put all the synthetic images in a unique directory and use `--image-dir` to give it ti TIAM. However, the name of the image files must have a strict pattern namely the prompt (low case, with spaces replaced by underscores) followed by `_n.ext` where `n` is a number between 0 and N-1 (il you generate N image per prompt) and `ext` is the file extension (jpg,png...)

<details>
<summary>Example of directory content</summary>

```bash
a_photo_of_a_bicycle_and_a_motorcycle_0.png
a_photo_of_a_bicycle_and_a_motorcycle_1.png
a_photo_of_a_bicycle_and_a_motorcycle_2.png
a_photo_of_a_bicycle_and_a_motorcycle_3.png
a_photo_of_a_bicycle_and_a_motorcycle_4.png
a_photo_of_a_bicycle_and_a_motorcycle_5.png
...
a_photo_of_a_bicycle_and_a_motorcycle_13.png
a_photo_of_a_bicycle_and_a_motorcycle_14.png
a_photo_of_a_bicycle_and_a_motorcycle_15.png
a_photo_of_a_bicycle_and_a_bench_0.png
a_photo_of_a_bicycle_and_a_bench_1.png
...
```

</details>

You can use symbolic link named as expected by TIAM with `ln -s ...`

#### Using a JSON file
You can specify the exact image path for all the prompt in a JSON file with arrays. The name of the array is the prompt (str) and the values are the path to the images (str).

<details>
<summary>Example of JSON file to link prompt and images</summary>

```
{
    "a photo of a giraffe and a bear": [
        "path/to/images/prompt1/myname.png",
        "path/to/images/prompt1/myname_345.png",
        "path/to/images/smth12.png",
        "path/to/images/prompt1/myname.jpg"
    ],
    "a photo of a zebra and a motorcycle": [
        "path/to/images/other_prompt/smthg.png",
        "path/to/images/elsewhere/strangenam.jpg",
        "path/to/aa.png",
        "path/to/images/bb.png"
    ]
}
```

</details>

### Estimate perfomances

Compute the TIAM score for the given images and dataset using the specified model.

```bash
tiam score --save-dir <SAVE_DIR> --image-dir <IMAGE_DIR> --dataset-path <DATASET_PATH_OR_URL> --model-detect-segment <MODEL_PATH_OR_URL> --batch-size <BATCH_SIZE> --detect-only
```

Where:

* `--save-dir`: Directory where the results will be saved.
* `--image-dir`: Directory containing the images to be scored. It can also be a tar file that contains the images directly. Default: `<SAVE_DIR>/images`
* `--dataset-path`: Path or URL to the (arrow) dataset directory. Not necessary if it is in `<SAVE_DIR>/dataset` (could be a symbolic link)
* `--model-detect-segment`: Path or URL to the model (default: "yolov8x-seg.pt"). TIAM downloads the default Yolo detection/segmentation model at first usage
* `--batch-size`: Batch size for processing (default: 32).
* `--detect-only`: Flag to indicate if only detection should be performed. In that case, a list of JSON file is created and can further be agregated with `tiam load-score` (see below)

## Load Score Data

Load data from multiple JSON files, display it and save the results to the specified directory.

```bash
tiam load-score --save-dir <SAVE_DIR> --path-to-json-files <PATH_TO_JSON_FILES>
```

### Options

* `--save-dir`: Directory where the results will be saved.
* `--path-to-json-files`: Path to the directory containing JSON files to be loaded.

# Usage Example

Examples command from the tests

```bash
tiam create-dataset --config-file src/tiam/data/2_colored_entities.yaml --save-dir tests/data/2_colored_entities_dataset

tiam score --save-dir tests/data/2_entities --image-dir tests/data/2_entities/images --dataset-path tests/data/2_entities/dataset_300_samples

tiam load-score --save-dir tests/data/load_score/images_and_seed_consistent --path-to-json-files tests/data/load_score/images_and_seed_consistent/tiam_score_per_prompt



# with JSON files

score    --save-dir    tests/data/2_entities    --image-dir    tests/data/2_entities/json_with_list.json    --dataset-path    tests/data/2_entities/prompts.csv
score   --save-dir    tests/data/2_entities    --image-dir    tests/data/2_entities/json_per_seed.json    --dataset-path    tests/data/2_entities/prompts.csv


```

# Citation

```bibtex
@InProceedings{Grimal_2024_WACV,
    author    = {Grimal, Paul and Le Borgne, Herv\'e and Ferret, Olivier and Tourille, Julien},
    title     = {TIAM - A Metric for Evaluating Alignment in Text-to-Image Generation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2890-2899}
}
```

### Acknowledgments

This work was granted access to the HPC resources of IDRIS under the allocation 2022-AD011014009 made by GENCI. This was also made possible by the use of the FactoryIA supercomputer, financially supported by the Ile-De-France Regional Council.
