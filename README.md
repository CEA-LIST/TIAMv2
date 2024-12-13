# [WACV 2024] TIAM - A Metric for Evaluating Alignment in Text-to-Image Generation

TIAM is a metric to estimate the performance of a visual language model (VLM) in terms of alignment of the prompt with the generated images. It estimates to which extent the entites (objects) and their attributes specified in the prompt are actually visible in the synthetic images. The work was [published at WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Grimal_TIAM_-_A_Metric_for_Evaluating_Alignment_in_Text-to-Image_Generation_WACV_2024_paper.html) and in a previous preprint on [![arxiv](https://img.shields.io/badge/arXiv-2307.05134-b31b1b.svg)](https://arxiv.org/abs/2307.05134).

This repo is a refactoring that makes it easier to use, while the the [original code](https://github.com/grimalPaul/TIAM) focused on reproducind the experiments of the paper.

> <details>
> Paul Grimal, Hervé Le Borgne, Olivier Ferret, Julien Tourille (2024) TIAM - A Metric for Evaluating Alignment in Text-to-Image Generation, WACV
>
> <summary> Abstract </summary>
> The progress in the generation of synthetic images has made it crucial to assess their quality. While several metrics have been proposed to assess the rendering of images, it is crucial for Text-to-Image (T2I) models, which generate images based on a prompt, to consider additional aspects such as to which extent the generated image matches the important content of the prompt. Moreover, although the generated images usually result from a random starting point, the influence of this one is generally not considered. In this article, we propose a new metric based on prompt templates to study the alignment between the content specified in the prompt and the corresponding generated images. It allows us to better characterize the alignment in terms of the type of the specified objects, their number, and their color. We conducted a study on several recent T2I models about various aspects. An additional interesting result we obtained with our approach is that image quality can vary drastically depending on the noise used as a seed for the images. We also quantify the influence of the number of concepts in the prompt, their order as well as their (color) attributes. Finally, our method allows us to identify some seeds that produce better images than others, opening novel directions of research on this understudied topic.
> </details>

## Install

Install with [uv](https://docs.astral.sh/uv//), that itself can be [installed with one line](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv build
source .venv/bin/activate
tiam --help
```

Future works will allow to install with `pip`

## Usage

To evaluate a VLM the general workflow consists to:

* create a dataset of prompts or use [an existing one](doc/benchmarking.md)
* generate several images per prompt with the VLM and either put them in an appropriate directory or specify their path in a JSON file
* evaluate with TIAM

To create a dataset, you can use one of the configuration files provided in `src/tiam/data/` then run e.g

```
tiam create-dataset --config-file src/tiam/data/sample.yaml --save-dir tests/data/sample
```

 let first consider  folder `SAVE_DIR` with the following strucure:

```bash
<SAVE_DIR>
 |-- prompts.csv
 |-- prompts.txt
 |-- images/
     |-- the_first_prompt_with_objects_and_attributes.0.png
     |-- the_first_prompt_with_objects_and_attributes.1.png
     |-- the_first_prompt_with_objects_and_attributes.2.png
         (...)
     |-- the_first_prompt_with_objects_and_attributes.15.png
     |-- the_second_prompt_with_objects_and_attributes.0.png
         (...)
         (...)
     |-- the_last_prompt_with_objects_and_attributes.15.png
 |-- dataset/
```

The directory `images/` contains the synthetic images created by your generative model. There are 16 images per prompt, each having a filename relating it to the prompt used to generate it. The file `prompt.csv` contain the prompts with a format explained [here](TODO). A sample of such file is provided for [2 object](tests/data/2_entities/prompts.csv).

Then, let transform the prompt in the CSV file into a dataset that can be used by TIAM:

```
tiam create-dataset --config-file src/tiam/data/2_colored_entities.yaml --save-dir tests/data/2_colored_entities_dataset

```

And get the perfomances with:

```
tiam score --save-dir SAVE_DIR 
```

**note**: `images` can be replaced by a tarball or a JSON file as explained [here](TODO). There are also alternative to `prompt.csv` to define the prompts, as explained [here](TODO)

## Detailled usage

### Creating the prompt.csv file

entity must be different
adj for each entity if adj

### Create a Dataset

Create a dataset based on the provided configuration file and save it to the specified path.

```bash
tiam create-dataset --config-file <config.yaml> --save-dir <SAVE_DIR>
```

Or available datasets on the hub:

* `Paulgrim/2_entities`: 300 prompts with 2 entities
* `Paulgrim/3_entities`: 300 prompts with 3 entities
* `Paulgrim/2_colored_entities`: 300 prompts with 2 entities and 2 colors
* `Paulgrim/3_colored_entities`: 300 prompts with 3 entities and 3 colors

```python
from datasets import load_dataset

dataset = load_dataset("Paulgrim/2_entities")
```

#### Options

* `--config-file`: Path to the configuration file for creating the dataset.
* `--save-dir`: folder to save the created prompt dataset.

### Compute TIAM Score

Compute the TIAM score for the given images and dataset using the specified model.

```bash
tiam score --save-dir <SAVE_DIR> --image-dir <IMAGE_DIR> --dataset-path-or-url <DATASET_PATH_OR_URL> --model-detect-segment <MODEL_PATH_OR_URL> --batch-size <BATCH_SIZE> --detect-only
```

#### Options

* `--save-dir`: Directory where the results will be saved.
* `--image-dir`: Directory containing the images to be scored.
* `--dataset-path-or-url`: Path or URL to the dataset or csv
* `--model-detect-segment`: Path or URL to the model (default: "yolov8x-seg.pt").
* `--batch-size`: Batch size for processing (default: 32).
* `--detect-only`: Flag to indicate if only detection should be performed.

### Load Score Data

Load data from multiple JSON files, display it and save the results to the specified directory.

```bash
tiam load-score --save-dir <SAVE_DIR> --path-to-json-files <PATH_TO_JSON_FILES>
```

#### Options

* `--save-dir`: Directory where the results will be saved.
* `--path-to-json-files`: Path to the directory containing JSON files to be loaded.

## Commands

* `create-dataset`: Create a dataset based on the provided configuration file and save it to the specified path.
* `score`: Compute the TIAM score for the given images and dataset using the specified model.
* `load-score`: Load data from multiple JSON files, display it and save the results to the specified directory.

## Usage Example

Examples command from the tests

```bash
tiam create-dataset --config-file src/tiam/data/2_colored_entities.yaml --save-dir tests/data/2_colored_entities_dataset

tiam score --save-dir tests/data/2_entities --image-dir tests/data/2_entities/images --dataset-path-or-url tests/data/2_entities/dataset_300_samples

tiam load-score --save-dir tests/data/load_score/images_and_seed_consistent --path-to-json-files tests/data/load_score/images_and_seed_consistent/tiam_score_per_prompt



# with JSON files

score    --save-dir    tests/data/2_entities    --image-dir    tests/data/2_entities/json_with_list.json    --dataset-path-or-url    tests/data/2_entities/prompts.csv
score   --save-dir    tests/data/2_entities    --image-dir    tests/data/2_entities/json_per_seed.json    --dataset-path-or-url    tests/data/2_entities/prompts.csv


```

## TODO

* [ ] add support to pass a csv with prompt entity 1, entity 2, entity 3, color 1, color 2, color 3, transform it in the current format of dataset to iterate on it
* [ ] Change the discriminator YOLO and Segmentation to
  * [ ] VQA ?
  * [ ] Grounding DINO + SAM ?

## Ressources

Metric vqa utilisable
ajouter le vqa score en utilisant ce model en disciminator
<https://github.com/linzhiqiu/t2v_metrics>

## Citation

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

## Acknowledgments

This work was granted access to the HPC resources of IDRIS under the allocation 2022-AD011014009 made by GENCI. This was also made possible by the use of the FactoryIA supercomputer, financially supported by the Ile-De-France Regional Council.
