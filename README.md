# [WACV 2024] TIAM - A Metric for Evaluating Alignment in Text-to-Image Generation

> Grimal Paul, Le Borgne Hervé, Ferret Olivier, Tourille Julien
>
> [![arxiv](https://img.shields.io/badge/arXiv-2307.05134-b31b1b.svg)](https://arxiv.org/abs/2307.05134)
>
> [[WACV 2024 pdf]](https://openaccess.thecvf.com/content/WACV2024/html/Grimal_TIAM_-_A_Metric_for_Evaluating_Alignment_in_Text-to-Image_Generation_WACV_2024_paper.html)
>
> <details>
> **<summary> Abstract </summary>**
> Université Paris-Saclay, CEA, List, F-91120, Palaiseau, France
>
> The progress in the generation of synthetic images has made it crucial to assess their quality. While several metrics have been proposed to assess the rendering of images, it is crucial for Text-to-Image (T2I) models, which generate images based on a prompt, to consider additional aspects such as to which extent the generated image matches the important content of the prompt. Moreover, although the generated images usually result from a random starting point, the influence of this one is generally not considered. In this article, we propose a new metric based on prompt templates to study the alignment between the content specified in the prompt and the corresponding generated images. It allows us to better characterize the alignment in terms of the type of the specified objects, their number, and their color. We conducted a study on several recent T2I models about various aspects. An additional interesting result we obtained with our approach is that image quality can vary drastically depending on the noise used as a seed for the images. We also quantify the influence of the number of concepts in the prompt, their order as well as their (color) attributes. Finally, our method allows us to identify some seeds that produce better images than others, opening novel directions of research on this understudied topic.
>
</details>

## Install

```bash
uv sync
uv build
tiam --help

A terme pip install tiam
```

## Usage

In one folder

```bash
<SAVE DIR>
- prompt.csv or datasets folder or load from hgging face with url
- images folder or tarball

tiam score --save-dir SAVE_DIR 
```

### Create a Dataset

Create a dataset based on the provided configuration file and save it to the specified path.

```bash
tiam create-dataset --config-path <CONFIG_PATH> --save-path <SAVE_PATH>
```

#### Options

- `--config-path`: Path to the configuration file for creating the dataset.
- `--save-path`: Path where the created dataset will be saved.

### Compute TIAM Score

Compute the TIAM score for the given images and dataset using the specified model.

```bash
tiam score --save-dir <SAVE_DIR> --image-dir <IMAGE_DIR> --dataset-path-or-url <DATASET_PATH_OR_URL> --model-path-or-url <MODEL_PATH_OR_URL> --batch-size <BATCH_SIZE> --detect-only
```

#### Options

- `--save-dir`: Directory where the results will be saved.
- `--image-dir`: Directory containing the images to be scored.
- `--dataset-path-or-url`: Path or URL to the dataset.
- `--model-path-or-url`: Path or URL to the model (default: "yolov8x-seg.pt").
- `--batch-size`: Batch size for processing (default: 32).
- `--detect-only`: Flag to indicate if only detection should be performed.

### Load Score Data

Load data from multiple JSON files, display it and save the results to the specified directory.

```bash
tiam load-score --save-dir <SAVE_DIR> --path-to-json-files <PATH_TO_JSON_FILES>
```

#### Options

- `--save-dir`: Directory where the results will be saved.
- `--path-to-json-files`: Path to the directory containing JSON files to be loaded.

## Commands

- `create-dataset`: Create a dataset based on the provided configuration file and save it to the specified path.
- `score`: Compute the TIAM score for the given images and dataset using the specified model.
- `load-score`: Load data from multiple JSON files, display it and save the results to the specified directory.

## Usage Example

Exmples command from the tests

```bash
tiam create-dataset --config-path src/tiam/data/2_colored_entities.yaml --save-path tests/data/2_colored_entities_dataset

tiam score --save-dir tests/data/2_entities --image-dir tests/data/2_entities/images --dataset-path-or-url tests/data/2_entities/dataset_300_samples

tiam load-score --save-dir tests/data/load_score/images_and_seed_consistent --path-to-json-files tests/data/load_score/images_and_seed_consistent/tiam_score_per_prompt
```

## TODO

- [ ] add support to pass a csv with prompt entity 1, entity 2, entity 3, color 1, color 2, color 3, transform it in the current format of dataset to iterate on it
- [ ] Change the discriminator YOLO and Segmentation to
  - [ ] VQA ?
  - [ ] Grounding DINO + SAM ?

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
