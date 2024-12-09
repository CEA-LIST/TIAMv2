We provide the scores of several models with fixed datasets of several types. Each dataset is restricted to at most 300 prompts to limit the computation time, including to generate the images. In practice, we verified that the relative order of the VLM performance is the same as when one uses all possible prompts for the given number of entities and attributes. TIAM scores are computed with 16 images per prompt.

Available datasets are:
* `Paulgrim/2_entities`: 300 prompts with 2 entities (without attribute)
* `Paulgrim/3_entities`: 300 prompts with 3 entities (without attribute)
* `Paulgrim/2_colored_entities`: 300 prompts with 2 entities, each with a color attribute
* `Paulgrim/3_colored_entities`: 300 prompts with 3 entities, each with a color attribute

To get them, let use [this method](../README.md#from-huggingface):
```bash
tiam get-hub-dataset --dataset-name "Paulgrim/2_entities" --save-dir "bench/2_entities/"
tiam get-hub-dataset --dataset-name "Paulgrim/2_colored_entities" --save-dir "bench/2_colored_entities/"
tiam get-hub-dataset --dataset-name "Paulgrim/3_entities" --save-dir "bench/3_entities/"
tiam get-hub-dataset --dataset-name "Paulgrim/3_colored_entities" --save-dir "bench/3_colored_entities/"
```

The 300 prompts are explicit in `<SAVE_DIR>/prompts.txt`. Let generate 16 images (size 512x512 is enough) per prompt with your VLM and save them in `bench/2_entities/images/`. Name them `prompt_with_underscore.nos_img.png` (it is [one of the possible methods](../README.md#linking-images-to-prompt) to link prompts and images to compute TIAM). Finally, you can run TIAM:

```
tiam score --save-dir bench/2_entities/ \
           --image-dir  bench/2_entities/images/ \
           --batch-size 16
```

# Results
At a **threshold of 0.25**, the TIAM scores are around:

| model | 2 entities | 2 entities+colors| 3 entities  | 3 entites+colors|
|:-------:|:-------:|:-------:|:-------:|:-------:|
| [SD 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 44.7 | 20.7 | 8.3 | 1.9 |
| [SD 2](https://huggingface.co/stabilityai/stable-diffusion-2) | 64.2 | 41.7 | 21.2 | 7.7 |

# Notes
To get the image filenames from the prompt you can use:
```python
>>> prompt="a photo of a cat and a car"
>>> nos_img=2                                       
>>> "_".join(prompt.split())+"."+str(nos_img)+".jpg"
'a_photo_of_a_cat_and_a_car.42.jpg'
```
After downloading the datasets, they are in "arrow format" in `<SAVE_DIR>/dataset`. If you test several models, you can put them in several directories and/or specify the dataset path. For example, for our benchmarking we had the following structure:

<details>

```
.
├── 2_colored_entities
│   ├── dataset
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── dataset_dict.json
│   ├── prompts.txt
│   ├── SDv1-4
│   │   ├── dataset_prompt
│   │   │   ├── data-00000-of-00001.arrow
│   │   │   ├── dataset_info.json
│   │   │   └── state.json
│   │   ├── images.tar
│   │   ├── index.txt
│   └── SDv2
│       ├── dataset_prompt
│       │   ├── data-00000-of-00001.arrow
│       │   ├── dataset_info.json
│       │   └── state.json
│       ├── images.tar
│       ├── index.txt
├── 2_entities
│   ├── dataset
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── dataset_dict.json
│   ├── prompts.txt
│   ├── SDv1-4
│   │   ├── dataset_prompt
│   │   │   ├── data-00000-of-00001.arrow
│   │   │   ├── dataset_info.json
│   │   │   └── state.json
│   │   ├── images.tar
│   │   └── index.txt
│   └── SDv2
│       ├── dataset_prompt
│       │   ├── data-00000-of-00001.arrow
│       │   ├── dataset_info.json
│       │   └── state.json
│       ├── images.tar
│       └── index.txt
├── 3_colored_entities
│   ├── dataset
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── dataset_dict.json
│   ├── prompts.txt
│   ├── SDv1-4
│   │   ├── dataset_prompt
│   │   │   ├── data-00000-of-00001.arrow
│   │   │   ├── dataset_info.json
│   │   │   └── state.json
│   │   ├── images.tar
│   │   ├── index.txt
│   └── SDv2
│       ├── dataset_prompt
│       │   ├── data-00000-of-00001.arrow
│       │   ├── dataset_info.json
│       │   └── state.json
│       ├── images.tar
│       ├── index.txt
└── 3_entities
    ├── dataset
    │   ├── data-00000-of-00001.arrow
    │   ├── dataset_info.json
    │   └── state.json
    ├── dataset_dict.json
    ├── prompts.txt
    ├── SDv1-4
    │   ├── images.tar
    │   └── index.txt
    └── SDv2
        ├── dataset_prompt
        │   ├── data-00000-of-00001.arrow
        │   ├── dataset_info.json
        │   └── state.json
        ├── images.tar
        ├── index.txt

```

</details>

And we used the following commands to get the TIAM scores:
```bash
export PATH2DIR="bench/2_entities" # or "bench/2_colored_entities" or ...

export model="SDv1-4/" # or export model="SDv2/"

tiam score --save-dir ${PATH2DIR}/${model}/ \
           --dataset-path ${PATH2DIR}/dataset \
           --image-dir ${PATH2DIR}/${model}/images.tar

```
The resulting JSON files for each prompt are saved in `${PATH2DIR}/${model}/tiam_score_per_prompt/`