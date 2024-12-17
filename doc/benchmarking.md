We provide the scores of several models with fixed datasets of several types. Each dataset is restricted to at most 300 prompts to limit the computation time, including to generate the images. In practice, we verified that the relative order of the VLM performance is the same as when one uses all possible prompts for the given number of entities and attributes. TIAM scores are computed with 16 images per prompt.

Available datasets are:
* `Paulgrim/2_entities`: 300 prompts with 2 entities
* `Paulgrim/3_entities`: 300 prompts with 3 entities
* `Paulgrim/2_colored_entities`: 300 prompts with 2 entities and 2 colors
* `Paulgrim/3_colored_entities`: 300 prompts with 3 entities and 3 colors

To get them, let use [this method](../README.md#from-huggingface):
```bash
tiam get-hub-dataset --dataset-name "Paulgrim/2_entities" --save-dir "bench/2_entities/"
tiam get-hub-dataset --dataset-name "Paulgrim/2_colored_entities" --save-dir "bench/2_colored_entities/"
tiam get-hub-dataset --dataset-name "Paulgrim/3_entities" --save-dir "bench/3_entities/"
tiam get-hub-dataset --dataset-name "Paulgrim/3_colored_entities" --save-dir "bench/3_colored_entities/"
```

The 300 prompts are explcit in `<SAVE_DIR>/prompts.txt`. Let generate 16 images (size 512x512 is enough) per prompt with your VLM and save them in `bench/2_entities/images/`. Name them `prompt_with_underscore.nos_img.png` (it is [one of the possible methods](../README.md#linking-images-to-prompt) to link prompts and images to compute TIAM). Finaly, you can run TIAM:

```
tiam score --save-dir bench/2_entities/ --image-dir  bench/2_entities/images/  --batch-size 16
```

# Results
| model | 2 entities | 2 entities+colors| 3 entities  | 3 entites+colors|
|:-------:|:-------:|:-------:|:-------:|:-------:|
| [SD 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 44.7 | | | |
| [SD 2](https://huggingface.co/stabilityai/stable-diffusion-2) | 64.2 | | | |

<!-- 
2 entities       44.7
2 entities + col 64.2
3 entities       
3 entities + col 

-->


