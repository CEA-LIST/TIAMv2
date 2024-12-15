We provide the scores of several models with fixed datasets of several types. Each dataset is restricted to at most 300 prompts to limit the computation time, including to generate the images. In practice, we verified that the relative order of the VLM performance is the same as when one uses all possible prompts for the given number of entities and attributes.

| model | 2 entities | 2 entities+colors| 3 entities  | 3 entites+colors|
|:-------:|:-------:|:-------:|:-------:|:-------:|
| SD 1.4 | 44.7 | | | |
| SD 2.0 | 64.2 | | | |

# 2 entities
* Get dataset and save the prompts in a text file `bench/2_entities/prompts.txt`. See [here](../README.md#from-huggingface) for the corresponding code.
* generate 16 images (size 512x512 is enough) per prompt with your VLM and save them in `bench/2_entities/images/`. Name them `prompt_with_underscore.nos_img.png` (it is [one of the possible methods](../README.md#linking-images-to-prompt) to link prompts and images to compute TIAM)
* run TIAM

```
tiam score --save-dir bench/2_entities/ --image-dir  bench/2_entities/images/  --batch-size 16
```

# 3 entities

# 2 entities with colors

# 3 entities with colors

TODO: for 2/3/4? objects with or w/o colors, explain how to (which command)
* get dataset from huggingface 
* get the corresponding prompts.txt
* run tiam

Also:
* report the results for several VLM, link to models, latex table?...
* being as concise as possible

