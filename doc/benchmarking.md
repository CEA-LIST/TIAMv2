We provide the scores of several models with fixed datasets of several types. Each dataset is restricted to at most 300 prompts to limit the computation time, including to generate the images. In practice, we verified that the relative order of the VLM performance is the same as when one uses all possible prompts for the given number of entities ans attributes.

# 2 entities
* Get dataset and save the prompts in a text file `bench/2_entities/prompts.txt`


<details>
<summary>code</summary>

```python
from datasets import load_dataset
dataset = load_dataset("Paulgrim/2_entities")

dataset['dataset']=dataset.pop('train')
dataset.save_to_disk('bench/2_entities')

with open("bench/2_entities/prompts.txt", "w") as f:
    for line in dataset['dataset']["prompt"]:
        f.write(f"{line}\n")
```

</details>

* generate 16 images per prompt with your VLM and save them in `bench/2_entities/images/`. Name them `prompt_with_underscore.nos_img.png`
* link prompts and image with [one possible method](../README.md#linking-images-to-prompt)
* run TIAM

TODO: for 2/3/4? objects with or w/o colors, explain how to (which command)
* get dataset from huggingface 
* get the corresponding prompts.txt
* run tiam

Also:
* report the results for several VLM, link to models, latex table?...
* being as concise as possible

