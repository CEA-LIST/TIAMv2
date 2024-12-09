# TIAM_v2

images in a folder with name prompt_seed.FORMAT
If seed are not consistent between images, score will be compute without regard to the seed.

Gérer un dataset avec plusieurs type de prompt pour calculer ? Le prévoir ? Non pas pour l'instant

- Chercher uniquement dans le folder dans les datasets les prompts avec le même nom si jamais le user a mis d'autres images que celles du dataset dans le dossier

Je fais la version facile et je réfléchis après por la version ou le user nous donne sa liste de prompt ? Je ne pense pas que ce soit possible ou alors données un certain format de csv pour que ce soit plus simple à gérer

prompt, entity1, entity2, entity3, color1, color2, color3 ou qqch comme ça

si pas de seed de noté on ne fait pas de score

J'ai besoin de garder l'information des noms des fichiers pour chaque prompt, je pense que peut être utile

Ou alors on fait prompt par prompt et c'est quand on charger les images pour un prompt que l'on détecte le comportement à avoir

n_classes_detected, c'est la proportion moyenne de classe par image
DOnc par exemple on essaie de générer des images avec trois objets, et on a un score de 0.6 ca veut dire que 60% des classes que l'on essie de générer sur l'images finale sont présentes. Le remultiplier par le nombre d'obet pour en comprendre mieux le score

## Command

̀```bash
tiam create-dataset  --config-path src/tiam/data/3_colored_entities.yaml --save-path data/3_colored_entities

tiam score --save-dir tests/data/2_entities\
--image-dir tests/data/2_entities/images
--dataset-path-or-url tests/data/2_entities/dataset_300_samples
̀```

## Ressources

Metric vqa utilisable
ajouter le vqa score en utilisant ce model en disciminator
<https://github.com/linzhiqiu/t2v_metrics>

/home/data/pgrimal/datasets_prompt
