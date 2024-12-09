from pathlib import Path

import numpy as np
from PIL import Image

from src.eval import TIAM
from tests.utils import IMG_COLOR, get_img, np_to_torch


path2model = Path("/home/pgrimal/Documents/code/data/yolov8xl-seg")
path_openvino = path2model / "yolov8x-seg_openvino_model"
path_onnx = path2model / "yolov8x-seg.onnx"
path_light = "/home/pgrimal/Documents/code/data/yolov8m-seg/yolov8m-seg.pt"

paramstest = {
    "batch_size": 2,
    "attribute_binding": False,
    "confs_for_score": [0.25, 0.4, 0.9],
    "iou_nms": 0.7,
    "min_conf_threshold": 0.25,
    "open_vocabulary": False,
}


def test_tiam_detect_no_color():
    paramstest["attribute_binding"] = False
    model: TIAM = TIAM(model_path=path_light, **paramstest)

    # pas couleur
    # teste une image
    img = get_img(IMG_COLOR, 0)
    image = np.array(Image.open(img["path"]))
    cl = img["classes"]
    seed = img["seed"]
    image = np_to_torch(image)
    model.predict(
        images=image,
        classes=cl,
        color_classes=None,
        seeds_used=[seed],
    )
    print(model.get_scores())

    # test batch de deux images, meme classes ici
    batch = [get_img(IMG_COLOR, 1), get_img(IMG_COLOR, 2)]
    cl = batch[0]["classes"]
    seed = [b["seed"] for b in batch]
    images = [np.array(Image.open(b["path"])) for b in batch]
    images = np_to_torch(images)
    model.predict(
        images=images,
        classes=cl,
        color_classes=None,
        seeds_used=seed,
    )
    tiam = model.get_scores()
    print(tiam)
    a, b, c = model.log_scores()
    print(a.to_markdown())
    print(b.to_markdown())
    print(c.to_markdown())
    # terminer le reste du batch et voir le score

    # couleur
    # test batch de deux images


def test_tiam_detect_color():
    paramstest["attribute_binding"] = True
    model: TIAM = TIAM(model_path=path_light, **paramstest)

    for i in range(len(IMG_COLOR)):
        img = get_img(IMG_COLOR, i)
        image = np.array(Image.open(img["path"]))
        cl = img["classes"]
        seed = img["seed"]
        colors = img["colors"]
        image = np_to_torch(image)
        model.predict(
            images=image,
            classes=cl,
            color_classes=colors,
            seeds_used=[seed],
        )
        print(model.get_scores())
    a, b, c = model.log_scores()
    print(a.to_markdown())
    print(b.to_markdown())
    print(c.to_markdown())


def test_tiam_colors_resize():
    paramstest["attribute_binding"] = True
    model: TIAM = TIAM(model_path=path_light, **paramstest)
    resize = (768, 768)
    batch_size = 2
    for i in range(len(IMG_COLOR)):
        img = get_img(IMG_COLOR, i)
        image = np.array(Image.open(img["path"]).resize(resize))
        cl = img["classes"]
        seed = img["seed"]
        colors = img["colors"]
        image = np_to_torch(image)
        image = image.repeat(batch_size, 1, 1, 1)
        seed = [seed] * batch_size
        print(image[0, 0, 0, 0])
        model.predict(
            images=image,
            classes=cl,
            color_classes=colors,
            seeds_used=seed,
        )
        # print(model.get_scores())
    a, b, c = model.log_scores()
    print(a.to_markdown())
    print(b.to_markdown())
    print(c.to_markdown())


# def test_tiam_boxplot():
#     paramstest["attribute_binding"] = False
#     model: TIAM = TIAM(model_path=path_light, **paramstest)

#     # test batch de deux images, meme classes ici

#     steps = 16
#     dataset_size = len(IMG_COLOR)
#     batch_size = paramstest["batch_size"]
#     for i in range(steps):
#         batch_idx = np.random.choice(dataset_size, batch_size, replace=False)
#         batch = [get_img(IMG_COLOR, batch_idx[k]) for k in range(batch_size)]


#     batch = [get_img(IMG_COLOR, 1), get_img(IMG_COLOR, 2)]
#     cl = batch[0]["classes"]
#     seed = [b["seed"] for b in batch]
#     images = [np.array(Image.open(b["path"])) for b in batch]
#     images = np_to_torch(images)
#     model.predict(
#         images=images,
#         classes=cl,
#         color_classes=None,
#         seeds_used=seed,
#     )
#     tiam = model.get_scores()
#     print(tiam)
#     a, b, c = model.log_scores()
#     print(a.to_markdown())
#     print(b.to_markdown())
#     print(c.to_markdown())


# def test_tiam_detect_openvino():
#     TIAM(path_openvino, **paramstest)

# test une image

# test batch de deux images
