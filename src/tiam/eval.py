import itertools
from typing import List, Optional

import numpy as np

from .coco import getlabels2numbers


def all_classes_detected(classes_from_prompt: List[str], detected_classes: List[int]):
    """
    Detect if at least each class from the prompt is detected one time in the image
    If in the prompt there is "a and a", it will be considered detected even if there is only one "a" detected
    Args:
        classes_from_prompt: list of classes from the prompt
        detected_classes: list of detected classes
    Returns:
        all_classes_detected: 1 if all classes are detected, 0 otherwise
        number of classes detected
    """
    labels2number = getlabels2numbers()
    classes_prompt = [labels2number[p] for p in classes_from_prompt]
    # remove duplicates
    detected_classes = list(set(detected_classes))
    # In case of sentence like "a and a"
    classes_prompt = list(set(classes_prompt))
    # compare the two lists and give a score of how many classes are detected / how many classes must be in the image
    len_classes_prompt = len(classes_prompt)
    score = 0
    for c in detected_classes:
        if c in classes_prompt:
            score += 1
    return 1 if score == len_classes_prompt else 0, score


def compute_iou_bbox(bbox1, bbox2):
    """Compute the IoU between two bboxs"""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def compute_iou_seg(mask1: np.array, mask2: np.array) -> float:
    """Compute the IoU between two segmentation masks

    Args:
        mask1 (np.array): segmentation mask of shape (h, w)
        mask2 (np.array): segmentation mask of shape (h, w)

    Returns:
        float: IoU
    """
    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - intersection
    return (intersection / (union)).item()


def multi_combinations(objects):
    # [['a','b'],['c','d']]
    # ('a', 'c'), ('a', 'd')
    # ('b', 'c'),('b', 'd')
    if len(objects) > 2:
        raise ValueError("We can only compare two objects at a time")
    return itertools.product(*objects)


def compare_order_in_prompt_with_detected_classes(
    classes_from_prompt: List[str],
    detected_classes,
):
    """
    Compare the order of the classes in the prompt with the classes that are detected
    i.e. if the prompt is "a and b" and the detected classes are for 2 images are [a, b] and [a]
    The score will be [1, 0.5], we detect better element a than b
    """
    labels2number = getlabels2numbers()
    classes_prompt = [labels2number[p] for p in classes_from_prompt]
    count = [0] * len(classes_prompt)
    detected_classes = detected_classes.tolist()
    for i, c in enumerate(classes_prompt):
        if c in detected_classes:
            count[i] += 1
    return count


def compute_binding_score(
    detected_classes,
    detected_colors,
    percentage_colors,
    colors_to_detect,
    threshold_colors,
    classes_with_color,
):
    """Compute TIAM

    Returns:
        correct_bind: 1 if all classes with color are detected and the color is correct, 0 otherwise
        count: list of 0 and 1, 1 if the class with color is detected and the color is correct, 0 otherwise

    """
    correct = {}
    score_is_null = False
    for c in classes_with_color:
        if c in detected_classes:
            # check if it is the right color
            correct[c] = 0
            idx_c = np.where(detected_classes == c)[0]
            for i in idx_c:
                if (
                    colors_to_detect[c] in detected_colors[i]
                    and percentage_colors[i][np.where(detected_colors[i] == colors_to_detect[c])[0]]
                    >= threshold_colors
                ):
                    correct[c] = 1
                    break
        else:
            score_is_null = True

    if len(correct) > 0 and not score_is_null:
        # all cls with color detected and correct bind
        if sum(correct.values()) == len(correct):
            correct_bind = 1
        else:
            correct_bind = 0
    else:
        correct_bind = 0
    count = [None] * len(classes_with_color)
    for i, c in enumerate(classes_with_color):
        if c in correct.keys():
            if correct[c]:
                count[i] = 1
            else:
                count[i] = 0
    return correct_bind, count


def to_homogeneous_array(array: List[List], add_value=np.nan):
    """Add value to the array to make it homogeneous

    Args:
        array (List[List]): list of list
        add_value (_type_, optional): Defaults to np.nan.

    Returns:
        homogeneous_array: list of list with the same length
    """
    max_len = 0
    for i in range(len(array)):
        if len(array[i]) > max_len:
            max_len = len(array[i])
    homogeneous_array = []
    for array_ in array:
        if len(array_) < max_len:
            homogeneous_array.append(array_ + [add_value] * (max_len - len(array_)))
        else:
            homogeneous_array.append(array_)
    return homogeneous_array


def eval(
    classes_from_prompt: List[str],
    detected_classes: List[int],
    bbox: List[List[float]],
    conf: Optional[List[float]],
    conf_min: List[float] = [0.25],
    binding=False,
    detected_colors=None,
    percentage_colors=None,
    colors_from_prompt=None,
    threshold_colors=0.4,
):
    """
    Args:
        classes_from_prompt: list of classes from the prompt
        detected_classes: list of detected classes
        bbox: list of bbox
        conf: list of confidence
        conf_min: minimum confidence to keep a detection
        binding: if True, we measure the attribute binding
        detected_colors: list of colors detected in the masks
        percentage_colors: percentage of each color detected in the masks
        colors_from_prompt: the labels in the prompt with associated color
        threshold_colors: threshold to consider a color as detected
    Returns:
        if binding is False:
            results_wo_binding: dict of results for each confidence threshold
            count_order_wo_binding: dict of count order for each confidence threshold
        if binding is True:
            (results_wo_binding, results_w_binding): tuple of dict of results for each confidence threshold w and w/o binding
            (count_order_wo_binding, count_order_w_binding): tuple of dict of count order for each confidence threshold w and w/o binding
    """

    results_wo_binding = {}
    count_order_wo_binding = {}

    bbox = np.array(bbox)
    detected_classes = np.array(detected_classes)
    conf = np.array(conf)
    if binding:
        results_w_binding = {}
        count_order_w_binding = {}
        labels2number = getlabels2numbers()
        colors_to_detect = {labels2number[label]: c for label, c in colors_from_prompt.items()}
        classes_with_color = list(colors_to_detect.keys())
        detected_colors = np.array(to_homogeneous_array(detected_colors))
        percentage_colors = np.array(to_homogeneous_array(percentage_colors))

    for c in conf_min:
        indices_to_keep = np.where(conf >= c)[0]
        bbox = bbox[indices_to_keep]
        detected_classes = detected_classes[indices_to_keep]
        conf = conf[indices_to_keep]
        results_wo_binding[c] = all_classes_detected(classes_from_prompt, detected_classes)
        count_order_wo_binding[c] = compare_order_in_prompt_with_detected_classes(
            classes_from_prompt=classes_from_prompt, detected_classes=detected_classes
        )
        if binding:
            detected_colors = detected_colors[indices_to_keep]
            percentage_colors = percentage_colors[indices_to_keep]
            (
                results_w_binding[c],
                count_order_w_binding[c],
            ) = compute_binding_score(
                classes_with_color=classes_with_color,
                colors_to_detect=colors_to_detect,
                detected_classes=detected_classes,
                detected_colors=detected_colors,
                percentage_colors=percentage_colors,
                threshold_colors=threshold_colors,
            )
    if binding:
        return (results_wo_binding, results_w_binding), (count_order_wo_binding, count_order_w_binding)
    else:
        return results_wo_binding, count_order_wo_binding
