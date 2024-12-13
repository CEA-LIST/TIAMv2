from pathlib import PosixPath
from string import Template

from datasets import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from .factory import (
    all_combinations,
    combinations,
    get_article,
    multi_combinations,
    permutation,
    unique_words,
)


class Template_(Template):
    def get_all_identifiers(self):
        ids = []
        for mo in self.pattern.finditer(self.template):
            named = mo.group("named") or mo.group("braced")
            if named is not None:
                # add a named group
                ids.append(named)
            elif (
                named is None
                and mo.group("invalid") is None
                and mo.group("escaped") is None
            ):
                # If all the groups are None, there must be
                # another group we're not expecting
                raise ValueError("Unrecognized named group in pattern", self.pattern)
        return ids


def get_template_params(config):
    template = Template_(config.template)
    params = template.get_identifiers()
    return template, params


def get_generate_fct(generate: str):
    if generate == "permutation":
        return permutation
    elif generate == "combination":
        return all_combinations
    elif generate == "ordered_combination":
        return combinations
    else:
        raise ValueError(
            "generate should be in ['permutation','combination','ordered_combination']\
            See in README.md for more details"
        )


def get_words(config):
    if "list" in config.keys() and config.list is not None:
        return OmegaConf.to_object(config.list)
    elif "path" in config.keys() and config.path is not None:
        return load_words(config.path)
    else:
        raise ValueError(
            "You should specify a list of 'words' in config. \
            Or a path to a file containing the list of words. \
            See README.md for more details."
        )


def load_words(path):
    # get the relativ path
    with open(path, "r") as f:
        words = [line.rstrip() for line in f]
    return words


def generate_dataset(
    iter_words,
    template,
    template_params,
    adj_params=None,
    labels_params=None,
    adj_apply_on=None,
    indefinite_article_params=None,
):
    prompts = []
    # all params
    params = []
    # only labels that we will detect
    labels = []
    # only adj that we will check
    adjs = []
    # wich adj is applied on which label
    for words in tqdm(iter_words):
        param = dict(zip(template_params, words))
        params.append(param.copy())

        adjs_ = {adj: param[adj] for adj in adj_params}
        labels_ = {label: param[label] for label in labels_params}
        adjs.append(adjs_)
        labels.append(labels_)
        if indefinite_article_params:
            for indefinite_article, word in indefinite_article_params:
                param[indefinite_article] = get_article(param[word])
        prompt = template.substitute(param)
        prompts.append(prompt)
    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "labels_params": labels,
            "adjs_params": adjs,
            "adj_apply_on": [adj_apply_on] * len(prompts),
        }
    )
    return dataset


def create_dataset(config):
    """
    config: configuration file (YAML)
    """
    if isinstance(config, PosixPath):
        config = OmegaConf.load(config)

    template, params = get_template_params(config)

    # filter a/an word
    apply_on = []  # (indefinite_article, word)
    params_with_possible_repetition = template.get_all_identifiers()
    for i, p in enumerate(params_with_possible_repetition):
        if p.startswith("ind_ar"):
            # search for the corresponding word and search the next param
            apply_on.append((p, params_with_possible_repetition[i + 1]))
            params.remove(p)
    # get adj and label
    labels = []
    ajds = []
    adj_label = {}
    for i, p in enumerate(params):
        if p.startswith("entity"):
            labels.append(p)
        if p.startswith("adj"):
            label = "entity" + p.split("adj")[1]
            adj_label[p] = label
            ajds.append(p)
    print(
        f"We detected the following labels: {labels}\
          \nand the following adjectives: {ajds}\
          \nBe sure that adj and label correspond {adj_label}"
    )

    multi_list = True
    for param in params:
        if param not in config.keys():
            multi_list = False

    if not multi_list:
        if "entities" in config.keys():
            words = get_words(config.entities)
        else:
            raise ValueError("You should specify a list of 'entities' in config.")
        if len(ajds) != 0:
            if "adjs" in config.keys():
                adjs = get_words(config.adjs)
            else:
                raise ValueError(
                    "We detect adjectives in your template but you dont give a list of adjectives in config with 'adjs' key."
                )
        words_list = []
        for param in params:
            if param.startswith("entity"):
                words_list.append(words)
            if param.startswith("adj"):
                words_list.append(adjs)
        iter_words = multi_combinations(words_list)
        if "unique" in config.keys():
            unique = OmegaConf.to_object(config.unique)
            if len(unique) != 0:
                positions = []
                for not_be_repeated in unique:
                    positions.append([params.index(p) for p in not_be_repeated])
                iter_words = unique_words(iter_words, positions)

    else:
        # multi_list
        words = []
        for param in params:
            words.append(get_words(config[param]))
        iter_words = multi_combinations(words)
        if "unique" in config.keys():
            unique = OmegaConf.to_object(config.unique)
            if len(unique) != 0:
                positions = []
                for not_be_repeated in unique:
                    positions.append([params.index(p) for p in not_be_repeated])
                iter_words = unique_words(iter_words, positions)

    return generate_dataset(
        iter_words=iter_words,
        template=template,
        labels_params=labels,
        adj_params=ajds,
        adj_apply_on=adj_label,
        template_params=params,
        indefinite_article_params=apply_on,
    )
