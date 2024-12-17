from typer.testing import CliRunner

from src.tiam.main import app

runner = CliRunner()


def test_generate_dataset():
    result = runner.invoke(
        app,
        [
            "create-dataset",
            "--config-file",
            "src/tiam/data/2_colored_entities.yaml",
            "--save-dir",
            "TO_DELETE_2_colored_entities_dataset",
        ],
    )
    assert result.exit_code == 0
    # tiam create-dataset --config-file src/tiam/data/2_colored_entities.yaml --save-dir TO_DELETE_2_colored_entities_dataset


def test_score_with_detection_wo_color():
    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_entities",
            "--image-dir",
            "tests/data/2_entities/images",
            "--dataset-path",
            "tests/data/2_entities/dataset_300_samples",
        ],
    )
    assert result.exit_code == 0


def test_score_with_detection_w_color():
    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_colored_entities",
            "--image-dir",
            "tests/data/2_colored_entities/images",
            "--dataset-path",
            "tests/data/2_colored_entities/dataset_300_samples",
        ],
    )
    assert result.exit_code == 0


def test_load_score():
    result = runner.invoke(
        app,
        [
            "load-score",
            "--save-dir",
            "tests/data/load_score/images_and_seed_consistent",
            "--path-to-json-files",
            "tests/data/load_score/images_and_seed_consistent/tiam_score_per_prompt",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "load-score",
            "--save-dir",
            "tests/data/load_score/images_and_seed_unconsistent",
            "--path-to-json-files",
            "tests/data/load_score/images_and_seed_unconsistent/tiam_score_per_prompt",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "load-score",
            "--save-dir",
            "tests/data/load_score/seeds_not_consistent_same_number_image",
            "--path-to-json-files",
            "tests/data/load_score/seeds_not_consistent_same_number_image/tiam_score_per_prompt",
        ],
    )
    assert result.exit_code == 0


def test_with_csv():
    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_entities",
            "--image-dir",
            "tests/data/2_entities/images",
            "--dataset-path",
            "tests/data/2_entities/prompts.csv",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_entities",
            "--image-dir",
            "tests/data/2_entities/images",
            "--dataset-path",
            "tests/data/2_entities/prompts_without_adj.csv",
        ],
    )
    assert result.exit_code == 0
    # tiam score --save-dir tests/data/2_entities --image-dir tests/data/2_entities/images --dataset-path tests/data/2_entities/prompts_without_adj.csv


def test_score_with_dataset_on_hub():
    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_entities",
            "--image-dir",
            "tests/data/2_entities/images",
            "--dataset-path",
            "Paulgrim/2_entities",
        ],
    )
    assert result.exit_code == 0


# tiam score --save-dir tests/data/2_entities --image-dir tests/data/2_entities/images --dataset-path Paulgrim/2_entities


def test_with_images_in_json():
    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_entities",
            "--image-dir",
            "tests/data/2_entities/json_with_list.json",
            "--dataset-path",
            "tests/data/2_entities/prompts.csv",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "score",
            "--save-dir",
            "tests/data/2_entities",
            "--image-dir",
            "tests/data/2_entities/json_per_seed.json",
            "--dataset-path",
            "tests/data/2_entities/prompts.csv",
        ],
    )
    assert result.exit_code == 0
