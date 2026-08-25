import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "train" / "eval_multiview_detector.py"
SPEC = importlib.util.spec_from_file_location("eval_multiview_detector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
best_f1 = MODULE.best_f1
score_dataset = MODULE.score_dataset


def _sample(camera, gt, pred):
    return {"camera": camera, "name": camera, "gt": gt, "pred": pred}


def test_score_dataset_reports_overall_and_per_camera_precision_recall():
    samples = [
        _sample(
            "mvNW",
            [[0.1, 0.1, 0.3, 0.5]],
            [
                [0.1, 0.1, 0.3, 0.5, 0.90],
                [0.6, 0.1, 0.8, 0.5, 0.80],
            ],
        ),
        _sample(
            "mvNE",
            [[0.1, 0.1, 0.3, 0.5], [0.6, 0.1, 0.8, 0.5]],
            [[0.6, 0.1, 0.8, 0.5, 0.40]],
        ),
    ]

    scored = score_dataset(samples, confidence=0.35, iou_threshold=0.5)

    assert scored["overall"] == {
        "images": 2,
        "ground_truth": 3,
        "predictions": 3,
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
    }
    assert scored["cameras"]["mvNW"]["fp"] == 1
    assert scored["cameras"]["mvNE"]["fn"] == 1


def test_best_f1_can_choose_a_stricter_threshold():
    samples = [
        _sample(
            "mvNW",
            [[0.1, 0.1, 0.3, 0.5]],
            [
                [0.1, 0.1, 0.3, 0.5, 0.90],
                [0.6, 0.1, 0.8, 0.5, 0.80],
            ],
        )
    ]

    picked = best_f1(samples, thresholds=[0.35, 0.85])

    assert picked["confidence"] == 0.85
    assert picked["metrics"]["precision"] == 1.0
    assert picked["metrics"]["recall"] == 1.0


def test_explicit_model_name_is_not_replaced_by_default_hub_weights():
    assert MODULE._resolve_model("yolo11s.pt") == Path("yolo11s.pt")


def test_missing_ground_truth_label_fails_closed():
    with pytest.raises(FileNotFoundError):
        MODULE._read_person_labels(Path("definitely-missing-person-label.txt"))
