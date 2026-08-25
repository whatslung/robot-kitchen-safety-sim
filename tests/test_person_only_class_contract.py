from train import prep_3way, prepare_yolo_split


def test_training_yaml_generators_use_only_person_class():
    assert prepare_yolo_split.CLASSES == ["person"]
    assert prep_3way.CLASSES == ["person"]
