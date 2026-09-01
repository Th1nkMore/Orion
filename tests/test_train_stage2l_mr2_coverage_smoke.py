import importlib


def test_mr2_rebinds_only_dataset_scope_and_identity():
    module = importlib.import_module("scripts.train_stage2l_mr2_coverage_smoke")
    base = importlib.import_module("scripts.train_stage2l_mr1_smoke")
    module._configure_base()
    assert base.SCHEMA == "orion.stage2l_mr2_expanded_coverage_smoke.v1"
    assert base.PROTOCOL_SCHEMA == "orion.stage2l_mr2_training_protocol.v1"
    assert base.DATASET_SCHEMA == "orion.stage2l_expanded_coverage_dataset.v1"
    assert base.PREFLIGHT_SCHEMA == "orion.stage2l_mr2_trainer_preflight.v1"
    assert base.EXPECTED_EVENT_COUNT == 17
    assert base.EXPECTED_TRAIN_EVENT_COUNT == 13
    assert base.EXPECTED_DEV_EVENT_COUNT == 4
    assert base.EXPECTED_GROUP_COUNT == 80
    assert base.EXPECTED_RECORD_COUNT == 1600
    assert base.ALLOWED_BOUNDED_OPTIMIZER_STEPS == (40,)
