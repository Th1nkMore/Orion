from scripts.audit_stage2a_optimization_smoke import _mean, _variant


def test_variant_uses_final_route_group_component():
    assert _variant("Town02/Route147/hazard/onpath_oracle") == "onpath_oracle"


def test_mean_handles_empty_and_nonempty_values():
    assert _mean([]) is None
    assert _mean([1.0, 2.0, 3.0]) == 2.0
