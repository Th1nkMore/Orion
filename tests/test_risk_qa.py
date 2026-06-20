import numpy as np

from uq_estimator.risk_qa import (
    build_risk_qa_answer,
    parse_natural_risk_qa_answer,
    parse_reliability_answer,
    parse_risk_qa_answer,
    reliability_level,
    reliability_percentile,
    render_natural_risk_qa_answer,
    render_reliability_answer,
    render_risk_qa_answer,
    select_critical_objects,
)


def test_reliability_rendering():
    assert reliability_percentile(0.0) == 100
    assert reliability_percentile(1.0) == 0
    assert reliability_level(24) == "very low"
    assert reliability_level(25) == "low"
    assert reliability_level(90) == "very high"


def test_critical_objects_are_filtered_and_ranked():
    boxes = np.array([
        [12.0, 0.0, 0.0],
        [4.0, -3.0, 0.0],
        [2.0, 0.0, 0.0],
        [40.0, 0.0, 0.0],
    ])
    names = ["car", "pedestrian", "traffic_cone", "truck"]
    objects = select_critical_objects(boxes, names)
    assert [item.category for item in objects] == [
        "traffic_cone",
        "pedestrian",
        "car",
    ]
    assert objects[1].position == "front-right"


def test_b2d_categories_are_normalized():
    objects = select_critical_objects(
        np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        ["vehicle.mercedes.coupe_2020", "walker.pedestrian.0007"],
    )
    assert [item.category for item in objects] == ["car", "pedestrian"]


def test_risk_qa_round_trip():
    objects = select_critical_objects(
        np.array([[8.0, 3.0, 0.0]]),
        ["bicycle"],
    )
    answer = build_risk_qa_answer(0.72, objects)
    parsed = parse_risk_qa_answer(render_risk_qa_answer(answer))
    assert parsed == answer
    level, natural_objects = parse_natural_risk_qa_answer(
        render_natural_risk_qa_answer(answer)
    )
    assert level == answer.reliability_level
    assert natural_objects == ("bicycle in the front-left",)
    assert parse_reliability_answer(
        render_reliability_answer(answer)
    ) == answer.reliability_level
