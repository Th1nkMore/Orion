import numpy as np

from mmcv.datasets.builder import PIPELINES
from mmcv.datasets.pipelines import Compose


def test_default_format_bundle3d_is_registered_and_stacks_multiview_images():
    assert PIPELINES.get("DefaultFormatBundle3D") is not None
    pipeline = Compose(
        [
            {
                "type": "DefaultFormatBundle3D",
                "class_names": ["vehicle"],
                "with_label": False,
            }
        ]
    )
    images = [
        np.full((4, 5, 3), fill_value=index, dtype=np.float32)
        for index in range(6)
    ]

    result = pipeline({"img": images})

    assert result["img"].stack is True
    assert tuple(result["img"].data.shape) == (6, 3, 4, 5)
