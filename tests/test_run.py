from types import SimpleNamespace

import numpy as np

from examples.run import RecordingOCRBackend


def test_recording_backend_uses_token_coordinate_image():
    source = np.zeros((10, 20, 3), dtype=np.uint8)
    preprocessed = np.full((30, 40, 3), 127, dtype=np.uint8)
    backend = SimpleNamespace(
        infer_with_visualization=lambda image: ([], preprocessed),
    )

    recording = RecordingOCRBackend(backend)
    recording.infer(source)

    assert recording.calls[0]["source_image_shape"] == [10, 20, 3]
    assert recording.calls[0]["image"].shape == (30, 40, 3)
    assert np.all(recording.calls[0]["image"] == 127)
