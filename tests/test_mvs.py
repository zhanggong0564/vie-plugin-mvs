from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from services.inference import OnnxRuntimeOptions, RunnerSpec
from vie_plugin_mvs.config import load_rules
from vie_plugin_mvs.inspectors import OCRLabelInspector
from vie_plugin_mvs.matcher import MaterialMatcher
from vie_plugin_mvs.model_config import MODEL_SPECS, MVSModelConfig
from vie_plugin_mvs.models import (
    InspectionStatus,
    LabelObservation,
    OCRToken,
    PackingListItem,
)
from vie_plugin_mvs.ocr import (
    ONNXRuntimeOCRBackend,
    _PPocrRecognitionPipeline,
)
from vie_plugin_mvs.plugin import mvs_router
from vie_plugin_mvs.service import MVSService, order_images
from vie_plugin_mvs.table_parser import FixedPackingListParser


def token(text, x, y, score=0.95):
    return OCRToken(
        text=text,
        confidence=score,
        polygon=[
            [x, y],
            [x + 80, y],
            [x + 80, y + 20],
            [x, y + 20],
        ],
    )


def test_rules_configure_supported_items_and_code_sources():
    rules = load_rules()

    assert rules.items["plug"].aliases == ("堵头", "Plug")
    assert rules.items["plug"].code_sources == ("remarks", "model")
    assert rules.items["direct_head"].code_pattern == r"FQ\d{6}"


def test_fixed_parser_reads_code_from_configured_candidate_column():
    rules = load_rules()
    parser = FixedPackingListParser(rules)
    tokens = [
        token("20", 80, 500),
        token("堵头", 220, 500),
        token("Plug", 220, 515),
        token("个/pcs", 620, 500),
        token("1", 700, 500),
        token("FQ001811", 900, 500),
        token("2:", 80, 535),
        token("油水分离器", 220, 535),
        token("KIT PIT410", 450, 535),
        token("21", 80, 570),
        token("直接头", 220, 570),
        token("Direct Head", 220, 595),
        token("FQ001767", 500, 570),
        token("个/pcs", 620, 570),
        token("1", 700, 570),
    ]

    items = parser.parse(tokens, image_width=1000, image_height=1000)

    assert [(item.item_key, item.material_code_source) for item in items] == [
        ("plug", "remarks"),
        ("direct_head", "model"),
    ]
    assert items[0].material_code == "FQ001811"
    assert items[1].material_code == "FQ001767"


def test_label_inspector_keeps_confusion_as_candidate_only():
    rules = load_rules()
    inspector = OCRLabelInspector(rules)

    observation = inspector.extract_tokens(
        [token("8200006184_FQOO1811_260708", 10, 10), token("堵头", 10, 40)]
    )

    assert observation.material_code is None
    assert observation.code_candidates == ("FQ001811",)
    assert observation.has_ambiguous_code is True


def test_label_inspector_retries_crop_without_promoting_candidate_directly():
    rules = load_rules()
    inspector = OCRLabelInspector(rules)
    calls = []

    class Backend:
        def infer(self, image):
            calls.append(image.shape)
            if len(calls) == 1:
                return [
                    token("8200006184_FQOO1811_260708", 10, 10),
                    token("堵头", 10, 40),
                ]
            return [token("FQ001811", 5, 5, score=0.97)]

        def decode_qr(self, image):
            return None

    image = np.random.default_rng(9).integers(
        40, 220, (200, 300, 3), dtype=np.uint8
    )
    observation = inspector.extract(image, expected_item=None, backend=Backend())

    assert len(calls) == 2
    assert observation.material_code == "FQ001811"
    assert observation.code_candidates == ("FQ001811",)


def test_matcher_returns_pass_fail_and_review():
    rules = load_rules()
    matcher = MaterialMatcher(rules)
    expected = [
        PackingListItem(
            line_no="20",
            item_key="plug",
            name_cn="堵头",
            name_en="Plug",
            model="",
            unit="个/pcs",
            quantity=1,
            remarks="FQ001811",
            material_code="FQ001811",
            material_code_source="remarks",
            confidence=0.95,
        )
    ]

    passed = matcher.evaluate(
        expected,
        LabelObservation(
            item_name="堵头",
            material_code="FQ001811",
            confidence=0.95,
        ),
    )
    failed = matcher.evaluate(
        expected,
        LabelObservation(
            item_name="堵头",
            material_code="FQ001767",
            confidence=0.95,
        ),
        selected_item_key="plug",
    )
    review = matcher.evaluate(
        expected,
        LabelObservation(
            item_name="堵头",
            code_candidates=("FQ001811",),
            has_ambiguous_code=True,
            confidence=0.95,
        ),
    )

    assert passed.status is InspectionStatus.PASS
    assert failed.status is InspectionStatus.FAIL
    assert review.status is InspectionStatus.REVIEW


def test_matcher_fails_clear_name_mismatch_for_selected_item():
    rules = load_rules()
    matcher = MaterialMatcher(rules)
    expected = [
        PackingListItem(
            line_no="20",
            item_key="plug",
            name_cn="堵头",
            material_code="FQ001811",
            confidence=0.95,
        )
    ]

    result = matcher.evaluate(
        expected,
        LabelObservation(item_name="直接头", confidence=0.95),
        selected_item_key="plug",
    )

    assert result.status is InspectionStatus.FAIL


def test_matcher_reviews_qr_ocr_conflict_and_multiple_labels():
    rules = load_rules()
    matcher = MaterialMatcher(rules)
    expected = [
        PackingListItem(
            line_no="20",
            item_key="plug",
            name_cn="堵头",
            material_code="FQ001811",
            material_code_source="remarks",
            confidence=0.95,
        )
    ]

    qr_conflict = matcher.evaluate(
        expected,
        LabelObservation(
            material_code="FQ001811",
            qr_text="FQ001767",
            confidence=0.95,
        ),
    )
    multiple = matcher.evaluate(
        expected,
        LabelObservation(
            material_code="FQ001811",
            detected_codes=("FQ001811",),
            multiple_labels=True,
            confidence=0.95,
        ),
    )

    assert qr_conflict.status is InspectionStatus.REVIEW
    assert multiple.status is InspectionStatus.REVIEW


def test_order_images_requires_manifest_suffix_and_sorts_numerically():
    images = [
        ("batch-10.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
        ("batch-2.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
        ("batch-1.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
    ]

    ordered = order_images(images)

    assert [name for name, _ in ordered] == [
        "batch-1.jpg",
        "batch-2.jpg",
        "batch-10.jpg",
    ]


def test_service_uses_first_image_as_manifest_and_inspects_remaining():
    rules = load_rules()
    manifest_tokens = [
        token("20", 80, 500),
        token("堵头", 220, 500),
        token("Plug", 220, 525),
        token("1", 750, 500),
        token("FQ001811", 900, 500),
    ]
    label_tokens = [
        token("8200006184_FQ001811_260708", 100, 100),
        token("堵头", 100, 130),
    ]
    backend = SimpleNamespace(
        infer=lambda image: manifest_tokens if image[0, 0, 0] == 1 else label_tokens,
        decode_qr=lambda image: None,
    )
    service = MVSService(rules=rules, ocr_backend=backend)
    rng = np.random.default_rng(7)
    manifest_image = rng.integers(40, 220, (1000, 1000, 3), dtype=np.uint8)
    label_image = rng.integers(40, 220, (1000, 1000, 3), dtype=np.uint8)
    manifest_image[0, 0, 0] = 1
    label_image[0, 0, 0] = 2

    result = service.inspect(
        [
            ("batch-2.jpg", label_image),
            ("batch-1.jpg", manifest_image),
        ]
    )

    assert result.manifest_items[0].material_code == "FQ001811"
    assert result.inspections[0].status is InspectionStatus.PASS


def test_service_reviews_bad_manifest_before_ocr():
    rules = load_rules()
    backend = SimpleNamespace(
        infer=lambda image: (_ for _ in ()).throw(AssertionError("OCR 不应执行")),
        decode_qr=lambda image: None,
    )
    service = MVSService(rules=rules, ocr_backend=backend)

    result = service.inspect(
        [
            ("batch-1.jpg", np.zeros((100, 100, 3), dtype=np.uint8)),
            ("batch-2.jpg", np.zeros((100, 100, 3), dtype=np.uint8)),
        ]
    )

    assert result.status is InspectionStatus.REVIEW
    assert "装箱清单图片质量" in result.message


def test_model_config_requires_all_five_local_models(tmp_path):
    config = MVSModelConfig.from_env(cwd=tmp_path)

    with pytest.raises(FileNotFoundError, match="PP-LCNet_x1_0_doc_ori"):
        config.validate()


def _onnx_model_config(tmp_path):
    paths = {}
    for spec in MODEL_SPECS:
        path = tmp_path / spec.key
        path.mkdir()
        (path / "inference.onnx").write_bytes(b"onnx")
        config = (
            "PostProcess:\n  character_dict:\n    - 堵\n    - 头\n"
            if spec.key == "text_recognition"
            else "{}\n"
        )
        (path / "inference.yml").write_text(config, encoding="utf-8")
        paths[spec.key] = path
    return MVSModelConfig(paths=paths)


def _onnx_settings():
    return SimpleNamespace(
        ONNX_REQUIRE_CUDA=False,
        ORT_CUDA_DEVICE_ID=0,
        ORT_CUDNN_CONV_ALGO_SEARCH="HEURISTIC",
        ORT_ARENA_EXTEND_STRATEGY="kSameAsRequested",
        ORT_CUDA_MEM_LIMIT_GB=0.0,
    )


def test_onnx_backend_creates_and_injects_five_framework_runners(tmp_path):
    config = _onnx_model_config(tmp_path)
    settings = _onnx_settings()
    runners = [MagicMock() for _ in MODEL_SPECS]

    with patch(
        "services.inference.group.create_inference_runner",
        side_effect=runners,
    ) as runner_factory:
        backend = ONNXRuntimeOCRBackend.from_settings(
            settings,
            device="gpu:2",
            model_config=config,
        )

    options = OnnxRuntimeOptions.from_settings(
        settings,
        require_cuda=True,
        cuda_device_id=2,
    )
    assert runner_factory.call_args_list == [
        call(
            RunnerSpec(
                scenario="mvs",
                onnx_path=str(config.paths[spec.key] / "inference.onnx"),
                model_role=spec.key,
            ),
            options,
            status_registry=None,
        )
        for spec in MODEL_SPECS
    ]
    assert list(backend.runners.values()) == runners
    assert backend.characters == ["堵", "头", " "]


def test_onnx_backend_cpu_device_uses_framework_cpu_provider():
    options = ONNXRuntimeOCRBackend._runtime_options(
        _onnx_settings(),
        "cpu",
    )

    assert options.providers == ("CPUExecutionProvider",)
    assert options.require_cuda is False


def test_onnx_backend_runner_failure_closes_created_runners(tmp_path):
    first_runner = MagicMock()
    with (
        patch(
            "services.inference.group.create_inference_runner",
            side_effect=[first_runner, RuntimeError("runner failed")],
        ),
        pytest.raises(RuntimeError, match="runner failed"),
    ):
        ONNXRuntimeOCRBackend.from_settings(
            _onnx_settings(),
            model_config=_onnx_model_config(tmp_path),
        )

    first_runner.close.assert_called_once_with()


def test_onnx_backend_ctc_decode_removes_blanks_and_duplicates():
    runner = SimpleNamespace(
        input_infos=[SimpleNamespace(name="x")],
    )
    pipeline = _PPocrRecognitionPipeline(
        runner,
        characters=["堵", "头", " "],
        input_height=48,
        max_width=3200,
    )
    predictions = np.zeros((1, 6, 4), dtype=np.float32)
    predictions[0, range(6), [0, 1, 1, 0, 2, 2]] = [
        0.9,
        0.8,
        0.7,
        0.95,
        0.85,
        0.75,
    ]

    results = pipeline.decode(predictions)

    assert [result.text for result in results] == ["堵头"]
    assert [result.score for result in results] == pytest.approx([0.825])


def test_plugin_exposes_only_multi_image_endpoint():
    routes = mvs_router.get_router().routes
    paths = [route.path for route in routes]

    assert paths == ["/mvs_inspect"]
    assert routes[0].response_model.__name__ == "MVSResponse"
