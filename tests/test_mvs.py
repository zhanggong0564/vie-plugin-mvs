from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from services.base import CoordinateSpace, Region
from services.inference import OnnxRuntimeOptions, RunnerSpec
from schemas.exceptions import InvalidParamsError
from vie_plugin_mvs.config import MVSSettings, load_rules
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
from vie_plugin_mvs.schemas import MVSParams
from vie_plugin_mvs.service import (
    MVSService,
    order_images,
    parse_single_image_name,
)
from vie_plugin_mvs.table_parser import FixedPackingListParser


TARGET_NAMES = ("堵头", "直接头", "油水分离器")
FULL_GUIDELINES = tuple(
    (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    for _ in range(5)
)
SERVICE_GUIDELINES = (
    FULL_GUIDELINES[0],
    (0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0),
    *FULL_GUIDELINES[2:],
)


def token(text, x, y, score=0.95):
    return OCRToken(
        text=text,
        recognition_score=score,
        region=Region(
            polygon=(
                (x, y),
                (x + 80, y),
                (x + 80, y + 20),
                (x, y + 20),
            ),
            space=CoordinateSpace.PIXEL,
        ),
    )


def test_rules_configure_supported_items_and_code_sources():
    rules = load_rules()

    assert rules.items["plug"].aliases == ("堵头", "Plug")
    assert rules.items["plug"].code_sources == ("remarks", "model")
    assert rules.items["direct_head"].code_pattern == r"FQ\d{6}"
    assert rules.item_key_for_name("Oil-water separator") == "oil_water_separator"


def test_request_parses_three_targets_and_five_normalized_quadrilaterals():
    guideline = ";".join(
        "0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9" for _ in range(5)
    )

    request = MVSParams(
        sn="SN001",
        modelParams={
            "product_type": "PackingList",
            "target_names": "堵头,直接头,油水分离器",
            "guideline_coordinates": guideline,
        }
    )

    assert request.modelParams.target_names == TARGET_NAMES
    assert len(request.modelParams.guideline_coordinates) == 5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_names", "堵头,堵头", "不能重复"),
        (
            "guideline_coordinates",
            "0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9",
            "5 组四边形",
        ),
        (
            "guideline_coordinates",
            ";".join("0.1,0.1,0.9,0.9,0.9,0.1,0.1,0.9" for _ in range(5)),
            "顶点顺序",
        ),
    ],
)
def test_request_rejects_invalid_four_image_contract(field, value, message):
    data = {
        "product_type": "PackingList",
        "target_names": ",".join(TARGET_NAMES),
        "guideline_coordinates": ";".join(
            "0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9" for _ in range(5)
        ),
    }
    data[field] = value

    with pytest.raises(ValueError, match=message):
        MVSParams(sn="SN001", modelParams=data)


def test_request_accepts_dynamic_targets_and_full_business_payload():
    request = MVSParams(
        sn="SN001",
        product="PackingList",
        type="",
        AICameraModel=None,
        custom_field="透传",
        modelParams={
            "product_type": "PackingList",
            "target_names": "堵头",
            "guideline_coordinates": ";".join(
                "0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9" for _ in range(3)
            ),
        },
    )

    assert request.modelParams.target_names == ("堵头",)
    assert len(request.modelParams.guideline_coordinates) == 3


def test_single_image_filename_uses_penultimate_number_as_sequence():
    parsed = parse_single_image_name(
        "中压-MVS包装-AI拍照-1-1785457380050.jpg"
    )

    assert parsed.prefix == "中压-MVS包装-AI拍照"
    assert parsed.sequence == 1
    assert parsed.timestamp == "1785457380050"


@pytest.mark.parametrize(
    "filename",
    [
        "中压-MVS包装-AI拍照-1.jpg",
        "中压-MVS包装-AI拍照-1785457380050.jpg",
        "-1-1785457380050.jpg",
        "中压-MVS包装-AI拍照-0-1785457380050.jpg",
    ],
)
def test_single_image_filename_rejects_invalid_contract(filename):
    with pytest.raises(InvalidParamsError, match="图片文件名必须"):
        parse_single_image_name(filename)


def test_rules_path_reads_environment_override(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "vie_plugin_mvs" / "rules.yaml"
    override = tmp_path / "rules.yaml"
    override.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("MVS_RULES_PATH", str(override))

    assert load_rules().items["plug"].display_name == "堵头"


def test_session_ttl_reads_typed_environment_override(monkeypatch):
    monkeypatch.setenv("MVS_SESSION_TTL_SECONDS", "60")

    assert MVSSettings().session_ttl_seconds == 60

    monkeypatch.setenv("MVS_SESSION_TTL_SECONDS", "0")
    with pytest.raises(ValueError):
        MVSSettings()


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
        ("oil_water_separator", None),
        ("direct_head", "model"),
    ]
    assert items[0].material_code == "FQ001811"
    assert items[2].material_code == "FQ001767"


def test_fixed_parser_accepts_line_number_near_cropped_page_edge():
    rules = load_rules()
    parser = FixedPackingListParser(rules)
    tokens = [
        token("20", 20, 500),
        token("堵头", 220, 500),
        token("1", 700, 500),
        token("FQ001811", 900, 500),
    ]

    items = parser.parse(tokens, image_width=1000, image_height=1000)

    assert [(item.item_key, item.material_code) for item in items] == [
        ("plug", "FQ001811")
    ]


def test_parser_uses_complete_header_to_follow_shifted_table_columns():
    rules = load_rules()
    parser = FixedPackingListParser(rules)

    def shifted(normalized_x):
        return (0.10 + 0.75 * normalized_x) * 1000 - 40

    headers = {
        "line_no": "序号",
        "name": "名称/Name",
        "model": "型号/Model",
        "unit": "单位/Unit",
        "quantity": "数量/qty.",
        "checked": "核对/checked by",
        "remarks": "备注/remarks",
    }
    tokens = [
        token(
            headers[name],
            shifted(sum(rules.columns[name]) / 2),
            200,
        )
        for name in rules.columns
    ]
    tokens.append(token("总箱数 total Qty.", shifted(0.70), 100))
    tokens.extend(
        [
            token("20", shifted(0.09), 500),
            token("堵头/Plug", shifted(0.24), 500),
            token("个/pcs", shifted(0.625), 500),
            token("1", shifted(0.715), 500),
            token("FQ001811", shifted(0.92), 500),
        ]
    )

    items = parser.parse(tokens, image_width=1000, image_height=1000)

    assert len(items) == 1
    assert items[0].item_key == "plug"
    assert items[0].material_code == "FQ001811"
    assert items[0].material_code_source == "remarks"


def test_parser_falls_back_when_headers_are_not_in_column_order():
    rules = load_rules()
    parser = FixedPackingListParser(rules)
    header_xs = {
        name: sum(bounds) / 2 * 1000 - 40
        for name, bounds in rules.columns.items()
    }
    header_xs["quantity"], header_xs["checked"] = (
        header_xs["checked"],
        header_xs["quantity"],
    )
    headers = {
        "line_no": "序号",
        "name": "名称",
        "model": "型号",
        "unit": "单位",
        "quantity": "数量",
        "checked": "核对",
        "remarks": "备注",
    }
    tokens = [
        token(headers[name], header_xs[name], 200) for name in rules.columns
    ]
    tokens.extend(
        [
            token("20", 80, 500),
            token("堵头", 220, 500),
            token("个/pcs", 620, 500),
            token("1", 700, 500),
            token("FQ001811", 900, 500),
        ]
    )

    items = parser.parse(tokens, image_width=1000, image_height=1000)

    assert len(items) == 1
    assert items[0].material_code == "FQ001811"


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


def test_label_inspector_filters_tokens_by_guideline_overlap():
    rules = load_rules()
    inspector = OCRLabelInspector(rules)
    tokens = [
        token("FQ001811", 20, 20),
        token("OUTSIDE", 400, 400),
        token("PARTIAL", 170, 40),
    ]
    guideline = (0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.2)

    filtered = inspector.filter_tokens(tokens, guideline, 1000, 1000, 0.9)

    assert [item.text for item in filtered] == ["FQ001811"]


def test_guided_inspector_decodes_qr_from_guideline_outer_crop():
    rules = load_rules()
    inspector = OCRLabelInspector(rules)
    backend = SimpleNamespace(
        infer=lambda image: [token("FQ001811", 20, 20)],
        decode_qr=MagicMock(return_value="FQ001811"),
    )
    image = np.random.default_rng(11).integers(
        40, 220, (100, 200, 3), dtype=np.uint8
    )

    inspector.extract_guided(
        image,
        expected_item=None,
        backend=backend,
        guideline=(0.1, 0.2, 0.6, 0.2, 0.6, 0.8, 0.1, 0.8),
        selected_item_key="plug",
    )

    assert backend.decode_qr.call_args.args[0].shape == (60, 100, 3)


def test_manifest_outer_crop_uses_quadrilateral_bounds():
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)
    guideline = (
        0.0371,
        0.0605,
        0.5163,
        0.0660,
        0.5151,
        0.9441,
        0.0436,
        0.9519,
    )

    crop = MVSService._outer_crop(image, guideline)

    assert crop.shape[:2] == (2675, 1918)


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
        ("batch-4.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
        ("batch-3.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
        ("batch-2.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
        ("batch-1.jpg", np.zeros((10, 10, 3), dtype=np.uint8)),
    ]

    ordered = order_images(images)

    assert [name for name, _ in ordered] == [
        "batch-1.jpg",
        "batch-2.jpg",
        "batch-3.jpg",
        "batch-4.jpg",
    ]


@pytest.mark.parametrize(
    "names",
    [
        ("batch-1.jpg", "batch-2.jpg"),
        (
            "batch-1.jpg",
            "batch-2.jpg",
            "batch-3.jpg",
            "batch-5.jpg",
        ),
        (
            "batch-1.jpg",
            "batch-2.jpg",
            "batch-3.jpg",
            "other-3.jpg",
        ),
    ],
)
def test_order_images_requires_exactly_one_of_each_sequence(names):
    images = [
        (name, np.zeros((10, 10, 3), dtype=np.uint8))
        for name in names
    ]

    with pytest.raises(ValueError):
        order_images(images)


def test_service_uses_first_image_as_manifest_and_inspects_remaining():
    rules = load_rules()
    manifest_tokens = [
        token("20", 80, 500),
        token("堵头", 220, 500),
        token("Plug", 220, 525),
        token("1", 750, 500),
        token("FQ001811", 900, 500),
        token("21", 80, 600),
        token("直接头", 220, 600),
        token("FQ001767", 500, 600),
        token("1", 750, 600),
        token("22", 80, 700),
        token("油水分离器", 220, 700),
        token("KIT PIT410", 500, 700),
        token("1", 750, 700),
        token("FQ001607", 900, 700),
    ]
    label_tokens = {
        2: [token("FQ001811", 100, 100), token("堵头", 100, 130)],
        3: [token("FQ001767", 100, 100), token("直接头", 100, 130)],
        4: [token("FQ001607", 100, 100), token("油水分离器", 100, 130)],
    }
    backend = SimpleNamespace(
        infer=lambda image: (
            manifest_tokens
            if image[0, 0, 0] == 1
            else (
                []
                if image[0, 0, 0] == 6
                else label_tokens[int(image[0, 0, 0])]
            )
        ),
        decode_qr=lambda image: None,
    )
    service = MVSService(rules=rules, ocr_backend=backend)
    rng = np.random.default_rng(7)
    manifest_image = rng.integers(40, 220, (1000, 1000, 3), dtype=np.uint8)
    manifest_image[:, :500, 0] = 1
    manifest_image[:, 500:, 0] = 6
    label_images = []
    for marker in range(2, 5):
        image = rng.integers(40, 220, (1000, 1000, 3), dtype=np.uint8)
        image[:, :, 0] = marker
        label_images.append(image)

    result = service.inspect(
        [
            ("batch-3.jpg", label_images[1]),
            ("batch-1.jpg", manifest_image),
            ("batch-4.jpg", label_images[2]),
            ("batch-2.jpg", label_images[0]),
        ],
        TARGET_NAMES,
        SERVICE_GUIDELINES,
    )

    assert result.manifest_items[0].material_code == "FQ001811"
    assert [inspection.status for inspection in result.inspections] == [
        InspectionStatus.PASS,
        InspectionStatus.PASS,
        InspectionStatus.PASS,
    ]

    request = MVSParams(
        sn="SN001",
        modelParams={
            "product_type": "PackingList",
            "target_names": TARGET_NAMES,
            "guideline_coordinates": SERVICE_GUIDELINES,
        },
    )
    manifest_response = service.inspect_single(
        "中压-MVS包装-AI拍照-1-1785457380050.jpg",
        manifest_image,
        request,
    )
    label_responses = [
        service.inspect_single(
            f"中压-MVS包装-AI拍照-{index}-17854573800{index}0.jpg",
            image,
            request,
        )
        for index, image in enumerate(label_images, start=2)
    ]

    assert manifest_response["verdict"] == "PASS"
    assert [item["scene"] for item in manifest_response["detailList"]] == list(
        TARGET_NAMES
    )
    assert [item["name"] for item in manifest_response["detailList"]] == [
        "FQ001811",
        "FQ001767",
        "FQ001607",
    ]
    assert [response["verdict"] for response in label_responses] == [
        "PASS",
        "PASS",
        "PASS",
    ]

    mismatched_request = MVSParams(
        sn="SN001",
        modelParams={
            "product_type": "OtherPackingList",
            "target_names": TARGET_NAMES,
            "guideline_coordinates": SERVICE_GUIDELINES,
        },
    )
    with pytest.raises(InvalidParamsError, match="检测参数与清单请求不一致"):
        service.inspect_single(
            "中压-MVS包装-AI拍照-2-1785457381050.jpg",
            label_images[0],
            mismatched_request,
        )


def test_single_label_requires_manifest_session():
    service = MVSService(
        rules=load_rules(),
        ocr_backend=SimpleNamespace(infer=lambda image: [], decode_qr=lambda image: None),
    )
    request = MVSParams(
        sn="SN002",
        modelParams={
            "product_type": "PackingList",
            "target_names": "堵头",
            "guideline_coordinates": FULL_GUIDELINES[:3],
        },
    )

    result = service.inspect_single(
        "中压-MVS包装-AI拍照-2-1785457381050.jpg",
        np.full((100, 100, 3), 100, dtype=np.uint8),
        request,
    )

    assert result["status"] == "false"
    assert result["verdict"] == "REVIEW"
    assert "先上传序号为 1" in result["message"]


def test_single_manifest_fails_when_requested_target_is_missing():
    service = MVSService(
        rules=load_rules(),
        ocr_backend=SimpleNamespace(infer=lambda image: [], decode_qr=lambda image: None),
    )
    service.quality_checker = SimpleNamespace(
        check=lambda image: SimpleNamespace(acceptable=True, reasons=())
    )
    service.table_parser = SimpleNamespace(
        parse=lambda tokens, image_width, image_height: [
            PackingListItem(
                line_no="20",
                item_key="plug",
                name_cn="堵头",
                material_code="FQ001811",
                confidence=0.95,
            )
        ]
    )
    request = MVSParams(
        sn="SN003",
        modelParams={
            "product_type": "PackingList",
            "target_names": "堵头,直接头",
            "guideline_coordinates": FULL_GUIDELINES[:4],
        },
    )

    result = service.inspect_single(
        "中压-MVS包装-AI拍照-1-1785457380050.jpg",
        np.full((100, 100, 3), 100, dtype=np.uint8),
        request,
    )

    assert result["verdict"] == "FAIL"
    assert [item["verdict"] for item in result["detailList"]] == [
        "PASS",
        "FAIL",
    ]
    assert result["detailList"][1]["name"] == ""


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
            ("batch-3.jpg", np.zeros((100, 100, 3), dtype=np.uint8)),
            ("batch-4.jpg", np.zeros((100, 100, 3), dtype=np.uint8)),
        ],
        TARGET_NAMES,
        FULL_GUIDELINES,
    )

    assert result.status is InspectionStatus.REVIEW
    assert "装箱清单图片质量" in result.message


def test_model_config_requires_all_five_local_models(tmp_path):
    config = MVSModelConfig.from_env(cwd=tmp_path)

    with pytest.raises(FileNotFoundError, match="PP-LCNet_x1_0_doc_ori"):
        config.validate()


def test_model_config_reads_environment_override(tmp_path, monkeypatch):
    override = tmp_path / "doc-orientation"
    monkeypatch.setenv("MVS_DOC_ORIENTATION_MODEL_DIR", str(override))

    config = MVSModelConfig.from_env(cwd=tmp_path)

    assert config.paths["doc_orientation"] == override.resolve()


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


def test_business_logic_builds_backend_from_framework_settings():
    from vie_plugin_mvs.business_logic import MVSJudgeApi

    settings = _onnx_settings()
    backend = MagicMock()
    service = MagicMock()
    with (
        patch(
            "vie_plugin_mvs.business_logic.ONNXRuntimeOCRBackend.from_settings",
            return_value=backend,
        ) as backend_factory,
        patch(
            "vie_plugin_mvs.business_logic.MVSService",
            return_value=service,
        ) as service_class,
    ):
        api = MVSJudgeApi(settings)

    backend_factory.assert_called_once_with(settings)
    service_class.assert_called_once_with(ocr_backend=backend)
    assert api.service is service


def test_business_logic_closes_backend_when_service_init_fails():
    from vie_plugin_mvs.business_logic import MVSJudgeApi

    backend = MagicMock()
    with (
        patch(
            "vie_plugin_mvs.business_logic.ONNXRuntimeOCRBackend.from_settings",
            return_value=backend,
        ),
        patch(
            "vie_plugin_mvs.business_logic.MVSService",
            side_effect=RuntimeError("service failed"),
        ),
        pytest.raises(Exception, match="MVS PP-OCRv5 模型初始化失败"),
    ):
        MVSJudgeApi(_onnx_settings())

    backend.close.assert_called_once_with()


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


def test_onnx_backend_keeps_geometry_helpers_compatible():
    from vie_plugin_mvs.ocr_geometry import mini_box, rotate_image

    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    contour = np.array(
        [[[0, 0]], [[4, 0]], [[4, 2]], [[0, 2]]],
        dtype=np.float32,
    )

    assert np.array_equal(
        ONNXRuntimeOCRBackend._rotate_image(image, 180),
        rotate_image(image, 180),
    )
    backend_box, backend_side = ONNXRuntimeOCRBackend._mini_box(contour)
    geometry_box, geometry_side = mini_box(contour)
    assert np.array_equal(backend_box, geometry_box)
    assert backend_side == geometry_side


def test_plugin_exposes_single_image_endpoint():
    routes = mvs_router.get_router().routes
    paths = [route.path for route in routes]

    assert paths == ["/mvs_inspect"]
    assert routes[0].response_model.__name__ == "CommonResponse"
    assert [field.name for field in routes[0].dependant.body_params] == [
        "file",
        "json_data",
    ]
