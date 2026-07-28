"""MVS 装箱清单物料检验调试入口。

默认使用仓库 ``demo/data/MVS`` 下的一张装箱单和一张堵头标签：

    conda run -n ppocr python plugins/vie-plugin-mvs/examples/run.py

首次下载五个官方模型并运行：

    conda run -n ppocr python plugins/vie-plugin-mvs/examples/run.py \
        --download-models
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vie_plugin_mvs.model_config import (  # noqa: E402
    MODEL_SPECS,
    MVSModelConfig,
)
from vie_plugin_mvs.ocr import PaddleOCRv5Backend  # noqa: E402
from vie_plugin_mvs.service import MVSService  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXIT_CODES = {"PASS": 0, "FAIL": 2, "REVIEW": 3}


class RecordingOCRBackend:
    """记录每次 OCR 输入、输出和耗时，推理行为仍委托生产后端。"""

    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def infer(self, image):
        started = time.perf_counter()
        tokens = self.backend.infer(image)
        self.calls.append(
            {
                "image": image.copy(),
                "tokens": tokens,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "qr_text": None,
            }
        )
        return tokens

    def decode_qr(self, image):
        started = time.perf_counter()
        value = self.backend.decode_qr(image)
        if self.calls:
            self.calls[-1]["qr_text"] = value
            self.calls[-1]["qr_elapsed_ms"] = (
                time.perf_counter() - started
            ) * 1000
        return value

    def close(self):
        self.backend.close()


def parse_args():
    parser = argparse.ArgumentParser(description="调试 MVS 装箱清单物料检验")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--manifest",
        type=Path,
        help="装箱清单图片；配合一个或多个 --label",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        help="包含按 -1/-2/... 命名图片的目录",
    )
    parser.add_argument(
        "--label",
        type=Path,
        action="append",
        default=[],
        help="实物标签图片，可重复传入",
    )
    parser.add_argument("--selected-item-key", help="指定物料 item_key")
    parser.add_argument("--device", default="gpu:0", help="例如 gpu:0 或 cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/vie-mvs-debug"),
        help="调试输出目录",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="从 Paddle BOS 全新下载五个官方模型到 weights/mvs",
    )
    return parser.parse_args()


def default_inputs():
    return (
        REPO_ROOT
        / "demo/data/MVS/manifest_images/5a03d7c0-d742-408d-af83-044104f7c7f7.JPG",
        [
            REPO_ROOT
            / "demo/data/MVS/images_to_inspect/01ecd42e-04cf-4c08-a3b3-0e7f6034e5ac.JPG"
        ],
    )


def resolve_inputs(args):
    if args.input_dir:
        paths = sorted(
            path
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(paths) < 2:
            raise SystemExit("--input-dir 至少需要两张图片")
        return [(path.name, read_image(path)) for path in paths]

    if args.manifest:
        if not args.label:
            raise SystemExit("--manifest 必须至少配合一个 --label")
        manifest, labels = args.manifest, args.label
    else:
        if args.label:
            raise SystemExit("单独使用 --label 时必须同时指定 --manifest")
        manifest, labels = default_inputs()

    images = [("mvs-1" + manifest.suffix.lower(), read_image(manifest))]
    images.extend(
        (f"mvs-{index}{path.suffix.lower()}", read_image(path))
        for index, path in enumerate(labels, start=2)
    )
    return images


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"无法读取图片: {path}")
    return image


def download_official_models(device):
    config = MVSModelConfig.from_env(cwd=REPO_ROOT)
    existing = [path for path in config.paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "拒绝覆盖已有 MVS 模型目录，请使用新的版本目录:\n"
            + "\n".join(str(path) for path in existing)
        )

    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    with tempfile.TemporaryDirectory(
        prefix="vie-mvs-ppocr-download-",
        dir="/tmp",
    ) as cache_dir:
        os.environ["PADDLE_PDX_CACHE_HOME"] = cache_dir
        from paddleocr import PaddleOCR

        pipeline = PaddleOCR(
            ocr_version="PP-OCRv5",
            doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
            doc_unwarping_model_name="UVDoc",
            text_detection_model_name="PP-OCRv5_server_det",
            textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            engine="onnxruntime",
            device=device,
        )
        del pipeline

        source_root = Path(cache_dir) / "official_models"
        sources = {}
        for spec in MODEL_SPECS:
            source = source_root / f"{spec.model_name}_onnx"
            if not source.is_dir() or not any(source.iterdir()):
                raise RuntimeError(
                    f"官方 ONNX 模型下载不完整: {spec.model_name}"
                )
            sources[spec.key] = source

        staging_root = Path(
            tempfile.mkdtemp(
                prefix=".mvs-model-staging-",
                dir=REPO_ROOT / "weights",
            )
        )
        try:
            staged = {}
            for spec in MODEL_SPECS:
                destination = staging_root / spec.relative_path.split("/")[-1]
                shutil.copytree(sources[spec.key], destination)
                staged[spec.key] = destination
            for spec in MODEL_SPECS:
                destination = config.paths[spec.key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged[spec.key].replace(destination)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    write_model_manifest(config)
    config.validate()
    print("五个 PP-OCRv5 官方模型已从 Paddle BOS 下载并校验。")


def write_model_manifest(config):
    root = (REPO_ROOT / "weights/mvs").resolve()
    records = []
    checksum_lines = []
    for spec in MODEL_SPECS:
        model_dir = config.paths[spec.key]
        files = []
        for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
            digest = sha256(path)
            relative = path.relative_to(root).as_posix()
            files.append({"path": relative, "sha256": digest, "size": path.stat().st_size})
            checksum_lines.append(f"{digest}  {relative}")
        records.append(
            {
                "model_name": spec.model_name,
                "directory": f"weights/mvs/{model_dir.name}",
                "files": files,
            }
        )
    manifest = {
        "source": "Paddle BOS",
        "paddleocr_version": "3.7.0",
        "paddlex_version": "3.7.2",
        "engine": "onnxruntime",
        "models": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_debug(output_dir, result, backend, timings):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_data = result.to_dict()
    result_data["timings_ms"] = timings
    (output_dir / "result.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    calls = []
    for index, call in enumerate(backend.calls):
        tokens = [
            {
                "text": token.text,
                "confidence": token.confidence,
                "polygon": token.polygon,
            }
            for token in call["tokens"]
        ]
        calls.append(
            {
                "call_index": index,
                "image_shape": list(call["image"].shape),
                "elapsed_ms": call["elapsed_ms"],
                "qr_elapsed_ms": call.get("qr_elapsed_ms"),
                "qr_text": call["qr_text"],
                "tokens": tokens,
            }
        )
        (output_dir / f"ocr_call_{index:02d}.json").write_text(
            json.dumps(calls[-1], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cv2.imwrite(
            str(output_dir / f"ocr_call_{index:02d}.jpg"),
            visualize_tokens(call["image"], call["tokens"]),
        )
    (output_dir / "ocr_calls.json").write_text(
        json.dumps(calls, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def visualize_tokens(image, tokens):
    canvas = image.copy()
    for index, token in enumerate(tokens):
        points = np.asarray(token.polygon, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(canvas, [points], True, (0, 255, 0), 2)
        origin = tuple(points[0])
        cv2.putText(
            canvas,
            f"{index}:{token.confidence:.2f}",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return canvas


def main():
    args = parse_args()
    total_started = time.perf_counter()
    if args.download_models:
        download_official_models(args.device)

    input_started = time.perf_counter()
    images = resolve_inputs(args)
    input_ms = (time.perf_counter() - input_started) * 1000
    model_started = time.perf_counter()
    backend = RecordingOCRBackend(PaddleOCRv5Backend(device=args.device))
    model_init_ms = (time.perf_counter() - model_started) * 1000
    service = MVSService(ocr_backend=backend)
    inspect_started = time.perf_counter()
    try:
        result = service.inspect(
            images,
            selected_item_key=args.selected_item_key,
        )
        timings = {
            "input_load": input_ms,
            "model_init": model_init_ms,
            "inspect": (time.perf_counter() - inspect_started) * 1000,
            "total": (time.perf_counter() - total_started) * 1000,
        }
        dump_debug(args.output_dir, result, backend, timings)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        print(f"调试输出: {args.output_dir}")
        return EXIT_CODES[result.status.value]
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
