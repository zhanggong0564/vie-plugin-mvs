"""MVS 装箱清单物料检验调试入口。

使用包含按 ``-1`` 至 ``-5`` 命名图片的目录运行：

    conda run -n mobile_vision python plugins/vie-plugin-mvs/examples/run.py \
        --input-dir /path/to/four-images

首次下载五个官方模型并运行：

    conda run -n mobile_vision python plugins/vie-plugin-mvs/examples/run.py \
        --download-models
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from vie_plugin_mvs.model_config import (  # noqa: E402
    MODEL_SPECS,
    MVSModelConfig,
)
from vie_plugin_mvs.ocr import ONNXRuntimeOCRBackend  # noqa: E402
from vie_plugin_mvs.service import MVSService  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXIT_CODES = {"PASS": 0, "FAIL": 2, "REVIEW": 3}
TARGET_NAMES = ("堵头", "直接头", "油水分离器")
GUIDELINES = (
    (0.0371, 0.0605, 0.5163, 0.0660, 0.5151, 0.9441, 0.0436, 0.9519),
    (0.5157, 0.0652, 0.9784, 0.0739, 0.9772, 0.9331, 0.5210, 0.9441),
    (0.1000, 0.3400, 0.9800, 0.3600, 0.9800, 0.7800, 0.1000, 0.7500),
    (0.0500, 0.3000, 1.0000, 0.3600, 1.0000, 0.8000, 0.0500, 0.7400),
    (0.0700, 0.2800, 0.9800, 0.3300, 0.9500, 0.7400, 0.0800, 0.7200),
)


class RecordingOCRBackend:
    """记录每次 OCR 输入、输出和耗时，推理行为仍委托生产后端。"""

    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def infer(self, image):
        started = time.perf_counter()
        infer_with_visualization = getattr(
            self.backend,
            "infer_with_visualization",
            None,
        )
        if callable(infer_with_visualization):
            tokens, visualization_image = infer_with_visualization(image)
        else:
            tokens = self.backend.infer(image)
            visualization_image = image
        self.calls.append(
            {
                "source_image_shape": list(image.shape),
                "image": visualization_image.copy(),
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


def resolve_inputs(args):
    if args.input_dir:
        paths = sorted(
            path
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(paths) != 4:
            raise SystemExit("--input-dir 必须恰好包含四张图片")
        return [(path.name, read_image(path)) for path in paths]

    if args.manifest:
        if len(args.label) != 3:
            raise SystemExit("--manifest 必须配合三个按业务顺序提供的 --label")
        manifest, labels = args.manifest, args.label
    else:
        raise SystemExit("必须提供 --input-dir，或同时提供 --manifest 和三个 --label")

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


def download_official_models(_device):
    config = MVSModelConfig.from_env(cwd=REPO_ROOT)
    existing = [path for path in config.paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "拒绝覆盖已有 MVS 模型目录，请使用新的版本目录:\n"
            + "\n".join(str(path) for path in existing)
        )

    with tempfile.TemporaryDirectory(
        prefix="vie-mvs-onnx-download-",
        dir="/tmp",
    ) as cache_dir_value:
        cache_dir = Path(cache_dir_value)
        sources = {}
        for spec in MODEL_SPECS:
            package_name = f"{spec.model_name}_onnx"
            archive = cache_dir / f"{package_name}.tar"
            url = (
                "https://paddle-model-ecology.bj.bcebos.com/"
                "paddlex/official_inference_model/paddle3.0.0/"
                f"{package_name}_infer.tar"
            )
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as stream:
                root = cache_dir.resolve()
                for member in stream.getmembers():
                    destination = (cache_dir / member.name).resolve()
                    if root not in destination.parents and destination != root:
                        raise RuntimeError(
                            f"官方模型压缩包包含非法路径: {member.name}"
                        )
                stream.extractall(cache_dir)
            source = cache_dir / package_name
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
        "runtime": "onnxruntime-gpu==1.20.1",
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
                "source_image_shape": call["source_image_shape"],
                "visualization_image_shape": list(call["image"].shape),
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
    backend = RecordingOCRBackend(
        ONNXRuntimeOCRBackend.from_settings(settings, device=args.device)
    )
    model_init_ms = (time.perf_counter() - model_started) * 1000
    service = MVSService(ocr_backend=backend)
    inspect_started = time.perf_counter()
    try:
        result = service.inspect(
            images,
            TARGET_NAMES,
            GUIDELINES,
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
