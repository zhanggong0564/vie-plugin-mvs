import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    env_name: str
    relative_path: str
    paddle_arg: str


MODEL_SPECS = (
    ModelSpec(
        key="doc_orientation",
        model_name="PP-LCNet_x1_0_doc_ori",
        env_name="MVS_DOC_ORIENTATION_MODEL_DIR",
        relative_path="weights/mvs/doc_ori_lcnet_v1",
        paddle_arg="doc_orientation_classify_model_dir",
    ),
    ModelSpec(
        key="doc_unwarping",
        model_name="UVDoc",
        env_name="MVS_DOC_UNWARPING_MODEL_DIR",
        relative_path="weights/mvs/doc_unwarp_uvdoc_v1",
        paddle_arg="doc_unwarping_model_dir",
    ),
    ModelSpec(
        key="text_detection",
        model_name="PP-OCRv5_server_det",
        env_name="MVS_TEXT_DETECTION_MODEL_DIR",
        relative_path="weights/mvs/text_det_ppocrv5s_v1",
        paddle_arg="text_detection_model_dir",
    ),
    ModelSpec(
        key="textline_orientation",
        model_name="PP-LCNet_x1_0_textline_ori",
        env_name="MVS_TEXTLINE_ORIENTATION_MODEL_DIR",
        relative_path="weights/mvs/textline_ori_lcnet_v1",
        paddle_arg="textline_orientation_model_dir",
    ),
    ModelSpec(
        key="text_recognition",
        model_name="PP-OCRv5_server_rec",
        env_name="MVS_TEXT_RECOGNITION_MODEL_DIR",
        relative_path="weights/mvs/text_rec_ppocrv5s_v1",
        paddle_arg="text_recognition_model_dir",
    ),
)


@dataclass(frozen=True)
class MVSModelConfig:
    paths: dict[str, Path]

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> "MVSModelConfig":
        root = cwd or Path.cwd()
        paths = {
            spec.key: Path(
                os.getenv(spec.env_name, str(root / spec.relative_path))
            ).resolve()
            for spec in MODEL_SPECS
        }
        return cls(paths=paths)

    def validate(self) -> None:
        missing = [
            f"{spec.model_name}: {self.paths[spec.key]}"
            for spec in MODEL_SPECS
            if not self.paths[spec.key].is_dir()
            or not any(self.paths[spec.key].iterdir())
        ]
        if missing:
            raise FileNotFoundError(
                "MVS 官方模型目录缺失或为空:\n" + "\n".join(missing)
            )

    def paddle_kwargs(self) -> dict[str, str]:
        kwargs = {}
        for spec in MODEL_SPECS:
            kwargs[spec.paddle_arg] = str(self.paths[spec.key])
        return kwargs
