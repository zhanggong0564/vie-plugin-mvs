from schemas.exceptions import ModelInferenceError
from services.base import BatchBusinessLogicBase
from services.scenario_registry import scenario_registry
from utils import vision_logger

from .ocr import ONNXRuntimeOCRBackend
from .service import MVSService


@scenario_registry.register("mvs")
class MVSJudgeApi(BatchBusinessLogicBase):
    """适配框架场景生命周期的多图片 MVS 服务。"""

    NORMALIZE = False

    def _initialize_model(self, settings) -> None:
        backend = None
        try:
            backend = ONNXRuntimeOCRBackend.from_settings(settings)
            self.service = MVSService(ocr_backend=backend)
        except Exception as exc:
            if backend is not None:
                backend.close()
            vision_logger.error(f"MVS PP-OCRv5 初始化失败: {exc}")
            raise ModelInferenceError(
                "MVS PP-OCRv5 模型初始化失败",
                scenario="mvs",
                original_error=exc,
            ) from exc

    def inspect_batch(self, images, request_params):
        return self.service.inspect(images, request_params.selected_item_key)

    def close(self) -> None:
        self.service.close()
