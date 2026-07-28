from schemas.exceptions import ModelInferenceError
from services.base import BusinessLogicBase
from services.scenario_registry import scenario_registry
from utils import vision_logger

from .service import MVSService


@scenario_registry.register("mvs")
class MVSJudgeApi(BusinessLogicBase):
    """适配框架场景生命周期的多图片 MVS 服务。"""

    NORMALIZE = False

    def _initialize_model(self, settings) -> None:
        del settings
        try:
            self.service = MVSService()
        except Exception as exc:
            vision_logger.error(f"MVS PP-OCRv5 初始化失败: {exc}")
            raise ModelInferenceError(
                "MVS PP-OCRv5 模型初始化失败",
                scenario="mvs",
                original_error=exc,
            ) from exc

    def inspect(self, images, selected_item_key=None):
        return self.service.inspect(images, selected_item_key)

    def business_post_process(self, ctx) -> None:
        raise NotImplementedError("MVS 使用多图片 inspect 接口")

    def close(self) -> None:
        self.service.close()
