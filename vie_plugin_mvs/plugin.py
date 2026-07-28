import json

import cv2
import numpy as np
from fastapi import File, Form, UploadFile

from config import settings
from routers.base_router import BaseRouter
from schemas.exceptions import InvalidImageError, InvalidParamsError

from . import business_logic  # noqa: F401  触发 ScenarioRegistry 注册
from .schemas import MVSParams, MVSResponse


class MVSRouter(BaseRouter):
    def __init__(self) -> None:
        super().__init__(
            router_name="mvs_router",
            api_path="/mvs_inspect",
            summary="装箱清单物料检验",
            description="多图片装箱清单物料检验",
            detector_type="mvs",
            tag="装箱清单物料检验",
            register_default_route=False,
        )
        self.router.post(
            "/mvs_inspect",
            summary="装箱清单物料检验",
            description="按文件名 -1、-2、-3 顺序上传；-1 为装箱清单，其余为实物标签。",
            response_model=MVSResponse,
        )(self._handle_batch_request)

    def request_schema(self, json_dict):
        return MVSParams(**json_dict)

    def get_inputs(self, request_params, image):
        raise NotImplementedError("MVS 使用多图片请求")

    async def _handle_batch_request(
        self,
        files: list[UploadFile] = File(..., description="按 -序号 命名的图片列表"),
        json_data: str = Form(default="{}", description="可选检验参数 JSON"),
    ):
        try:
            params = self.request_schema(json.loads(json_data))
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidParamsError(f"MVS 参数校验失败: {exc}") from exc
        if len(files) < 2:
            raise InvalidParamsError("MVS 至少需要 1 张装箱清单和 1 张实物标签")

        images = []
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        for upload in files:
            raw = await upload.read()
            if not raw or len(raw) > max_bytes:
                raise InvalidImageError(
                    f"图片为空或超过 {settings.MAX_UPLOAD_MB}MB: {upload.filename}"
                )
            image = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise InvalidImageError(f"图片解码失败: {upload.filename}")
            images.append((upload.filename or "unknown.jpg", image))

        detector = self.get_detector_singleton()
        try:
            result = await self.inference_admission.run(
                self.detector_type,
                detector.inspect,
                images,
                params.selected_item_key,
            )
        except ValueError as exc:
            raise InvalidParamsError(str(exc)) from exc
        return {
            "code": 1,
            "message": "成功",
            "result": result.to_dict(),
        }


mvs_router = MVSRouter()
