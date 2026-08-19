import numpy as np

from routers.base_router import BaseRouter
from schemas.data_base import InputParamsBusiness

from . import business_logic  # noqa: F401  触发 ScenarioRegistry 注册
from .schemas import MVSParams


class MVSRouter(BaseRouter):
    def __init__(self) -> None:
        super().__init__(
            router_name="mvs_router",
            api_path="/mvs_inspect",
            summary="装箱清单物料检验",
            description=(
                "每次上传一张图片；文件名格式为 <业务名称>-<序号>-<13位时间戳>。"
                "-1 为左右装箱清单，-2 起依次对应 target_names。"
            ),
            detector_type="mvs",
            tag="装箱清单物料检验",
        )

    def request_schema(self, json_dict):
        return MVSParams(**json_dict)

    def get_inputs(self, request_params, image):
        raise RuntimeError("MVS 输入必须包含原始文件名")

    def prepare_inputs(
        self,
        request_params: MVSParams,
        image: np.ndarray,
        original_filename: str,
    ) -> InputParamsBusiness:
        return InputParamsBusiness(
            image=image,
            SN=request_params.sn,
            product_type=request_params.modelParams.product_type,
            extra={
                "filename": original_filename,
                "request_params": request_params,
            },
        )


mvs_router = MVSRouter()
