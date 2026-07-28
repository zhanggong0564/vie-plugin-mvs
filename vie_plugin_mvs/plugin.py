from routers.base_batch_router import BaseBatchRouter

from . import business_logic  # noqa: F401  触发 ScenarioRegistry 注册
from .schemas import MVSParams, MVSResponse


class MVSRouter(BaseBatchRouter):
    def __init__(self) -> None:
        super().__init__(
            router_name="mvs_router",
            api_path="/mvs_inspect",
            summary="装箱清单物料检验",
            description=(
                "按文件名 -1、-2、-3 顺序上传；"
                "-1 为装箱清单，其余为实物标签。"
            ),
            detector_type="mvs",
            tag="装箱清单物料检验",
            response_model=MVSResponse,
            min_files=2,
        )

    def request_schema(self, json_dict):
        return MVSParams(**json_dict)

    def build_batch_response(self, result):
        return {
            "code": 1,
            "message": "成功",
            "result": result,
        }


mvs_router = MVSRouter()
