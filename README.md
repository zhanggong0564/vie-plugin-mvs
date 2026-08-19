# MVS 装箱清单物料检验插件

插件通过 `vie.plugins` 注册 `/api/v1/mvs_inspect` 单图片即时检测接口。每次请求
上传一张图片，文件名必须为 `<业务名称>-<序号>-<13位毫秒时间戳>.<扩展名>`。
`-1-` 是左右两份装箱清单，`-2-` 起依次对应 `target_names` 中的待检物料标签；
同一产品通过请求中的非空 `sn` 关联清单解析结果。

```bash
curl -X POST http://127.0.0.1:3001/api/v1/mvs_inspect \
  -F 'file=@中压-MVS包装-AI拍照-1-1785457380050.jpg' \
  -F 'json_data=<test.json'
```

`modelParams.target_names` 必须提供至少一个有序检测项；`guideline_coordinates`
必须提供 `2 + target_names 数量` 组归一化四顶点，前两组裁剪左右清单，后续组
依次过滤实物图片的 OCR 文字框。每张图片同步返回统一 `CommonResponse`，业务
状态为 `PASS`、`FAIL` 或 `REVIEW`。实物图片早于清单图片到达时返回 `REVIEW`。

## 运行依赖

- `onnxruntime-gpu==1.20.1`
- OpenCV、NumPy 和 PyYAML
- PP-OCRv5 完整五模型流水线

插件直接使用 ONNX Runtime 串联文档方向、UVDoc 矫正、文本检测、文本行方向和
文本识别五个模型，不依赖 PaddleOCR 或 PaddleX。文本检测最长边限制为 1536
像素，避免高分辨率现场图片产生过大的推理中间张量。五个模型统一通过框架
`services.inference.InferenceRunner` 执行，方向分类和 CTC 解码复用
`services.base` 公共管线。

首次下载官方模型并运行仓库样例：

```bash
conda run -n mobile_vision python plugins/vie-plugin-mvs/examples/run.py \
  --download-models
```

后续直接调试：

```bash
conda run -n mobile_vision python plugins/vie-plugin-mvs/examples/run.py \
  --manifest demo/data/MVS/manifest_images/lQDPJws1TeiPbhvND6DNC7iwmr6EaAnYPSQKP0Joak1aAA_3000_4000.png \
  --label demo/data/MVS/images_to_inspect/01ecd42e-04cf-4c08-a3b3-0e7f6034e5ac.JPG \
  --label demo/data/MVS/images_to_inspect/0b6f7312-8a83-47f3-8484-8e90b263a6ea.JPG \
  --label demo/data/MVS/images_to_inspect/af104e9f-2915-4b7c-b30f-48f80526a918.JPG
```

`--input-dir` 读取按 `-1` 至 `-4` 命名的图片序列，并可通过 `--device`、
`--output-dir` 调整场景。输出目录包含
`result.json`、每次 OCR 的 token JSON、二维码内容、框选图和阶段耗时。
退出码分别为 PASS=`0`、FAIL=`2`、REVIEW=`3`。

需要人工调整双清单引导线时，可启动浏览器编辑器：

```bash
conda run -n mobile_vision python \
  plugins/vie-plugin-mvs/examples/guideline_editor.py
```

打开 `http://127.0.0.1:8011`，拖动左右清单的四个角点；页面底部会实时生成
包含三张标签坐标的完整 `guideline_coordinates`。

可使用以下环境变量：

- `MVS_OCR_DEVICE`：默认 `gpu:0`；GPU 模式缺少 CUDA Provider 时直接报错
- `MVS_DOC_ORIENTATION_MODEL_DIR`：文档方向模型目录
- `MVS_DOC_UNWARPING_MODEL_DIR`：文档矫正模型目录
- `MVS_TEXT_DETECTION_MODEL_DIR`：本地 PP-OCRv5 检测模型目录
- `MVS_TEXTLINE_ORIENTATION_MODEL_DIR`：文本行方向模型目录
- `MVS_TEXT_RECOGNITION_MODEL_DIR`：本地 PP-OCRv5 识别模型目录
- `MVS_RULES_PATH`：外部物料规则 YAML；缺省使用插件内置规则
- `MVS_SESSION_TTL_SECONDS`：按 SN 缓存清单解析结果的秒数，默认 1800

生产部署应将模型放入 `weights/mvs/` 的版本化目录，并通过上述变量指定路径，避免
启动时下载模型。
