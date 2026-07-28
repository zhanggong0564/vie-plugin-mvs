# MVS 装箱清单物料检验插件

插件通过 `vie.plugins` 注册 `/api/v1/mvs_inspect` 多图片接口。上传图片文件名必须以
`-序号` 结尾，`-1` 是装箱清单，`-2` 起是待检物料标签。

```bash
curl -X POST http://127.0.0.1:3001/api/v1/mvs_inspect \
  -F 'files=@packing-1.jpg' \
  -F 'files=@packing-2.jpg' \
  -F 'json_data={"selected_item_key":"plug"}'
```

`selected_item_key` 可省略，省略时按物料编码和名称自动匹配清单项目。响应中的业务
状态为 `PASS`、`FAIL` 或 `REVIEW`。

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
  --manifest demo/data/MVS/manifest_images/5a03d7c0-d742-408d-af83-044104f7c7f7.JPG \
  --label demo/data/MVS/images_to_inspect/01ecd42e-04cf-4c08-a3b3-0e7f6034e5ac.JPG
```

也可以用 `--input-dir` 读取按 `-1/-2/...` 命名的图片序列，并通过
`--selected-item-key`、`--device`、`--output-dir` 调整场景。输出目录包含
`result.json`、每次 OCR 的 token JSON、二维码内容、框选图和阶段耗时。
退出码分别为 PASS=`0`、FAIL=`2`、REVIEW=`3`。

可使用以下环境变量：

- `MVS_OCR_DEVICE`：默认 `gpu:0`；GPU 模式缺少 CUDA Provider 时直接报错
- `MVS_DOC_ORIENTATION_MODEL_DIR`：文档方向模型目录
- `MVS_DOC_UNWARPING_MODEL_DIR`：文档矫正模型目录
- `MVS_TEXT_DETECTION_MODEL_DIR`：本地 PP-OCRv5 检测模型目录
- `MVS_TEXTLINE_ORIENTATION_MODEL_DIR`：文本行方向模型目录
- `MVS_TEXT_RECOGNITION_MODEL_DIR`：本地 PP-OCRv5 识别模型目录
- `MVS_RULES_PATH`：外部物料规则 YAML；缺省使用插件内置规则

生产部署应将模型放入 `weights/mvs/` 的版本化目录，并通过上述变量指定路径，避免
启动时下载模型。
