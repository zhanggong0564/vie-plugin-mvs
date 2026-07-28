# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- 新增基于 PP-OCRv5 的装箱清单与物料标签识别流程。
- 新增固定装箱单坐标解析、配置化物料规则和 PASS/FAIL/REVIEW 三态匹配。
- 首批支持堵头 `FQ001811` 和直接头 `FQ001767`。
- 新增多图片顺序校验、二维码冲突、易混字符候选及图像质量复核规则。
- 新增 PP-OCRv5 五模型本地路径校验和 `examples/run.py` 场景调试入口。

### 修复

- 修复 `examples/run.py` 将 UVDoc 矫正坐标绘制到原图导致 OCR 框偏移的问题。

### 变更

- OCR 后端改为纯 ONNX Runtime 五模型流水线，移除 PaddleOCR 和 PaddleX 运行依赖。
- GPU 模式强制校验 CUDA Execution Provider，避免静默回退到 CPU。
- 五模型统一接入框架 InferenceRunner，并复用公共分类与 CTC 识别管线。
- 五模型改由框架 Runner Group 统一管理生命周期，并登记可观测的模型角色。
- 多图片端点接入框架批量路由与业务基类，复用上传暂存、调用统计和逐图关联回流。
