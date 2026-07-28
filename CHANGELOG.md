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
