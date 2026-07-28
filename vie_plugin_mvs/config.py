import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ItemRule:
    item_key: str
    display_name: str
    aliases: tuple[str, ...]
    code_sources: tuple[str, ...]
    code_pattern: str
    inspector: str
    required_fields: tuple[str, ...]
    confidence_threshold: float


@dataclass(frozen=True)
class QualityRule:
    min_blur_variance: float
    min_brightness: float
    max_brightness: float


@dataclass(frozen=True)
class MVSRules:
    items: dict[str, ItemRule]
    columns: dict[str, tuple[float, float]]
    quality: QualityRule


def load_rules(path: str | None = None) -> MVSRules:
    rule_path = Path(
        path
        or os.getenv("MVS_RULES_PATH")
        or Path(__file__).with_name("rules.yaml")
    )
    try:
        raw = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"MVS 规则配置不可用: {rule_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("MVS 规则配置必须是映射")

    try:
        raw_columns = raw["template"]["columns"]
        raw_quality = raw["quality"]
        raw_items = raw["items"]
    except (KeyError, TypeError) as exc:
        raise ValueError("MVS 规则配置缺少 template、quality 或 items") from exc

    columns = {}
    for name in (
        "line_no",
        "name",
        "model",
        "unit",
        "quantity",
        "checked",
        "remarks",
    ):
        bounds = raw_columns.get(name)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not 0 <= float(bounds[0]) < float(bounds[1]) <= 1
        ):
            raise ValueError(f"MVS 列范围非法: {name}")
        columns[name] = (float(bounds[0]), float(bounds[1]))

    items = {}
    for item_key, value in raw_items.items():
        try:
            code_pattern = str(value["code_pattern"])
            re.compile(code_pattern)
            threshold = float(value["confidence_threshold"])
            if not 0 <= threshold <= 1:
                raise ValueError
            item = ItemRule(
                item_key=item_key,
                display_name=str(value["display_name"]),
                aliases=tuple(str(alias) for alias in value["aliases"]),
                code_sources=tuple(str(source) for source in value["code_sources"]),
                code_pattern=code_pattern,
                inspector=str(value["inspector"]),
                required_fields=tuple(
                    str(field) for field in value["required_fields"]
                ),
                confidence_threshold=threshold,
            )
        except (KeyError, TypeError, ValueError, re.error) as exc:
            raise ValueError(f"MVS 物料规则非法: {item_key}") from exc
        if not item.aliases or not item.code_sources:
            raise ValueError(f"MVS 物料规则不能为空: {item_key}")
        items[item_key] = item

    try:
        quality = QualityRule(
            min_blur_variance=float(raw_quality["min_blur_variance"]),
            min_brightness=float(raw_quality["min_brightness"]),
            max_brightness=float(raw_quality["max_brightness"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MVS 图像质量规则非法") from exc
    return MVSRules(items=items, columns=columns, quality=quality)
