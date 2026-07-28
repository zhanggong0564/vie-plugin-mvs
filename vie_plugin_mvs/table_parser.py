import re
from collections import defaultdict

from .config import MVSRules
from .models import OCRToken, PackingListItem


_LINE_NO = re.compile(r"^(\d{1,3})[.:：]?$")
_INTEGER = re.compile(r"\d+")


class FixedPackingListParser:
    """按固定装箱单列坐标及序号行锚点恢复目标物料。"""

    def __init__(self, rules: MVSRules) -> None:
        self.rules = rules

    def parse(
        self,
        tokens: list[OCRToken],
        image_width: int,
        image_height: int,
    ) -> list[PackingListItem]:
        del image_height
        anchors = [
            token
            for token in tokens
            if self._column(token, image_width) == "line_no"
            and _LINE_NO.fullmatch(token.text.strip())
        ]
        anchors.sort(key=lambda token: token.center[1])
        if not anchors:
            return []

        rows = []
        anchor_ys = [anchor.center[1] for anchor in anchors]
        for index, anchor in enumerate(anchors):
            upper = (
                float("-inf")
                if index == 0
                else (anchor_ys[index - 1] + anchor_ys[index]) / 2
            )
            lower = (
                float("inf")
                if index == len(anchors) - 1
                else (anchor_ys[index] + anchor_ys[index + 1]) / 2
            )
            row_tokens = [
                token for token in tokens if upper <= token.center[1] < lower
            ]
            line_match = _LINE_NO.fullmatch(anchor.text.strip())
            assert line_match is not None
            parsed = self._parse_row(line_match.group(1), row_tokens, image_width)
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _parse_row(
        self,
        line_no: str,
        tokens: list[OCRToken],
        image_width: int,
    ) -> PackingListItem | None:
        grouped = defaultdict(list)
        for token in sorted(tokens, key=lambda item: (item.center[1], item.center[0])):
            column = self._column(token, image_width)
            if column:
                grouped[column].append(token)

        row_text = " ".join(token.text for token in tokens)
        matches = [
            rule
            for rule in self.rules.items.values()
            if any(alias.casefold() in row_text.casefold() for alias in rule.aliases)
        ]
        if len(matches) != 1:
            return None
        rule = matches[0]

        fields = {
            name: " ".join(token.text for token in grouped[name]).strip()
            for name in ("name", "model", "unit", "quantity", "remarks")
        }
        material_code = None
        material_code_source = None
        for source in rule.code_sources:
            match = re.search(rule.code_pattern, fields.get(source, ""))
            if match:
                material_code = match.group(0)
                material_code_source = source
                break

        quantity_match = _INTEGER.search(fields["quantity"])
        relevant = [
            token.confidence
            for name in ("name", material_code_source)
            if name
            for token in grouped[name]
        ]
        chinese_alias = next(
            (alias for alias in rule.aliases if any("\u4e00" <= c <= "\u9fff" for c in alias)),
            rule.display_name,
        )
        english_alias = next(
            (alias for alias in rule.aliases if alias.isascii()),
            "",
        )
        return PackingListItem(
            line_no=line_no,
            item_key=rule.item_key,
            name_cn=chinese_alias,
            name_en=english_alias,
            model=fields["model"],
            unit=fields["unit"],
            quantity=int(quantity_match.group(0)) if quantity_match else None,
            remarks=fields["remarks"],
            material_code=material_code,
            material_code_source=material_code_source,
            confidence=sum(relevant) / len(relevant) if relevant else 0.0,
        )

    def _column(self, token: OCRToken, image_width: int) -> str | None:
        normalized_x = token.center[0] / image_width
        for name, (left, right) in self.rules.columns.items():
            if left <= normalized_x < right or (
                name == "remarks" and normalized_x == right
            ):
                return name
        return None
