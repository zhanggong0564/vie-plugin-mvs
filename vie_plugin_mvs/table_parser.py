import re
from collections import defaultdict
from statistics import median

from .config import MVSRules
from .models import OCRToken, PackingListItem


_LINE_NO = re.compile(r"^(\d{1,3})[.:：]?$")
_INTEGER = re.compile(r"\d+")
_HEADER_NORMALIZER = re.compile(r"[\W_]+", re.UNICODE)
_MAX_HEADER_Y_SPREAD = 0.06
_MIN_ADJACENT_HEADER_GAP = 0.025
_MAX_LOCAL_SCALE_DEVIATION = 0.45


class FixedPackingListParser:
    """优先按可信表头动态定位，失败时回退固定列坐标。"""

    def __init__(self, rules: MVSRules) -> None:
        self.rules = rules

    def parse(
        self,
        tokens: list[OCRToken],
        image_width: int,
        image_height: int,
    ) -> list[PackingListItem]:
        columns = self._dynamic_columns(tokens, image_width, image_height)
        if columns is None:
            columns = self.rules.columns
        anchors = [
            token
            for token in tokens
            if self._column(token, image_width, columns) == "line_no"
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
            parsed = self._parse_row(
                line_match.group(1),
                row_tokens,
                image_width,
                columns,
            )
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _parse_row(
        self,
        line_no: str,
        tokens: list[OCRToken],
        image_width: int,
        columns: dict[str, tuple[float, float]],
    ) -> PackingListItem | None:
        grouped = defaultdict(list)
        for token in sorted(tokens, key=lambda item: (item.center[1], item.center[0])):
            column = self._column(token, image_width, columns)
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

    def _dynamic_columns(
        self,
        tokens: list[OCRToken],
        image_width: int,
        image_height: int,
    ) -> dict[str, tuple[float, float]] | None:
        candidates = {}
        for name, aliases in self.rules.column_headers.items():
            matches = [
                token
                for token in tokens
                if any(
                    self._normalize_header(alias)
                    in self._normalize_header(token.text)
                    for alias in aliases
                )
            ]
            if not matches:
                return None
            candidates[name] = matches

        header_centers = self._coherent_header_centers(
            candidates,
            image_width,
            image_height,
        )
        if header_centers is None:
            return None

        observed_xs = [center[1] for center in header_centers]
        gaps = [
            right - left for left, right in zip(observed_xs, observed_xs[1:])
        ]
        if any(gap < _MIN_ADJACENT_HEADER_GAP for gap in gaps):
            return None

        template_xs = [
            sum(self.rules.columns[name]) / 2 for name, _, _ in header_centers
        ]
        scales = [
            observed_gap / template_gap
            for observed_gap, template_gap in zip(
                gaps,
                (
                    right - left
                    for left, right in zip(template_xs, template_xs[1:])
                ),
            )
        ]
        median_scale = median(scales)
        if any(
            abs(scale / median_scale - 1) > _MAX_LOCAL_SCALE_DEVIATION
            for scale in scales
        ):
            return None

        boundaries = [bounds[0] for bounds in self.rules.columns.values()]
        boundaries.append(next(reversed(self.rules.columns.values()))[1])
        mapped = [
            self._interpolate(boundary, template_xs, observed_xs)
            for boundary in boundaries
        ]
        if mapped[0] < 0 or mapped[-1] > 1:
            return None
        names = list(self.rules.columns)
        return {
            name: (mapped[index], mapped[index + 1])
            for index, name in enumerate(names)
        }

    def _coherent_header_centers(
        self,
        candidates: dict[str, list[OCRToken]],
        image_width: int,
        image_height: int,
    ) -> list[tuple[str, float, float]] | None:
        best = None
        anchor_ys = {
            token.center[1] / image_height
            for matches in candidates.values()
            for token in matches
        }
        for anchor_y in anchor_ys:
            selected = [
                min(
                    matches,
                    key=lambda token: abs(
                        token.center[1] / image_height - anchor_y
                    ),
                )
                for matches in candidates.values()
            ]
            ys = [token.center[1] / image_height for token in selected]
            spread = max(ys) - min(ys)
            if spread > _MAX_HEADER_Y_SPREAD:
                continue
            centers = [
                (
                    name,
                    token.center[0] / image_width,
                    token.center[1] / image_height,
                )
                for name, token in zip(candidates, selected)
            ]
            xs = [center[1] for center in centers]
            if any(left >= right for left, right in zip(xs, xs[1:])):
                continue
            if best is None or spread < best[0]:
                best = (spread, centers)
        return None if best is None else best[1]

    @staticmethod
    def _normalize_header(value: str) -> str:
        return _HEADER_NORMALIZER.sub("", value).casefold()

    @staticmethod
    def _interpolate(
        value: float,
        source: list[float],
        target: list[float],
    ) -> float:
        index = 0
        while index < len(source) - 2 and value > source[index + 1]:
            index += 1
        source_left, source_right = source[index : index + 2]
        target_left, target_right = target[index : index + 2]
        ratio = (value - source_left) / (source_right - source_left)
        return target_left + ratio * (target_right - target_left)

    def _column(
        self,
        token: OCRToken,
        image_width: int,
        columns: dict[str, tuple[float, float]],
    ) -> str | None:
        normalized_x = token.center[0] / image_width
        for name, (left, right) in columns.items():
            if left <= normalized_x < right or (
                name == "remarks" and normalized_x == right
            ):
                return name
        return None
