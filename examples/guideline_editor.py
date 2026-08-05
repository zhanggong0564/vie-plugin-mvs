"""Browser editor for MVS manifest guideline quadrilaterals."""

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = (
    REPO_ROOT
    / "demo/data/MVS/manifest_images/"
    "lQDPJws1TeiPbhvND6DNC7iwmr6EaAnYPSQKP0Joak1aAA_3000_4000.png"
)
MANIFEST_GUIDELINES = (
    (0.0371, 0.0605, 0.5163, 0.0660, 0.5151, 0.9441, 0.0436, 0.9519),
    (0.5157, 0.0652, 0.9784, 0.0739, 0.9772, 0.9331, 0.5210, 0.9441),
)
LABEL_GUIDELINES = (
    (0.1000, 0.3400, 0.9800, 0.3600, 0.9800, 0.7800, 0.1000, 0.7500),
    (0.0500, 0.3000, 1.0000, 0.3600, 1.0000, 0.8000, 0.0500, 0.7400),
    (0.0700, 0.2800, 0.9800, 0.3300, 0.9500, 0.7400, 0.0800, 0.7200),
)


def _pixel_polygons(width: int, height: int) -> list[list[list[float]]]:
    return [
        [
            [values[index] * width, values[index + 1] * height]
            for index in range(0, 8, 2)
        ]
        for values in MANIFEST_GUIDELINES
    ]


def _html(width: int, height: int) -> bytes:
    initial = json.dumps(_pixel_polygons(width, height))
    label_guidelines = json.dumps(LABEL_GUIDELINES)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MVS 清单引导线编辑器</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: system-ui, sans-serif;
    }}
    body {{
      margin: 0;
      background: #111827;
      color: #f9fafb;
    }}
    header, .controls, .output {{
      padding: 12px 18px;
    }}
    header {{
      display: flex;
      align-items: baseline;
      gap: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
    }}
    .hint {{
      color: #cbd5e1;
      font-size: 14px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      background: #1f2937;
    }}
    button {{
      padding: 7px 12px;
      border: 1px solid #64748b;
      border-radius: 6px;
      background: #334155;
      color: white;
      cursor: pointer;
    }}
    button.active {{
      border-color: #facc15;
      background: #854d0e;
    }}
    #viewport {{
      height: calc(100vh - 300px);
      min-height: 420px;
      overflow: auto;
      background: #030712;
    }}
    svg {{
      display: block;
      width: 100%;
      min-width: 900px;
      touch-action: none;
      user-select: none;
    }}
    polygon {{
      fill-opacity: .14;
      stroke-width: 10;
      vector-effect: non-scaling-stroke;
    }}
    circle {{
      stroke: white;
      stroke-width: 4;
      vector-effect: non-scaling-stroke;
      cursor: grab;
    }}
    textarea {{
      box-sizing: border-box;
      width: 100%;
      min-height: 86px;
      padding: 10px;
      border: 1px solid #475569;
      border-radius: 6px;
      background: #0f172a;
      color: #e2e8f0;
      font-family: ui-monospace, monospace;
      resize: vertical;
    }}
    .output-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>MVS 清单引导线编辑器</h1>
    <span class="hint">拖动圆点调整；重新绘制时按左上、右上、右下、左下依次点击。</span>
  </header>
  <div class="controls">
    <button id="leftButton" class="active">编辑左清单</button>
    <button id="rightButton">编辑右清单</button>
    <button id="redrawButton">重新绘制当前清单</button>
    <button id="resetButton">恢复初始坐标</button>
  </div>
  <div id="viewport">
    <svg id="editor" viewBox="0 0 {width} {height}">
      <image href="/image" width="{width}" height="{height}"/>
      <g id="overlay"></g>
    </svg>
  </div>
  <div class="output">
    <div class="output-row">
      <strong>完整 guideline_coordinates</strong>
      <button id="copyButton">复制坐标</button>
    </div>
    <textarea id="coordinates" readonly></textarea>
  </div>
  <script>
    const width = {width};
    const height = {height};
    const initial = {initial};
    const trailing = {label_guidelines};
    const colors = ["#22c55e", "#38bdf8"];
    let polygons = structuredClone(initial);
    let active = 0;
    let redrawing = false;
    let drag = null;

    const svg = document.getElementById("editor");
    const overlay = document.getElementById("overlay");
    const output = document.getElementById("coordinates");
    const leftButton = document.getElementById("leftButton");
    const rightButton = document.getElementById("rightButton");

    function svgPoint(event) {{
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const transformed = point.matrixTransform(
        svg.getScreenCTM().inverse()
      );
      return [
        Math.max(0, Math.min(width, transformed.x)),
        Math.max(0, Math.min(height, transformed.y)),
      ];
    }}

    function normalized(values) {{
      return values.flatMap(([x, y]) => [
        (x / width).toFixed(4),
        (y / height).toFixed(4),
      ]).join(",");
    }}

    function updateOutput() {{
      output.value = [
        ...polygons.map(normalized),
        ...trailing.map(values => values.map(
          value => Number(value).toFixed(4)
        ).join(",")),
      ].join(";");
    }}

    function render() {{
      overlay.replaceChildren();
      polygons.forEach((points, polygonIndex) => {{
        if (points.length >= 2) {{
          const polygon = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "polygon"
          );
          polygon.setAttribute(
            "points",
            points.map(point => point.join(",")).join(" ")
          );
          polygon.setAttribute("fill", colors[polygonIndex]);
          polygon.setAttribute("stroke", colors[polygonIndex]);
          overlay.appendChild(polygon);
        }}
        points.forEach((point, pointIndex) => {{
          const circle = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "circle"
          );
          circle.setAttribute("cx", point[0]);
          circle.setAttribute("cy", point[1]);
          circle.setAttribute("r", 22);
          circle.setAttribute("fill", colors[polygonIndex]);
          circle.addEventListener("pointerdown", event => {{
            event.preventDefault();
            drag = {{polygonIndex, pointIndex}};
            circle.setPointerCapture(event.pointerId);
          }});
          overlay.appendChild(circle);
        }});
      }});
      updateOutput();
    }}

    function select(index) {{
      active = index;
      redrawing = false;
      leftButton.classList.toggle("active", index === 0);
      rightButton.classList.toggle("active", index === 1);
    }}

    leftButton.addEventListener("click", () => select(0));
    rightButton.addEventListener("click", () => select(1));
    document.getElementById("redrawButton").addEventListener("click", () => {{
      polygons[active] = [];
      redrawing = true;
      render();
    }});
    document.getElementById("resetButton").addEventListener("click", () => {{
      polygons = structuredClone(initial);
      redrawing = false;
      render();
    }});
    document.getElementById("copyButton").addEventListener("click", async () => {{
      await navigator.clipboard.writeText(output.value);
    }});

    window.addEventListener("pointermove", event => {{
      if (!drag) return;
      polygons[drag.polygonIndex][drag.pointIndex] = svgPoint(event);
      render();
    }});
    window.addEventListener("pointerup", () => {{
      drag = null;
    }});
    svg.addEventListener("click", event => {{
      if (!redrawing || polygons[active].length >= 4) return;
      polygons[active].push(svgPoint(event));
      if (polygons[active].length === 4) redrawing = false;
      render();
    }});

    render();
  </script>
</body>
</html>
""".encode("utf-8")


def _handler(image_path: Path, width: int, height: int):
    page = _html(width, height)
    image = image_path.read_bytes()
    image_type = mimetypes.guess_type(image_path.name)[0] or "image/png"

    class GuidelineHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send(page, "text/html; charset=utf-8")
                return
            if self.path == "/image":
                self._send(image, image_type)
                return
            self.send_error(404)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(format % args)

    return GuidelineHandler


def parse_args():
    parser = argparse.ArgumentParser(description="编辑 MVS 双清单引导四边形")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")
    height, width = image.shape[:2]
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler(image_path, width, height),
    )
    print(f"MVS 引导线编辑器: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
