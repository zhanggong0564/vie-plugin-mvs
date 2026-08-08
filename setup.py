"""二进制 wheel 构建（通用，自动探测本插件包名）。

cython 把业务模块编译为 .so，wheel 仅含 __init__.py + .so + 元数据，不落明文业务源码。
元数据（name/version/entry-points/dependencies）由 pyproject.toml [project] 提供。
"""
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

# 自动定位本插件唯一的 vie_plugin_* 包目录
PKG = next(p.name for p in Path(__file__).parent.iterdir()
           if p.is_dir() and p.name.startswith("vie_plugin_"))

# 保留 __init__.py 为纯 py，其余业务模块编译成 .so（排除 build/ 中间产物）
py_sources = [str(p) for p in Path(PKG).rglob("*.py")
              if p.name != "__init__.py" and "build" not in p.parts]


class BuildPyInitOnly(build_py):
    """只把 __init__.py 作为源码打入 wheel；其余 .py 已编成 .so，剔除以防明文泄露。"""

    def find_package_modules(self, package, package_dir):
        return [m for m in super().find_package_modules(package, package_dir) if m[1] == "__init__"]


setup(
    # build_dir 把 .c 写到 build/；annotation_typing=False 关闭 Cython3 注解类型强制
    # （否则 FastAPI Form()/pydantic 字段注解冲突报 "Expected str, got Form"）
    ext_modules=cythonize(
        py_sources, build_dir="build",
        compiler_directives={"language_level": "3", "annotation_typing": False},
    ),
    cmdclass={"build_py": BuildPyInitOnly},
)
