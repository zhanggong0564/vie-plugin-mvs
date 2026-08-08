import runpy
from pathlib import Path

from Cython import Build
import setuptools


def test_plugin_uses_binary_wheel_build_contract(monkeypatch):
    plugin_root = Path(__file__).resolve().parents[1]
    setup_kwargs = {}
    cython_kwargs = {}

    def fake_setup(**kwargs):
        setup_kwargs.update(kwargs)

    def fake_cythonize(sources, **kwargs):
        cython_kwargs.update(kwargs)
        return sources

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setattr(Build, "cythonize", fake_cythonize)
    monkeypatch.chdir(plugin_root)

    setup_globals = runpy.run_path(
        str(plugin_root / "setup.py"), run_name="__build_contract__"
    )

    sources = {Path(source).name for source in setup_kwargs["ext_modules"]}
    expected = {
        source.name
        for source in (plugin_root / "vie_plugin_mvs").glob("*.py")
        if source.name != "__init__.py"
    }
    assert sources == expected
    assert cython_kwargs == {
        "build_dir": "build",
        "compiler_directives": {
            "language_level": "3",
            "annotation_typing": False,
        },
    }
    assert setup_kwargs["cmdclass"]["build_py"] is setup_globals[
        "BuildPyInitOnly"
    ]
