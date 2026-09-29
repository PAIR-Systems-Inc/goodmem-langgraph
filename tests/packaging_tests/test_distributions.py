"""Check the built wheel independently of the source checkout."""

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_distribution_contains_only_the_shared_entry_points(tmp_path: Path) -> None:
    wheel = next(Path("dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        sources = {name for name in archive.namelist() if name.endswith(".py")}
        assert sources == {
            "goodmem_langgraph/__init__.py",
            "goodmem_langgraph/tools/__init__.py",
        }
        assert "goodmem_langgraph/py.typed" in archive.namelist()
        archive.extractall(tmp_path)
    code = (
        "import pathlib,sys; sys.path.insert(0,sys.argv[1]); "
        "import goodmem_langgraph,goodmem_langchain; "
        "assert pathlib.Path(goodmem_langgraph.__file__).is_relative_to(sys.argv[1]); "
        "assert goodmem_langgraph.GoodMemRetriever is goodmem_langchain.GoodMemRetriever"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)], cwd=tmp_path, check=True
    )
    with tarfile.open(next(Path("dist").glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert any(name.endswith("/tests/unit_tests/test_graphs.py") for name in names)
        assert any(name.endswith("/CHANGELOG.md") for name in names)
        assert not any(
            "/.venv/" in name or name.endswith("/_client.py") for name in names
        )
