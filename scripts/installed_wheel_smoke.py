#!/usr/bin/env python3
"""Smoke-test an installed bookextract wheel."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: installed_wheel_smoke.py <wheel-path>", file=sys.stderr)
        return 2

    wheel_path = Path(args[0]).resolve()
    if not wheel_path.is_file():
        print(f"wheel not found: {wheel_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = venv_dir / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", str(wheel_path)], check=True)
        script = """
import json
from importlib import resources
import bookextract
from bookextract.schema import load_wire_schema

schema = load_wire_schema()
assert isinstance(schema, dict)
defaults = resources.files("bookextract.resources").joinpath("pandoc-epub-v1.json").read_text()
parsed = json.loads(defaults)
assert parsed["from"] == "markdown+footnotes"
print("wheel-smoke-ok")
"""
        result = subprocess.run([str(python), "-c", script], check=True, capture_output=True, text=True)
        if "wheel-smoke-ok" not in result.stdout:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
    print(f"wheel smoke passed for {wheel_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
