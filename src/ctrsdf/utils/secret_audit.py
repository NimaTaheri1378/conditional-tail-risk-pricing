from __future__ import annotations

import argparse
import re
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[^'\"\s]{8,}"),
    re.compile(r"(?i)fred[_-]?api"),
    re.compile(r"(?i)bls[_-]?api"),
    re.compile(r"(?i)bea[_-]?api"),
    re.compile(r"(?i)eia[_-]?api"),
]

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "data", "artifacts", "tests"}
ALLOW_FILES = {".env.example", "DATA_ACCESS.md", ".gitignore", "secret_audit.py"}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def audit(root: Path) -> list[str]:
    findings: list[str] = []
    for path in iter_files(root):
        if path.name in ALLOW_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(str(path.relative_to(root)))
                break
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    findings = audit(Path(args.root).resolve())
    if findings:
        print("Potential secret references found:")
        for finding in findings:
            print(f"  {finding}")
        raise SystemExit(1)
    print("Secret audit passed.")


if __name__ == "__main__":
    main()
