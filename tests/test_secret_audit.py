from pathlib import Path

from ctrsdf.utils.secret_audit import audit


def test_secret_audit_flags_password(tmp_path: Path):
    (tmp_path / "x.txt").write_text("password = supersecretvalue", encoding="utf-8")
    assert audit(tmp_path) == ["x.txt"]
