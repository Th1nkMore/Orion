from pathlib import Path

from scripts.audit_stage2l_v10_view_binding import _literal_assignment


def test_literal_assignment_reads_tuple(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("ORDER = ('a', 'b')\n", encoding="utf-8")
    assert _literal_assignment(source, "ORDER") == ("a", "b")
