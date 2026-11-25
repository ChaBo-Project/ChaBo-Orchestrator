"""build_info(): version/commit reported by /health and /eval-query."""
from components.build_info import build_info


def test_release_build_reports_baked_version(monkeypatch):
    monkeypatch.setenv("CHABO_VERSION", "0.9.6")
    monkeypatch.setenv("CHABO_COMMIT", "abc1234")
    assert build_info() == {"version": "0.9.6", "commit": "abc1234"}


def test_other_builds_report_dev(monkeypatch):
    monkeypatch.delenv("CHABO_VERSION", raising=False)
    monkeypatch.setenv("CHABO_COMMIT", "")
    assert build_info() == {"version": "dev", "commit": "unknown"}
