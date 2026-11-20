import configparser

from config_render import apply_override


def test_override_replaces_values_case_insensitively_and_keeps_percent(tmp_path):
    rendered = tmp_path / "params.override.cfg"
    rendered.write_text("[retriever]\nTOP_K = 5\nPATTERN = 100%\n", encoding="utf-8")
    override = tmp_path / "override.cfg"
    override.write_text("[retriever]\ntop_k = 20\n", encoding="utf-8")

    swept = apply_override(rendered, override)

    result = configparser.ConfigParser(interpolation=None)
    result.read(rendered)
    assert result.get("retriever", "top_k") == "20"
    assert result.get("retriever", "pattern") == "100%"
    assert swept == {"retriever": {"top_k": "20"}}