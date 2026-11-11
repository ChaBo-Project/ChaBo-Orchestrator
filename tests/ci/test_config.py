from components.utils import getconfig


def test_inline_comments_are_stripped_from_values(tmp_path, monkeypatch):
    monkeypatch.delenv("INSTANCE_CONFIG_DIR", raising=False)
    cfg = tmp_path / "params.cfg"
    cfg.write_text(
        "[generator]\n"
        "INFERENCE_PROVIDER = novita   # which provider\n"
        "docs_url = https://host/page#anchor\n",
        encoding="utf-8",
    )

    config = getconfig(str(cfg))

    assert config.get("generator", "INFERENCE_PROVIDER") == "novita"
    assert config.get("generator", "docs_url") == "https://host/page#anchor"
