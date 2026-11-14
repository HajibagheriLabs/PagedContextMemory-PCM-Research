"""configs/assets.yaml is well-formed and the PG-19 loader refuses the train split."""
import pytest

from pcm.hub import load_assets, missing_files
from pcm.tasks import pg19


def test_assets_yaml_has_required_entries():
    cfg = load_assets()
    for key in ("reader", "reader_base", "reader_dev", "embedder"):
        assert cfg["models"][key].startswith("Qwen/")
    assert cfg["expected_geometry"]["kv_bytes_per_token_bf16"] == 114_688
    assert cfg["datasets"]["babilong"]["lengths"][0] == "0k"
    assert cfg["datasets"]["babilong"]["lengths"][-1] == "256k"
    assert set(cfg["datasets"]["pg19"]["splits"]) == {"validation", "test"}


def test_pg19_train_split_is_refused():
    with pytest.raises(ValueError):
        pg19.split_files("train")
    with pytest.raises(ValueError):
        next(pg19.stream_books("train"))


def test_missing_files_reports_absent_and_empty(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty.json").write_text("", encoding="utf-8")
    assert missing_files(tmp_path, ["a.json", "empty.json", "nope.json"]) == ["empty.json", "nope.json"]
