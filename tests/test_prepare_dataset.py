import json

import pandas as pd

from scripts.prepare_dataset import read_goemotions, write_outputs


def test_goemotions_reader_excludes_multilabel_rows(tmp_path):
    source = tmp_path / "ekman"
    source.mkdir()
    (source / "labels.txt").write_text(
        "anger\ndisgust\nfear\njoy\nneutral\nsadness\nsurprise\n", encoding="utf-8"
    )
    for name in ("train.tsv", "dev.tsv", "test.tsv"):
        (source / name).write_text(
            "single joy\t3\tid-one\nambiguous\t3,4\tid-two\n", encoding="utf-8"
        )
    frame = read_goemotions(source)
    assert len(frame) == 3
    assert set(frame["label"]) == {"joy"}
    assert not frame["id"].str.contains("id-two").any()


def test_dataset_manifest_contains_stable_checksums(tmp_path):
    rows = []
    for split in ("train", "validation", "test"):
        rows.append(
            {"id": f"{split}-1", "text": "sample", "label": "neutral", "language": "en", "split": split}
        )
    write_outputs(pd.DataFrame(rows), tmp_path, ["unit-test"], 42)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "dataset_v1"
    assert set(manifest["sha256"]) == {"train.csv", "validation.csv", "test.csv"}
