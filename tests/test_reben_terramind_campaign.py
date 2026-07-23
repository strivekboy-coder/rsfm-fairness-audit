from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from rsfm_fairness_audit.adapters.terramind import TerraMindAdapter
from rsfm_fairness_audit.reben_terramind_campaign import extract_reben_embeddings_chunked


WORK = Path("work/test_reben_terramind")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


class _Model:
    def eval(self):
        return self

    def extract_embeddings(self, images):
        values = next(iter(images.values()))
        return values.mean(axis=(2, 3))[:, :4]


class _Dataset:
    def __init__(self, count=5):
        self.rows = [
            {
                "sample_id": f"S2A_X_T33TWM_{index:02d}_00",
                "patch_id": f"S2A_X_T33TWM_{index:02d}_00",
                "country": "DE",
                "source_tile_id": "T33TWM",
                "independent_unit_id": f"S2A_X_T33TWM_{index:02d}_00",
            }
            for index in range(count)
        ]

    def load_metadata(self):
        return list(self.rows)

    def load_sample(self, index):
        row = dict(self.rows[index])
        row["label_vector"] = [int(label == index % 19) for label in range(19)]
        return {
            "image": np.full((12, 8, 8), 2000 + index, dtype=np.float32),
            "metadata": row,
        }

    def loader_info(self):
        return {"loader": "unit_test", "split": "train"}


def test_chunked_extraction_is_aligned_and_resume_safe():
    adapter = TerraMindAdapter(
        sensor_mode="S2",
        input_profile="reben_l2a",
        image_size=8,
        strict_range_check=True,
        model=_Model(),
    )
    first = extract_reben_embeddings_chunked(_Dataset(), adapter, WORK / "cache", batch_size=2, chunk_size=3)
    embeddings = np.load(first["embeddings"], mmap_mode="r")
    labels = np.load(first["labels"], mmap_mode="r")
    assert embeddings.shape == (5, 4)
    assert labels.shape == (5, 19)
    del embeddings, labels
    assert len(first["metadata"].read_text(encoding="utf-8").splitlines()) == 5
    second = extract_reben_embeddings_chunked(_Dataset(), adapter, WORK / "cache", batch_size=2, chunk_size=3)
    assert second["manifest"] == first["manifest"]
