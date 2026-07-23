from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from rsfm_fairness_audit.persistent_cache import (
    PersistentCacheError,
    hydrate_output,
    persist_output,
    validate_storage_contract,
)


WORK = Path("work/test_persistent_cache")


@pytest.fixture(autouse=True)
def _clean_work():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    yield
    if WORK.exists():
        shutil.rmtree(WORK)


def test_persist_and_hydrate_are_non_destructive_and_resume_safe():
    live = WORK / "live"
    persistent = WORK / "persistent"
    (live / "stage").mkdir(parents=True)
    (live / "stage" / "manifest.json").write_text('{"complete": true}', encoding="utf-8")
    assert persist_output(live, persistent, label="unit-test") == 1
    shutil.rmtree(live)
    assert hydrate_output(live, persistent) == 1
    assert (live / "stage" / "manifest.json").read_text(encoding="utf-8") == '{"complete": true}'
    assert hydrate_output(live, persistent) == 0


def test_storage_contract_rejects_same_or_nested_roots():
    with pytest.raises(PersistentCacheError, match="must differ"):
        validate_storage_contract(WORK / "same", WORK / "same")
    with pytest.raises(PersistentCacheError, match="contain"):
        validate_storage_contract(WORK / "live", WORK / "live" / "mirror")
