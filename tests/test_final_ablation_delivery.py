import os
from pathlib import Path

import pytest

from paper_futu.final_ablation_core import digest
from scripts import finalize_final_ablation_delivery as delivery


@pytest.fixture
def workspace(tmp_path):
    report,data,models = [tmp_path/p for p in ["report","data","models"]]
    for path in [report,data,models,report/"logs",report/"monitor",report/"parallel_training"]:
        path.mkdir(parents=True,exist_ok=True)
    (report/"research_report.html").write_text("stable result")
    (data/"frozen.parquet").write_bytes(b"frozen data")
    (models/"model.txt").write_text("stable model")
    return tmp_path,report,data,models


def test_mutable_files_classified_without_fake_hashes(workspace):
    root,report,data,models = workspace
    for path in [report/"logs/run.log",report/"monitor/latest.json",report/"parallel_training/controller.lock",
                 report/"pipeline_status.json",report/"model.txt.tmp"]:
        path.write_text("mutable")
    m = delivery.build_manifest(root,report,data,models)
    assert set(m["files"]) == {"report/research_report.html","data/frozen.parquet","models/model.txt"}
    assert m["excluded"] and all("sha256" not in r for r in m["excluded"])
    assert any(r["kind"]=="directory_prefix" and r["path"]=="report/monitor" for r in m["excluded"])


def test_monitor_and_logs_can_change_without_invalidating_delivery(workspace):
    root,report,data,models = workspace
    m = delivery.build_manifest(root,report,data,models)
    (report/"monitor/new_observation.json").write_text("later observation")
    (report/"logs/later.log").write_text("new log")
    (report/"pipeline_status.json").write_text("complete")
    assert delivery.verify_manifest(m,root,inventory_roots=(report,data,models))==3


def test_unknown_static_permission_error_is_not_silently_skipped(workspace):
    root,report,data,models = workspace
    unknown = report/"unknown_static.bin"
    unknown.write_bytes(b"important")
    def checked_hash(path):
        if Path(path)==unknown:
            raise PermissionError(13,"static read denied")
        return digest(path)
    with pytest.raises(PermissionError,match="static read denied"):
        delivery.build_manifest(root,report,data,models,hash_file=checked_hash)


def test_static_change_and_unknown_new_static_file_require_new_manifest(workspace):
    root,report,data,models = workspace
    m = delivery.build_manifest(root,report,data,models)
    (report/"final_review.html").write_text("new immutable analysis")
    with pytest.raises(ValueError,match="inventory changed"):
        delivery.verify_manifest(m,root,inventory_roots=(report,data,models))
    (report/"final_review.html").unlink()
    (models/"model.txt").write_text("changed model")
    with pytest.raises(ValueError,match="artifact changed"):
        delivery.verify_manifest(m,root)


def test_historical_incomplete_task_is_classified_not_promoted(workspace):
    root,report,data,models = workspace
    (report/"tasks").mkdir()
    (report/"tasks/old_failed.json").write_text('{"status":"failed"}')
    m = delivery.build_manifest(root,report,data,models,incomplete_ids={"old_failed"})
    assert "report/tasks/old_failed.json" not in m["files"]
    assert m["incomplete_historical_task_ids"]==["old_failed"]
    assert any("not promoted" in r["reason"] for r in m["excluded"])


def test_parent_static_review_is_included_and_verified(workspace):
    root,report,data,models = workspace
    (report/"final_review.html").write_text("independent review")
    (report/"final_review.json").write_text('{"checked":true}')
    m = delivery.build_manifest(root,report,data,models)
    assert "report/final_review.html" in m["files"] and "report/final_review.json" in m["files"]
    assert delivery.verify_manifest(m,root,inventory_roots=(report,data,models))==5


@pytest.mark.skipif(os.name!="nt",reason="Windows mandatory byte-range locking behavior")
def test_controller_byte_lock_reproduces_digest_permission_error_then_releases(tmp_path):
    lock = tmp_path/"controller.lock"
    result = delivery.locked_digest_probe(lock)
    assert result["reproduced"] and result["errno"]==13
    assert result["read_after_lock_release_sha256"]==digest(lock)
