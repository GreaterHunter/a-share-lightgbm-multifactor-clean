"""One explicit, audited Windows disk_usage compatibility migration; no science changes."""
from __future__ import annotations

import ast
import copy
import hashlib
import time

from paper_futu import research_s6_2026 as study


def main():
    import psutil
    for proc in psutil.process_iter(["pid","cmdline"]):
        if "paper_futu.research_s6_2026" in (proc.info["cmdline"] or []):
            raise RuntimeError("Wait until original controller and workers have exited")
    path = study.OUT / "protocol.json"
    old = study.read(path)
    current = (study.ROOT / "paper_futu/research_s6_2026.py").read_text(encoding="utf-8")
    after = "completed=completed,total=len(names),free_gib=free,disk_free_gib=psutil.disk_usage(str(ROOT)).free/2**30,"
    before = "completed=completed, total=len(names), free_gib=free, disk_free_gib=psutil.disk_usage(ROOT).free/2**30,"
    if current.count(after) != 1:
        raise ValueError("Migration applies only to the reviewed disk_usage path fix")
    original = current.replace(after,before)
    files = {"research_s6_2026.py":hashlib.sha256(original.encode("utf-8")).hexdigest(),
        **{name:study.digest(study.ROOT / "paper_futu" / name) for name in ("s6_2026_account.py","s6_2026_report.py")}}
    if study.identity(files) != old["implementation_sha256"]:
        raise ValueError("Original implementation cannot be reconstructed exactly")
    def scientific_tree(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name != "execute_queue"]
        return ast.dump(tree,include_attributes=False)
    if scientific_tree(original) != scientific_tree(current):
        raise ValueError("Scientific code outside scheduler changed")
    original_protocol_hash = study.digest(path)
    records = []
    for task_path in (study.OUT / "tasks").glob("*.json"):
        task = study.read(task_path)
        if task["status"] != "complete" or task["kind"] != "fit":
            raise ValueError("Migration only supports the completed initial fit cache")
        if task["implementation_sha256"] != old["implementation_sha256"] or task["protocol_sha256"] != original_protocol_hash:
            raise ValueError("Unexpected prior cache binding")
        study.verify_artifacts(task)
        if task["result"]["label_end_max"] > task["result"]["cutoff"]:
            raise ValueError("Cached fit crossed its cutoff")
        records.append((task_path,task))
    audit_before = study.read(study.OUT / "audit.json")
    archive = study.OUT / "compatibility" / f"scheduler_{time.time_ns()}"
    study.save(archive / "protocol_before.json",old)
    study.save(archive / "audit_before.json",audit_before)
    new = copy.deepcopy(old)
    new["implementation_sha256"] = study.implementation()
    study.save(path,new)
    for task_path,task in records:
        study.save(archive / task_path.name,task)
        task["implementation_sha256"] = new["implementation_sha256"]
        task["protocol_sha256"] = study.digest(path)
        task["compatibility_review"] = str(archive / "review.json")
        study.save(task_path,task)
    audit_before["protocol_sha256"] = study.digest(path)
    study.save(study.OUT / "audit.json",audit_before)
    study.save(archive / "review.json",dict(status="passed",reason="psutil Windows requires str path",
        old_implementation_sha256=old["implementation_sha256"],new_implementation_sha256=new["implementation_sha256"],
        original_protocol_sha256=original_protocol_hash,new_protocol_sha256=study.digest(path),
        scientific_ast_unchanged=True,account_and_report_sources_unchanged=True,
        checked_complete_fits=len(records),artifacts_unchanged=True))
    study.check_sources()
    print(f"COMPATIBILITY PASSED: {len(records)} completed fits retained")


if __name__ == "__main__":
    main()
