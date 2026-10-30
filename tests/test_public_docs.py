"""The public entry-point documentation must not link to private artifacts."""
from pathlib import Path
import json
import re
import csv
import hashlib
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["README.md", "CONTRIBUTING.md", "SECURITY.md",
         "docs/getting_started.md", "docs/architecture.md",
         "docs/data_contract.md", "docs/results.md", "docs/publication_audit.md",
         "docs/research_report.md", "docs/history/annual_top50_report.md",
         "reports/current_strategy/README.md", "reports/main_strategy/README.md",
         "reports/current_strategy/total_return/README.md", "docs/prospective_s6.md",
         "paper_futu/README.md", "paper_futu/CSI300_PAPER.md", "paper_futu/S6_CURRENT_DEPLOYMENT.md"]
PAGES += ["docs/achievements.md", "docs/current_paper_strategy.md",
          "docs/artifact_retention.md", "reports/README.md", "reports/research/barra/README.md"]


def test_published_research_artifact_hashes():
    for manifest in (ROOT / "reports/research").glob("*/evidence.json"):
        evidence = json.loads(manifest.read_text(encoding="utf-8"))
        for item in evidence["artifacts"]:
            name = item.get("file", item.get("path"))
            expected = item.get("public_sha256", item.get("sha256"))
            assert Path(name).name == name
            assert hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest() == expected


def test_public_documentation_links_resolve():
    missing = []
    for name in PAGES:
        page = ROOT / name
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            path = unquote(target.split("#", 1)[0])
            if not (page.parent / path).exists():
                missing.append(f"{name}: {target}")
    assert not missing, "Broken public links: " + "; ".join(missing)


def test_demo_does_not_depend_on_broker_packages():
    text = (ROOT / "examples/offline_demo.py").read_text(encoding="utf-8")
    assert "paper_futu" not in text
    assert "SYNTHETIC ONLY" in text


def test_public_parameters_are_not_deployment_bindings():
    path = ROOT / "configs/experiments/s6_head6_40_public.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["portfolio"]["top_k"] == 20
    assert config["retrain_months"] == 3
    assert not {"account_suffix", "model_path", "effective_signal_date", "first_execution_date"} & config.keys()
    assert not re.search(r"[A-Za-z]:[\\/]", path.read_text(encoding="utf-8"))


def test_public_financial_evidence_is_marked_as_derived():
    config = json.loads((ROOT / "docs/evidence/financial_coverage.json").read_text(encoding="utf-8"))
    assert "Derived public copy" in config["publication_note"]
    assert all(path.startswith("data/") for path in config["source_sha256"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in config["source_sha256"].values())


def test_total_return_public_evidence_is_portable_and_sanitized():
    report = ROOT / "reports/current_strategy/total_return"
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    daily = ROOT / "reports/current_strategy/phase_daily.csv"
    assert hashlib.sha256(daily.read_bytes()).hexdigest() == manifest["account_daily_sha256"]
    assert not re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", json.dumps(manifest))
    private_columns = {"account_id", "acc_id", "account_suffix", "instrument", "order_id", "model_path"}
    for name, expected in manifest["artifacts"].items():
        assert Path(name).name == name
        path = report / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                assert not private_columns.intersection(next(csv.reader(handle)))
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "reports/current_strategy/total_return/* -text" in attributes
