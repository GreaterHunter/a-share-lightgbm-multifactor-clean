"""The public entry-point documentation must not link to private artifacts."""
from pathlib import Path
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["README.md", "CONTRIBUTING.md", "SECURITY.md",
         "docs/getting_started.md", "docs/architecture.md",
         "docs/data_contract.md", "docs/results.md", "docs/publication_audit.md",
         "paper_futu/README.md", "paper_futu/CSI300_PAPER.md", "paper_futu/S6_CURRENT_DEPLOYMENT.md"]


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
