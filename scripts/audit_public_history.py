"""Read-only Git history screening. Never print candidate secret values.

This is a bounded pattern/inventory audit, not proof that a repo has no secrets.
Only reachable refs are scanned; remote caches, forks and untracked files are not.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "cloud_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openai_token": re.compile(r"\bsk-(?:proj-|svcacct-)[A-Za-z0-9_-]{30,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential_url": re.compile(r"https?://[^\s/:@]+:[^\s/@]{4,}@"),
    "assigned_secret": re.compile(r'''(?i)(?:password|passwd|api_key|api_secret|access_token|refresh_token)\s*["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']'''),
    "full_account_id": re.compile(r'''(?i)(?:acc_id|account_id)\s*["']?\s*[:=]\s*["']?\d{7,}'''),
    "personal_home_path": re.compile(r"(?i)[A-Z]:[/\\]+Users[/\\]+[A-Za-z0-9_.-]+"),
    "workspace_path": re.compile(r"(?i)[A-Z]:[/\\]+(?:Internship|Deep_Learning)[/\\]+"),
}


def inspect_text(text: str) -> list[dict]:
    findings = []
    for category, pattern in PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            findings.append({"category": category, "count": len(matches),
                             "lines": sorted({text.count('\n', 0, m.start()) + 1 for m in matches})})
    return findings


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def audit(root: Path, refs: str = "--all") -> dict:
    paths = {}
    for line in git(root, "rev-list", "--objects", refs).decode("utf-8").splitlines():
        oid, _, path = line.partition(" ")
        paths[oid] = path
    request = ("\n".join(paths) + "\n").encode("ascii")
    meta = subprocess.run(["git", "-C", str(root), "cat-file", "--batch-check"],
                          input=request, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    blobs = [(oid, int(size)) for oid, kind, size in
             (line.split() for line in meta.stdout.decode().splitlines()) if kind == "blob"]
    findings, binary, oversized, sensitive_paths = [], [], [], []
    text_count = 0
    for oid, size in blobs:
        path = paths[oid]
        basename = Path(path).name.lower()
        if (basename == ".env" or basename.endswith((".pem", ".key", ".p12", ".pfx"))
                or path in {"configs/futu_cn_paper.json", "configs/experiments/s6_head6_40_research_main.json"}
                or (path.startswith(("data/", "models/", "reports/paper_futu/"))
                    and basename != ".gitkeep")):
            sensitive_paths.append({"path": path, "blob": oid, "bytes": size})
        if size > 20 * 1024 * 1024:
            oversized.append({"path": path, "blob": oid, "bytes": size})
            continue
        data = git(root, "cat-file", "blob", oid)
        try:
            text = data.decode("utf-8-sig")
            if "\0" in text:
                raise UnicodeError("binary")
        except UnicodeError:
            binary.append({"path": path, "blob": oid, "bytes": size})
            continue
        text_count += 1
        findings.extend({"path": path, "blob": oid, **item} for item in inspect_text(text))
    authors = git(root, "log", refs, "--format=%ae%n%ce").decode("utf-8").splitlines()
    return {"scope": refs, "head": git(root, "rev-parse", "HEAD").decode().strip(),
            "commits": int(git(root, "rev-list", "--count", refs)),
            "blobs": len(blobs), "text_blobs_scanned": text_count,
            "total_blob_bytes": sum(size for _, size in blobs),
            "finding_counts": dict(Counter(item["category"] for item in findings)),
            "findings": findings, "sensitive_paths": sensitive_paths,
            "binary_requires_review": binary, "oversized_not_scanned": oversized,
            "unique_commit_emails": len(set(authors)),
            "non_noreply_commit_emails": len({a for a in authors if a and "noreply" not in a.lower()}),
            "limitation": "Heuristic screening, not a security certification; no secret values printed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-only", action="store_true", help="Scan HEAD ancestry instead of all refs; still includes history")
    parser.add_argument("--summary", action="store_true", help="Omit per-blob details, retaining counts and inventory")
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], "HEAD" if args.head_only else "--all")
    if args.summary:
        result["finding_locations"] = sorted({item["path"] for item in result.pop("findings")})
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
