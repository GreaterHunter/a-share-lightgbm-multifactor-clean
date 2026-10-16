"""Download and verify a pinned fresh Qlib snapshot without replacing research data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import requests

from src.config import PROJECT_ROOT

TAG = "2026-10-11"
DIRECTORY = PROJECT_ROOT / "data/interim/futu_cn_provider" / TAG
BASE_URL = f"https://github.com/chenditc/investment_data/releases/download/{TAG}"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest_response = requests.get(f"{BASE_URL}/qlib_bin.manifest.json", timeout=30)
    manifest_response.raise_for_status()
    manifest = manifest_response.json()
    if manifest["release_tag"] != TAG or manifest["target_trade_date"] != "2026-10-09":
        raise ValueError("数据快照标识或交易日期不符合预期")
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = DIRECTORY / "qlib_bin.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".gz.part")
        if temporary.exists():
            raise FileExistsError(f"此前的下载未完成，请检查：{temporary}")
        downloaded = 0
        milestone = 0
        with requests.get(f"{BASE_URL}/qlib_bin.tar.gz", stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with temporary.open("xb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - milestone >= 64 * 1024 * 1024:
                        print(f"Downloaded {downloaded / 1024**2:.0f} MiB", flush=True)
                        milestone = downloaded
        if downloaded != int(manifest["archive_size_bytes"]):
            raise ValueError("数据包大小不符")
        temporary.rename(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["archive_sha256"].removeprefix("sha256:"):
        raise ValueError("数据包 SHA256 不匹配")
    extracted = DIRECTORY / "extracted"
    if not (extracted / "complete.json").exists():
        extracted.mkdir(exist_ok=True)
        print("Verified SHA256; extracting snapshot", flush=True)
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            for member in members:
                resolved = (extracted / member.name).resolve()
                if not resolved.is_relative_to(extracted.resolve()) or member.issym() or member.islnk():
                    raise ValueError(f"压缩包存在不允许的路径或链接：{member.name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"压缩包存在非数据文件：{member.name}")
            source.extractall(extracted, members=members, filter="data")
        (extracted / "complete.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Fresh provider extracted: {extracted}", flush=True)


if __name__ == "__main__":
    main()
