"""Download the pinned official Windows OpenD archive into the project cache."""

from pathlib import Path
import hashlib

import requests


URL = "https://softwaredownload.futunn.com/Futu_OpenD_10.11.7108_Windows.7z"
EXPECTED_SIZE = 249932432
DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "data/interim/futu_opend_10_11/Futu_OpenD_10.11.7108_Windows.7z"
)


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        if DESTINATION.stat().st_size != EXPECTED_SIZE:
            raise ValueError("Existing archive has an unexpected size; it was not replaced")
        print(f"Archive already downloaded: {DESTINATION}", flush=True)
        return
    temporary = DESTINATION.with_suffix(".7z.part")
    if temporary.exists():
        raise FileExistsError(f"An earlier incomplete download exists: {temporary}")
    digest = hashlib.sha256()
    downloaded = 0
    milestone = 0
    with requests.get(URL, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        if int(response.headers.get("content-length", "0")) != EXPECTED_SIZE:
            raise ValueError("Official archive size changed; inspect the response before saving")
        with temporary.open("xb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if downloaded == 0 and not chunk.startswith(b"7z\xbc\xaf\x27\x1c"):
                    raise ValueError("Response is not a 7-Zip archive")
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded - milestone >= 32 * 1024 * 1024:
                    print(f"Downloaded {downloaded / 1024 ** 2:.0f} MiB", flush=True)
                    milestone = downloaded
    if downloaded != EXPECTED_SIZE:
        raise ValueError("Incomplete OpenD download")
    if DESTINATION.exists():
        raise FileExistsError("Destination appeared during download; it was not replaced")
    temporary.rename(DESTINATION)
    print(f"Archive: {DESTINATION}", flush=True)
    print(f"SHA256: {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
