"""Download Qwen's official ModelScope checkpoint and verify every SHA-256."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_file(item, destination, url):
    if destination.exists() and destination.stat().st_size == item["Size"]:
        if digest(destination) == item["Sha256"]:
            print(f"verified existing {destination.name}", flush=True)
            return
    temporary = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(4):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset >= item["Size"]:
                if offset == item["Size"] and digest(temporary) == item["Sha256"]:
                    temporary.replace(destination)
                    print(f"verified complete partial {destination.name}", flush=True)
                    return
                temporary.unlink()
                offset = 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with opener.open(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                total = offset if append else 0
                last_report = total
                with temporary.open("ab" if append else "wb") as stream:
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
                        total += len(block)
                        if total - last_report >= 256 * 1024 * 1024:
                            print(f"{destination.name}: {total / 1e9:.2f}/{item['Size'] / 1e9:.2f} GB", flush=True)
                            last_report = total
            if temporary.stat().st_size != item["Size"] or digest(temporary) != item["Sha256"]:
                temporary.unlink()
                raise ValueError(f"size/SHA-256 mismatch: {destination.name}")
            temporary.replace(destination)
            print(f"verified {destination.name} ({item['Size']} bytes)", flush=True)
            return
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                temporary.unlink(missing_ok=True)
            print(f"attempt {attempt + 1} {destination.name}: {type(error).__name__}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/qwen3-4b-modelscope-files.json"))
    parser.add_argument("--output", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    args = parser.parse_args()
    metadata = json.loads(args.manifest.read_text())
    if not metadata.get("Success"):
        raise ValueError("unsuccessful ModelScope metadata response")
    data = metadata["Data"]
    files = [f for f in data["Files"] if f["Type"] == "blob" and not f["Path"].startswith(".")]
    # Latest commit ID is omitted by this API; the most recently modified file records it.
    revision = max(files, key=lambda f: f["CommittedDate"])["Revision"]
    args.output.mkdir(parents=True, exist_ok=True)
    source = dict(repository="Qwen/Qwen3-4B-Instruct-2507", service="modelscope.cn",
                  snapshot_revision=revision, files=files, verified=False)
    source_path = args.output / "source_manifest.json"
    source_path.write_text(json.dumps(source, indent=2))

    def download(item):
        relative = Path(item["Path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid repository path")
        destination = args.output / relative
        query = urllib.parse.urlencode({"Revision": revision, "FilePath": item["Path"]})
        url = "https://modelscope.cn/api/v1/models/Qwen/Qwen3-4B-Instruct-2507/repo?" + query
        download_file(item, destination, url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))
    source["verified"] = True
    source_path.write_text(json.dumps(source, indent=2))
    print(f"all checkpoint files verified: {args.output}", flush=True)


if __name__ == "__main__":
    main()
