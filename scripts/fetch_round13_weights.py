"""Fetch and verify the official Round 13 Qwen2.5-7B-Instruct BF16 snapshot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha1, sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

REPO = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
HF = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
MIRROR_API = "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-7B-Instruct/repo/files?Revision=master&Root="
MIRROR = "https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct/resolve/master"
DESTINATION = Path("models/Qwen2.5-7B-Instruct")
WORK = Path(".cache/round13/models")
OUTPUT = Path("results/round13_comrad/model_adaptation/weight_availability.json")
METADATA = ("config.json", "generation_config.json", "merges.txt", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE", "README.md")


def read_json_url(url, local):
    if not local.exists():
        with urllib.request.urlopen(url, timeout=30) as source:
            data = source.read()
        local.write_bytes(data)
    return json.loads(local.read_text())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def exact(path, row):
    return (path.is_file() and path.stat().st_size == row["size"] and
            digest(path) == row["lfs"]["sha256"])


def git_blob(path, row):
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    if len(data) != row["size"] or sha1(header + data).hexdigest() != row["blobId"]:
        raise ValueError(f"metadata differs from source Git blob: {path.name}")
    return sha256(data).hexdigest()


def publish_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.with_suffix(".tmp")
    draft.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    draft.replace(path)


def download_range(job):
    name, start, end, size, part, mirror_ok = job
    length = end - start + 1
    if part.is_file() and part.stat().st_size == length:
        return
    headers = part.with_suffix(".headers")
    temporary = part.with_suffix(".attempt")
    expected = f"{start}-{end}/{size}"
    for attempt in range(8):
        host = MIRROR if mirror_ok and attempt % 2 else HF
        url = f"{host}/{name}?r13range={start}-{end}&attempt={attempt}"
        result = subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error", "--http1.1",
            "--connect-timeout", "15", "--max-time", "100", "--speed-time", "30",
            "--speed-limit", "300000", "--max-filesize", str(length), "--range", f"{start}-{end}",
            "--dump-header", str(headers), "--output", str(temporary), url,
        ], capture_output=True, text=True, check=False)
        content_ranges = re.findall(r"^content-range:\s*bytes\s+(\S+)",
                                    headers.read_text(errors="replace") if headers.exists() else "",
                                    re.IGNORECASE | re.MULTILINE)
        if (not result.returncode and expected in content_ranges and temporary.exists() and
                temporary.stat().st_size == length):
            temporary.replace(part)
            return
    raise RuntimeError(f"range {name} {expected} failed after eight bounded attempts: {result.stderr.strip()}")


def assemble(row, prefix, parts, destination):
    name = row["rfilename"]
    path = destination / name
    temporary = path.with_suffix(".assembling")
    with temporary.open("wb") as writer:
        for source in ([prefix] if prefix is not None else []) + parts:
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    if not exact(temporary, row):
        raise ValueError(f"assembled bytes for {name} failed official SHA256 verification")
    temporary.replace(path)
    print(f"Verified {name} against the official LFS SHA256", flush=True)


def fetch(destination, work, workers, chunk_size):
    destination.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    manifest = read_json_url(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
                             work / "qwen2_5_manifest.json")
    if manifest["sha"] != REVISION or manifest["id"] != REPO or manifest.get("gated"):
        raise ValueError("unexpected official model revision or access status")
    rows = {file["rfilename"]:file for file in manifest["siblings"]}
    metadata_hashes = {}
    for name in METADATA:
        target = destination / name
        if not target.exists():
            urllib.request.urlretrieve(f"{HF}/{name}", target)
        metadata_hashes[name] = git_blob(target, rows[name])
    try:
        mirror = read_json_url(MIRROR_API, work / "qwen2_5_modelscope.json")
        mirrors = {row["Name"]:row for row in mirror["Data"]["Files"]}
    except (OSError, KeyError, ValueError):
        mirrors = {}
    shards = [rows[name] for name in sorted(rows) if name.endswith(".safetensors")]
    if len(shards) != 4:
        raise ValueError("snapshot does not have four expected safetensors shards")
    jobs, plans = [], []
    for file in shards:
        name, length = file["rfilename"], file["size"]
        path = destination / name
        if exact(path, file):
            continue
        prefix = max((p for p in destination.glob(f"{name}.partial-*") if 0 < p.stat().st_size <= length),
                     key=lambda p:p.stat().st_size, default=None)
        if prefix is not None and prefix.stat().st_size == length:
            if exact(prefix, file):
                prefix.replace(path)
                continue
            prefix = None
        offset = prefix.stat().st_size if prefix is not None else 0
        parts = []
        for start in range(offset, length, chunk_size):
            end = min(length, start + chunk_size) - 1
            part = work / f"{name}.{start}.{end}.verified_range"
            jobs.append((name, start, end, length, part,
                         mirrors.get(name, {}).get("Sha256") == file["lfs"]["sha256"]))
            parts.append(part)
        plans.append((file, prefix, parts))
    # Oscillating between shards avoids keeping a single CDN connection busy indefinitely.
    jobs.sort(key=lambda job: (job[1] // chunk_size, job[0]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download_range, jobs))
    for row, prefix, parts in plans:
        assemble(row, prefix, parts, destination)
    hashes = {**metadata_hashes, **{row["rfilename"]:row["lfs"]["sha256"] for row in shards}}
    sources = {row["rfilename"]:row["size"] for row in shards}
    receipt = dict(repository=REPO, service="Hugging Face", revision=REVISION, verified=True,
                   files=[dict(Path=name, Sha256=value) for name, value in sorted(hashes.items())],
                   weights_format="safetensors", weight_shard_bytes=sources,
                   mirror_verified_shards=[name for name in sources if
                                           mirrors.get(name, {}).get("Sha256") == hashes[name]])
    publish_json(destination / "source_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DESTINATION)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--range-mib", type=int, default=64)
    args = parser.parse_args()
    if args.range_mib < 1 or args.workers < 1:
        parser.error("workers and range length must be positive")
    qwen = fetch(args.model, args.work, args.workers, args.range_mib * 1024**2)
    llama_release = read_json_url("https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct",
                                  args.work / "llama_manifest.json")
    llama_rev = llama_release["sha"]
    meta_url = f"https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/{llama_rev}/config.json"
    status = subprocess.run(["curl", "--silent", "--location", "--max-time", "20", "--output", "/dev/null",
                             "--write-out", "%{http_code}", meta_url], capture_output=True, text=True, check=True)
    publish_json(args.output, dict(qwen25=qwen, llama31=dict(repository="meta-llama/Llama-3.1-8B-Instruct",
        revision=llama_rev, gated=llama_release["gated"], anonymous_config_status=int(status.stdout),
        local_weight_directory_present=Path("models/Llama-3.1-8B-Instruct").is_dir())))


if __name__ == "__main__":
    main()
