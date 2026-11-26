"""Run same-resource A/B/M replay with the two measured full frontends."""
import argparse
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

from scripts.replay_round12 import analyze, digest

RESULTS=Path("results")
R11=RESULTS/"round11_workpoints"
R12=RESULTS/"round12_frontend"
R13=RESULTS/"round13_comrad"


def restore_round12_trace(path, spec, root):
    """Materialize the published Round 12 32K archive on a fresh checkout."""
    if path.is_file():
        if digest(path) != spec["reassembled_sha256"]:
            raise ValueError(f"archive checksum differs: {path}")
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for name, expected in sorted(spec["parts"].items()):
                part = root / name
                if digest(part) != expected:
                    raise ValueError(f"archive part checksum differs: {part}")
                with part.open("rb") as source:
                    copyfileobj(source, stream)
            stream.flush()
            if digest(temporary) != spec["reassembled_sha256"]:
                raise ValueError("reassembled Round 12 archive checksum differs")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=R13/"replay")
    p.add_argument("--include-new",action="store_true")
    p.add_argument("--workers",type=int,default=5)
    args=p.parse_args()
    publication = json.loads((R12/"verification.json").read_text())["publication"]
    restore_round12_trace(R12/"32k/extended/states.jsonl.gz", publication, R12)
    sets={
        "archived_round11":[R11/k/"common/states.jsonl.gz" for k in ("32k","64k")],
        "expanded_round12":[R12/k/"extended/states.jsonl.gz" for k in ("32k","64k")],
    }
    if args.include_new:
        sets["new_round13"]=[R13/k/"common/states.jsonl.gz" for k in ("32k","64k")]
    src=R13/"hardware/frontend_calibration.json"
    for group in sets.values():
        for file in group:
            receipt=json.loads((file.parent/"complete.json").read_text())
            expected=receipt.get("state_sha256") or receipt["states_sha256"]
            if expected!=digest(file):
                raise ValueError(f"physical trace checksum differs at {file}")
    archives={f"archived_round11/{k}":R11/k/"replay/step_timing.csv" for k in ("32k","64k")}
    analyze(sets,args.output,calibration=src,archives=archives,
            policies=("A","B","M"),workers=args.workers)


if __name__=="__main__":
    main()
