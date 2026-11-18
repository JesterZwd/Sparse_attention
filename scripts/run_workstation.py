"""Run a logged GPU smoke test and/or 16K exploratory Minimal Probe."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    checkpoint = root / "models/Qwen3-4B-Instruct-2507"
    source = json.loads((checkpoint / "source_manifest.json").read_text())
    if not source.get("verified"):
        raise RuntimeError("finish SHA-256 verified model download first")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not visible")
    results = root / "results"
    results.mkdir(exist_ok=True)
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                       device_properties=str(torch.cuda.get_device_properties(0)),
                       driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                       "--format=csv,noheader"], text=True).strip())
    (results / "environment.json").write_text(json.dumps(environment, indent=2))
    stages = ["smoke", "pilot"] if args.stage == "all" else [args.stage]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1")
    for stage in stages:
        name = "smoke_4k" if stage == "smoke" else "pilot_16k"
        output = results / name
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        commands = [
            [sys.executable, "-m", "gqa_probe.capture", "--model", str(checkpoint),
             "--output", str(output), "--context", "4096" if stage == "smoke" else "16384",
             "--samples", "1" if stage == "smoke" else "2", "--decode-steps",
             "2" if stage == "smoke" else "4", "--layers",
             "0,17,35" if stage == "smoke" else "0,7,14,21,28,35", "--groups",
             "0" if stage == "smoke" else "all"],
            [sys.executable, "-m", "gqa_probe.evaluate", str(output / "traces"),
             "--output", str(output / "evaluation")],
        ]
        started = time.monotonic()
        with (results / f"{name}.log").open("x") as log:
            for command in commands:
                log.write(json.dumps({"command": command}) + "\n")
                log.flush()
                with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"{stage} command failed with exit {code}; see {log.name}")
        print(f"{stage} complete in {time.monotonic() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
