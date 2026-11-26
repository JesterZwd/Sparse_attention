"""Run the Round 13 workstation stages after the shared GPU becomes idle."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT=Path("results/round13_comrad")
LOG=Path(".cache/round13")


def write_status(status, **details):
    target = LOG / "queue_status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(status=status, pid=os.getpid(), **details)) + "\n")
    temporary.replace(target)


def wait_exit(pids):
    handles=[]
    try:
        for pid in pids:
            try:
                handles.append(os.pidfd_open(pid))
            except ProcessLookupError:
                pass
        if handles:
            select.select(handles,[],[])
    finally:
        for fd in handles:
            os.close(fd)


def wait_idle():
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device and "," not in device:
        command.append(f"--id={device}")

    def active():
        result=subprocess.check_output(command,text=True)
        return [int(pid) for pid in result.split()]

    while True:
        busy=active()
        if busy:
            wait_exit(busy)
        else:
            time.sleep(30)
            if not active():
                return


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid",type=int,help="Optional batch leader already using the shared GPU")
    args=parser.parse_args()
    LOG.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",HF_HUB_OFFLINE="1",
             CUDA_DEVICE_ORDER="PCI_BUS_ID",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (LOG/"gpu_queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (LOG/"queue.pid").write_text(f"{os.getpid()}\n")
        write_status("waiting", wait_pid=args.wait_pid)
        if args.wait_pid is not None:
            wait_exit([args.wait_pid])
        wait_idle()

        failures = {}

        def run(label,module,*params,gpu=True):
            path=LOG/f"queue_{label}.log"
            write_status("running", stage=label, failures=failures)
            command=[sys.executable,"-u","-m",module,*params]
            for attempt in range(2 if gpu else 1):
                with path.open("a") as stream:
                    start = stream.tell()
                    code=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                if not code:
                    return True
                if attempt != 0 or not gpu:
                    break
                with path.open() as stream:
                    stream.seek(start)
                    if "CUDA out of memory" not in stream.read():
                        break
                wait_idle()
            failures[label] = dict(exit_code=code, log=str(path))
            return False

        primary="scripts.run_round13"
        run("qwen3_64k",primary,"qwen3_64k")
        if run("qwen25_smoke","scripts.check_round13_model","--stages","smoke"):
            for phase in ("qwen25_fit","qwen25_cal","qwen25_test"):
                if not run(phase,primary,phase):
                    break
        if run("fwe_screen",primary,"fwe_screen"):
            chosen=json.loads((ROOT/"task_selection.json").read_text())["chosen"]
            ready = chosen == "qwen3_32k_fwe"
            if chosen == "qwen3_32k_cwe":
                ready = run("prepare_cwe","scripts.prepare_ruler","--config",
                            str(ROOT/"prepare_qwen3_32k_cwe.json"),gpu=False)
                ready = ready and run("verify_cwe","scripts.prepare_round13_data","verify-ruler",gpu=False)
            if ready:
                run("fwe_test",primary,"fwe_test")
        run("longbench",primary,"longbench")
        captures = [run(f"capture_{length}","scripts.capture_round13_comrad",length)
                    for length in ("32k","64k")]
        if all(captures):
            run("final_replay","scripts.replay_round13_comrad","--include-new",gpu=False)
        run("quality_report","scripts.report_round13_comrad",gpu=False)
        write_status("incomplete" if failures else "complete", results=str(ROOT), failures=failures)
        if failures:
            raise SystemExit(1)


if __name__=="__main__":
    main()
