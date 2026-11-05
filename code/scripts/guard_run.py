"""Runs one job of a queue so that a failed job always ends and the queue can print its STOP (CLAUDE.md 13.8: "Stop and report on any failed self-test or run").

Why it exists. E3 fold 1 failed at 21:43:16 on 4 Nov 2026 with a CUDA out-of-memory error. Python finished, but the process stayed in ExitProcess, with its exit code set and one thread spinning in the CUDA runtime's DLL detach (nvcuda64.dll), so the queue waited on it and printed no STOP. Stop-Process and taskkill had no effect; TerminateProcess, through CIM, ended it at 23:05 and the GPU memory came back (CHANGES row 117). os._exit would not help: it goes through the same detach.

What it does. It starts the job with this same interpreter (python <arguments after -->), with stdout and stderr appended to --log and stdin empty, and waits. Nothing else about the job changes: same working directory, same environment. It ends the job with TerminateProcess in two cases, and says so on its own stdout and in the log:
  1. The exit code has been set for --exit-limit seconds (60) and the process has not ended. Python has finished by then, so no file is left half written. It returns that exit code. Normal exits took 0.8-4.3 s after run_log.md (68 runs).
  2. The job is still running and the log has not grown for --quiet-limit seconds (3600). It returns 1. A run writes one line per epoch; the longest silence on record is its inference and scoring, about 15 minutes.
If the process still has not ended 120 s after TerminateProcess, the guard says so and returns anyway, with a code that cannot read as success, so that the queue is never left waiting and never starts the next job beside it.
A hand kill while the queue lives must pass a reason: Invoke-CimMethod -Query 'select * from Win32_Process where ProcessId=N' -MethodName Terminate -Arguments @{Reason=[uint32]1}. Without one the job's exit code becomes 0, even after a failure; the queue's check of run_log.md catches that. If a later move of the run folder is refused, look for python.exe with --multiprocessing-fork and the job's parent_pid: a killed job's DataLoader workers can outlive it for a while and hold the log open.

Exit code: the job's own, or 1 when that code would read as 0 or be out of range for the shell (a shell keeps the code modulo 256, so 256 would read as success).
Run from work_ms/: python scripts/guard_run.py --log out/stdout.txt -- -W ignore -m itareo.ugi_seg.train --cfg ... --fold 1 --seed 0 --out out
"""
import argparse
import os
import subprocess
import sys
import time

try:
    import _winapi
except ImportError:  # the tests of this file run on Windows; elsewhere only the plain wait is available
    _winapi = None

GRACE_AFTER_TERMINATE = 120


def say(f, text):
    line = f"guard: {text} {time.strftime('%F %T')}"
    try:
        print(line, flush=True)
        f.write((line + "\n").encode("utf-8"))
        f.flush()
    except OSError:  # a full disk or a closed pipe must not stop the guard from ending the job
        pass


def end(p, f, code, why):
    try:
        _winapi.TerminateProcess(p._handle, code)
    except PermissionError:  # it ended in that instant, as subprocess.Popen.terminate also allows
        pass
    say(f, f"{why}; TerminateProcess")
    try:
        return p.wait(timeout=GRACE_AFTER_TERMINATE)
    except subprocess.TimeoutExpired:
        say(f, f"the process has not ended {GRACE_AFTER_TERMINATE} s after TerminateProcess; returning anyway")
    return code or 1


def guarded(cmd, log, exit_limit, quiet_limit, poll):
    with open(log, "ab") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        size, last_growth, exit_seen = os.fstat(f.fileno()).st_size, time.monotonic(), None
        while True:
            try:
                return p.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                pass
            if _winapi is None:
                continue
            now = time.monotonic()
            code = _winapi.GetExitCodeProcess(p._handle)
            if code != _winapi.STILL_ACTIVE:
                exit_seen = now if exit_seen is None else exit_seen
                if now - exit_seen >= exit_limit:
                    return end(p, f, code, f"exit code {code} set, process {p.pid} not ended after {now - exit_seen:.0f} s")
                continue
            s = os.fstat(f.fileno()).st_size
            if s != size:
                size, last_growth = s, now
            elif now - last_growth >= quiet_limit:
                return end(p, f, 1, f"process {p.pid} still running, log silent for {now - last_growth:.0f} s")


def shell_code(rc):
    return rc if 0 <= rc < 256 else 1


def main():
    if "--" not in sys.argv:
        raise SystemExit("guard_run.py: usage: --log FILE [--exit-limit S] [--quiet-limit S] [--poll S] -- <python arguments>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--exit-limit", type=float, default=60.0)
    ap.add_argument("--quiet-limit", type=float, default=3600.0)
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(sys.argv[1:i])
    job = sys.argv[i + 1:]
    if not job:
        raise SystemExit("guard_run.py: nothing to run")
    rc = guarded([sys.executable, *job], a.log, a.exit_limit, a.quiet_limit, a.poll)
    raise SystemExit(shell_code(rc))


if __name__ == "__main__":
    main()
