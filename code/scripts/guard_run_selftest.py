"""Self-test of scripts/guard_run.py, CPU only. A stand-in for the real hang: a small DLL (scripts/guard_spin_dll.py writes it into --out) whose detach handler spins in user mode for a bounded time, so that the process keeps its exit code set and does not end, as process 44928 did inside nvcuda64.dll.
Cases: the stuck failure and a stuck clean exit (rule 1 names itself, the job has ended, the traceback is in the log); a stuck exit below the limit, left alone; the same with a DataLoader with persistent workers, whose workers go by Python's own shutdown before the guard acts; the quiet rule with live workers (they must go after the kill and the folder must move); a plain exit code; an exit code that a shell would read as 0; a silent job (rule 2 names itself); a job that keeps printing past the quiet limit; a log with content kept; TerminateProcess on a process that has just ended; a process that does not end within the grace; and a call from bash with empty stdin.
Writes <out>/guard_selftest.json; exit 1 if a case fails. From work_ms/: python scripts/guard_run_selftest.py --out <new folder>
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = HERE / "guard_run.py"
RULE1, RULE2 = "exit code", "log silent"


def job_script(mode, dll):
    head = f"import ctypes, os, sys, time\nctypes.WinDLL(r'{dll}')\nprint('pid', os.getpid(), flush=True)\n"
    loader = "import ctypes, os, sys, time\nimport torch\nclass D(torch.utils.data.Dataset):\n    def __len__(self):\n        return 64\n    def __getitem__(self, i):\n        return torch.zeros(4)\n"
    return {
        "stuck_failure": head + "raise RuntimeError('stand-in failure')\n",
        "stuck_clean": head + "print('finished', flush=True)\n",
        "plain3": "import os, sys\nprint('pid', os.getpid(), flush=True)\nprint('three', flush=True)\nsys.exit(3)\n",
        "code256": "import os, sys\nprint('pid', os.getpid(), flush=True)\nsys.exit(256)\n",
        "silent": "import os, time\nprint('pid', os.getpid(), flush=True)\ntime.sleep(600)\n",
        "chatty": "import os, time\nprint('pid', os.getpid(), flush=True)\nfor i in range(9):\n    print(i, flush=True)\n    time.sleep(1)\n",
        "workers": loader + f"if __name__ == '__main__':\n    ctypes.WinDLL(r'{dll}')\n    dl = torch.utils.data.DataLoader(D(), batch_size=2, num_workers=2, persistent_workers=True)\n    print('pid', os.getpid(), flush=True)\n"
                   "    for i, b in enumerate(dl):\n        if i == 3:\n            raise RuntimeError('stand-in failure in the loop')\n",
        "quiet_workers": loader + "if __name__ == '__main__':\n    dl = torch.utils.data.DataLoader(D(), batch_size=2, num_workers=2, persistent_workers=True)\n    it = iter(dl)\n    next(it)\n    print('pid', os.getpid(), flush=True)\n    time.sleep(600)\n",
    }[mode]


def job_pid(text):
    m = re.search(r"^pid (\d+)", text, re.M)
    return int(m.group(1)) if m else None


def process_gone(pid, wait_s=10):
    t0 = time.time()
    while time.time() - t0 < wait_s:
        o = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], stdout=subprocess.PIPE, text=True)
        if o.returncode == 0 and f'"{pid}"' not in o.stdout:
            return True
        time.sleep(0.5)
    return False


def workers_gone(pid, wait_s=60):
    """No python.exe with parent_pid=<pid> in its command line, within wait_s; a failed query is not gone."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        o = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"], stdout=subprocess.PIPE, text=True)
        if o.returncode == 0 and f"parent_pid={pid}" not in o.stdout:
            return True, round(time.time() - t0, 1)
        time.sleep(2)
    return False, wait_s


def run_case(out, name, mode, dll, expect_rc, rule, max_s, extra=("--exit-limit", "3", "--quiet-limit", "3", "--poll", "0.5"), traceback=False, seed_log=None):
    d = out / name
    d.mkdir()
    (d / "job.py").write_text(job_script(mode, dll), encoding="utf-8")
    log = d / "stdout.txt"
    if seed_log:
        log.write_text(seed_log, encoding="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(GUARD), "--log", str(log), *extra, "--", str(d / "job.py")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300, cwd=d)
    sec = time.time() - t0
    text = log.read_text(encoding="utf-8", errors="replace")
    pid = job_pid(text)
    res = {"rc": r.returncode, "seconds": round(sec, 1), "guard_stdout": r.stdout.strip()[-300:], "log_tail": text[-300:], "job_pid": pid}
    acted = "guard:" in r.stdout
    tests = {"rc": r.returncode == expect_rc, "seconds": sec <= max_s, "guard_acted_as_expected": acted == bool(rule), "guard_line_in_log": ("guard:" in text) == bool(rule), "pid_seen": pid is not None}
    if rule:
        tests["rule_named"] = rule in r.stdout
    tests["job_ended"] = process_gone(pid) if pid else False
    if traceback:
        tests["traceback_in_log"] = "Traceback" in text
    if seed_log:
        tests["old_content_kept"] = text.startswith(seed_log)
    res["tests"] = tests
    res["passed"] = all(tests.values())
    return res, d, pid


def units(out):
    """end() on a process that has just ended, and on one that does not end within the grace."""
    spec = importlib.util.spec_from_file_location("guard_run_unit", GUARD)
    g = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(g)
    res = {}
    d = out / "units"
    d.mkdir()
    with open(d / "log.txt", "ab") as f:
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        try:
            res["already_ended_returns"] = g.end(p, f, 0, "unit: already ended")
            res["already_ended_raised"] = False
        except Exception as e:
            res["already_ended_raised"] = repr(e)
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

        class NoKill:
            STILL_ACTIVE = g._winapi.STILL_ACTIVE
            GetExitCodeProcess = staticmethod(g._winapi.GetExitCodeProcess)
            TerminateProcess = staticmethod(lambda h, c: None)
        real, grace = g._winapi, g.GRACE_AFTER_TERMINATE
        g._winapi, g.GRACE_AFTER_TERMINATE = NoKill, 1
        try:
            res["grace_exit_code_0_returns"] = g.end(p, f, 0, "unit: does not end")
        finally:
            g._winapi, g.GRACE_AFTER_TERMINATE = real, grace
            p.terminate()
            p.wait()
    res["passed"] = res["already_ended_raised"] is False and res["already_ended_returns"] == 0 and res["grace_exit_code_0_returns"] == 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dll = out / "spin.dll"
    subprocess.run([sys.executable, str(HERE / "guard_spin_dll.py"), str(dll), "2.4e11"], check=True, stdout=subprocess.DEVNULL)
    short = out / "spin_short.dll"
    subprocess.run([sys.executable, str(HERE / "guard_spin_dll.py"), str(short), "2.4e10"], check=True, stdout=subprocess.DEVNULL)
    # the stand-in's spin length without any guard: it must far outlast the limit used below
    cal = out / "calibrate"
    cal.mkdir()
    (cal / "job.py").write_text(job_script("stuck_failure", dll), encoding="utf-8")
    t0 = time.time()
    subprocess.run([sys.executable, str(cal / "job.py")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    spin_s = round(time.time() - t0, 1)
    res = {"spin_seconds_without_guard": spin_s}
    ok = spin_s > 20
    for name, mode, rc, rule, max_s, tb in (("stuck_failure", "stuck_failure", 1, RULE1, 15, True), ("stuck_clean", "stuck_clean", 0, RULE1, 15, False), ("plain3", "plain3", 3, None, 10, False), ("code256", "code256", 1, None, 10, False), ("silent", "silent", 1, RULE2, 15, False), ("chatty", "chatty", 0, None, 15, False)):
        r, _, _ = run_case(out, name, mode, dll, rc, rule, max_s, traceback=tb)
        res[name] = r
        ok = ok and r["passed"]
    # a stuck exit below the limit is left alone
    r, _, _ = run_case(out, "stuck_below_limit", "stuck_failure", short, 1, None, 40, extra=("--exit-limit", "60", "--quiet-limit", "60", "--poll", "0.5"), traceback=True)
    res["stuck_below_limit"] = r
    ok = ok and r["passed"]
    # a log with content is appended to
    r, _, _ = run_case(out, "log_kept", "plain3", dll, 3, None, 10, seed_log="an older line\n")
    res["log_kept"] = r
    ok = ok and r["passed"]
    # failure inside a DataLoader loop: the workers go by Python's own shutdown, then the exit sticks, then rule 1
    r, d, pid = run_case(out, "workers", "workers", dll, 1, RULE1, 120, extra=("--exit-limit", "3", "--quiet-limit", "90", "--poll", "0.5"), traceback=True)
    gone, t_gone = workers_gone(pid) if pid else (False, 0)
    r["tests"]["workers_gone"] = gone
    r["workers_gone_after_s"] = t_gone
    try:
        shutil.move(str(d), str(out / "workers_moved"))
        r["tests"]["folder_moved"] = True
    except OSError as e:
        r["tests"]["folder_moved"] = False
        r["move_error"] = str(e)
    r["passed"] = all(r["tests"].values())
    res["workers"] = r
    ok = ok and r["passed"]
    # the quiet rule with live workers: the job prints once and goes silent; after the kill the workers must go and the folder must move
    r, d, pid = run_case(out, "quiet_workers", "quiet_workers", dll, 1, RULE2, 120, extra=("--exit-limit", "3", "--quiet-limit", "8", "--poll", "0.5"))
    gone, t_gone = workers_gone(pid) if pid else (False, 0)
    r["tests"]["workers_gone"] = gone
    r["workers_gone_after_s"] = t_gone
    try:
        shutil.move(str(d), str(out / "quiet_workers_moved"))
        r["tests"]["folder_moved"] = True
    except OSError as e:
        r["tests"]["folder_moved"] = False
        r["move_error"] = str(e)
    r["passed"] = all(r["tests"].values())
    res["quiet_workers"] = r
    ok = ok and r["passed"]
    r = units(out)
    res["units"] = r
    ok = ok and r["passed"]
    # from bash, as the queue calls it
    d = out / "from_bash"
    d.mkdir()
    (d / "job.py").write_text(job_script("plain3", dll), encoding="utf-8")
    py = sys.executable.replace("\\", "/")
    bash = shutil.which("bash")  # PATH order, not CreateProcess's, which would find the WSL bash in System32 first
    o = subprocess.run([bash, "-c", f"cd '{d.as_posix()}' && '{py}' -W ignore '{GUARD.as_posix()}' --log stdout.txt -- -W ignore job.py < /dev/null; echo rc=$?"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    r = {"bash": bash, "stdout": o.stdout.strip(), "log": (d / "stdout.txt").read_text(encoding="utf-8", errors="replace").strip() if (d / "stdout.txt").exists() else None}
    r["passed"] = "rc=3" in o.stdout and "three" in (r["log"] or "")
    res["from_bash"] = r
    ok = ok and r["passed"]
    res["passed"] = bool(ok)
    res["finished"] = time.strftime("%F %T")
    (out / "guard_selftest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
