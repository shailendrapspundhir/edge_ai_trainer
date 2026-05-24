"""Thin wrapper around the `adb` CLI.

We don't take a hard dep on `adbutils` because most operations are trivially
shell-able and we want this module to import even when the extras aren't
installed. If `adbutils` is present, we'll use it for a couple of richer
operations (file pull, port forward) — but every method here has a shell
fallback.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eat.config import get_settings
from eat.logging_setup import get_logger

log = get_logger("android.adb")


@dataclass(frozen=True)
class Device:
    serial: str
    state: str
    model: str | None = None
    android_version: str | None = None


def _adb(args: Iterable[str], serial: str | None = None, check: bool = True,
         timeout: int = 60) -> subprocess.CompletedProcess:
    bin_ = get_settings().adb_bin
    cmd = [bin_]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    log.debug("adb_call", cmd=" ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)


def list_devices() -> list[Device]:
    out = _adb(["devices", "-l"], check=False).stdout
    devs: list[Device] = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        serial, state = parts[0], parts[1]
        model = None
        for p in parts[2:]:
            if p.startswith("model:"):
                model = p.split(":", 1)[1]
        ver = None
        try:
            ver_out = _adb(["shell", "getprop", "ro.build.version.release"], serial=serial,
                           check=False, timeout=10).stdout.strip()
            ver = ver_out or None
        except Exception:  # noqa: BLE001
            pass
        devs.append(Device(serial=serial, state=state, model=model, android_version=ver))
    return devs


def first_serial() -> str:
    s = get_settings().android_device_serial
    if s:
        return s
    devs = [d for d in list_devices() if d.state == "device"]
    if not devs:
        raise RuntimeError("no Android device connected (adb returned none in state 'device')")
    return devs[0].serial


def push(local: Path, remote: str, serial: str | None = None) -> None:
    _adb(["push", str(local), remote], serial=serial)


def pull(remote: str, local: Path, serial: str | None = None) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    _adb(["pull", remote, str(local)], serial=serial)
    return local


def shell(cmd: str, serial: str | None = None, timeout: int = 60) -> str:
    return _adb(["shell", cmd], serial=serial, timeout=timeout).stdout


def install_apk(apk: Path, serial: str | None = None, replace: bool = True) -> None:
    args = ["install"]
    if replace:
        args.append("-r")
    args.append(str(apk))
    _adb(args, serial=serial, timeout=180)


def uninstall(package: str, serial: str | None = None) -> None:
    _adb(["uninstall", package], serial=serial, check=False)


def remote_mkdir(path: str, serial: str | None = None) -> None:
    _adb(["shell", f"mkdir -p {shlex.quote(path)}"], serial=serial)


def start_intent(
    component: str,
    action: str,
    extras: dict[str, str],
    serial: str | None = None,
    wait: bool = True,
    timeout: int = 600,
) -> str:
    """`am start --wait -n <component> -a <action> [--es <k> <v>]...`"""
    args = ["shell", "am", "start"]
    if wait:
        args.append("--wait")
    args += ["-n", component, "-a", action]
    for k, v in extras.items():
        args += ["--es", k, shlex.quote(v)]
    return _adb(args, serial=serial, timeout=timeout).stdout
