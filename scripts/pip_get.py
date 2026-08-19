#!/usr/bin/env python3
"""Install PyPI packages when system ``pip`` HTTPS/SSL is broken.

Uses urllib + certifi (proven working on this machine) to download wheels from
pypi.org, then ``pip install --no-deps`` from local files under D:\\ml_cache\\pip_wheels.

Example:
  python scripts/pip_get.py peft accelerate
  python scripts/pip_get.py peft --with-deps
"""
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

# Avoid broken env proxies for this downloader (system IWR may work via proxy;
# Python urllib3 SSL through 127.0.0.1:7897 is broken here).
for _k in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy",
):
    os.environ.pop(_k, None)

try:
    import certifi
except ImportError:
    print("need certifi", file=sys.stderr)
    sys.exit(1)

WHEEL_DIR = Path(os.environ.get("ML_CACHE_ROOT", r"D:\ml_cache")) / "pip_wheels"


def _opener():
    ctx = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ctx),
    )


def get_bytes(url: str, timeout: int = 120) -> bytes:
    with _opener().open(url, timeout=timeout) as r:
        return r.read()


def pypi_json(pkg: str) -> dict:
    return json.loads(get_bytes(f"https://pypi.org/pypi/{pkg}/json").decode())


def pick_wheel(files: list, py_tag: str = "cp310") -> dict | None:
    wheels = [f for f in files if f.get("packagetype") == "bdist_wheel"]
    if not wheels:
        return None

    def score(f: dict) -> int:
        n = f["filename"]
        s = 0
        if "py3-none-any" in n or "py2.py3-none-any" in n:
            s += 100
        if py_tag in n:
            s += 50
        if "win_amd64" in n:
            s += 20
        if "win32" in n:
            s -= 5
        if "manylinux" in n or "macosx" in n:
            s -= 30
        return s

    wheels.sort(key=score, reverse=True)
    best = wheels[0]
    if score(best) < 0:
        return None
    return best


def download_pkg(pkg: str, version: str | None = None) -> Path:
    info = pypi_json(pkg)
    ver = version or info["info"]["version"]
    files = info["releases"].get(ver) or []
    if not files:
        raise SystemExit(f"no release {pkg}=={ver}")
    pick = pick_wheel(files)
    if pick is None:
        # sdist fallback
        sdists = [f for f in files if f.get("packagetype") == "sdist"]
        if not sdists:
            raise SystemExit(f"no wheel/sdist for {pkg}=={ver}")
        pick = sdists[0]
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = WHEEL_DIR / pick["filename"]
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"cached {dest}")
        return dest
    print(f"download {pkg}=={ver} → {pick['filename']}")
    dest.write_bytes(get_bytes(pick["url"]))
    print(f"saved {dest} ({dest.stat().st_size} bytes)")
    return dest


_REQ_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")


def required_names(pkg: str) -> list[str]:
    info = pypi_json(pkg)
    out = []
    for r in info["info"].get("requires_dist") or []:
        if "extra ==" in r:
            continue
        m = _REQ_NAME.match(r.strip())
        if m:
            out.append(m.group(1))
    return out


def pip_install_local(path: Path) -> None:
    cmd = [sys.executable, "-m", "pip", "install", str(path), "--no-deps"]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("packages", nargs="+", help="PyPI package names")
    ap.add_argument("--with-deps", action="store_true", help="also fetch declared deps")
    ap.add_argument("--version", default=None, help="pin version for the first package only")
    args = ap.parse_args()

    todo = list(args.packages)
    if args.with_deps:
        extra = []
        for p in list(todo):
            extra.extend(required_names(p))
        # de-dup preserve order
        seen = set()
        merged = []
        for p in todo + extra:
            k = p.lower().replace("_", "-")
            if k not in seen:
                seen.add(k)
                merged.append(p)
        todo = merged

    paths = []
    for i, pkg in enumerate(todo):
        ver = args.version if i == 0 else None
        paths.append(download_pkg(pkg, ver))

    for p in paths:
        pip_install_local(p)

    print("done:", ", ".join(todo))


if __name__ == "__main__":
    main()
