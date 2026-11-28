#!/usr/bin/env python3
"""Provide a workspace-local CPython with headers and libpython; never touch the system.

Prints the interpreter path on stdout (progress goes to stderr):

    python3 scripts/bootstrap_python.py --base build_110_deps

The interpreter is a uv-managed standalone CPython installed under <base>/python.
uv is a pinned and checksummed release downloaded to <base>/uv (or the binary
named by the UV environment variable). Nothing is written outside <base>.

This file is shared verbatim by PyClassifiers and Platform and only needs the
Python that runs the Makefile (>=3.6): it must not import anything of the project.
"""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

PYTHON_VERSION = "3.12"
UV_VERSION = "0.13.7"
UV_URL = "https://github.com/astral-sh/uv/releases/download/" + UV_VERSION + "/"
# sha256 of the official release archives of UV_VERSION.
UV_SHA256 = {
    "aarch64-apple-darwin": "81b811b57e4624df7ed3aac148e0359100d9962f3d9dce00aec533dc2e086731",
    "x86_64-apple-darwin": "20c47e4c1f630b8abaff8c291fa9b9aa300fef92c4fbff95adf198b55287cda7",
    "x86_64-unknown-linux-gnu": "c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83",
    "aarch64-unknown-linux-gnu": "71279f8d3c67241e807b9d8db872d1bf541579fb9f08b7d1eb1682196d1e57dc",
}
ROOT = Path(__file__).resolve().parents[1]


def log(message):
    print(message, file=sys.stderr, flush=True)


def local_base(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if ROOT not in path.parents:
        raise RuntimeError("Path must remain strictly inside the workspace: %s" % path)
    first = path.relative_to(ROOT).parts[0]
    if not (first == "build" or first.startswith("build_")):
        raise RuntimeError("Bootstrap outputs must remain under build or build_*: %s" % path)
    return path


def triple():
    machine = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}.get(
        platform.machine().lower())
    system = {"darwin": "apple-darwin", "linux": "unknown-linux-gnu"}.get(sys.platform)
    if not machine or not system:
        raise RuntimeError("No pinned uv release for %s/%s; install uv and put it on the PATH"
                           % (sys.platform, platform.machine()))
    return machine + "-" + system


def digest(path):
    checksum = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(url, destination):
    log("Downloading " + url)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "pyclassifiers-bootstrap"})
        with urllib.request.urlopen(request, timeout=120) as response, open(str(destination), "wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as error:  # e.g. an interpreter without CA certificates
        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("Download failed (%s) and curl is not available" % error)
        subprocess.check_call([curl, "--fail", "--silent", "--show-error", "--location",
                               "--max-time", "300", "--output", str(destination), url])


def ensure_uv(base):
    # A uv found on the PATH is deliberately ignored: its version decides which
    # CPython patch release is installed, and every machine must get the same one.
    selected = os.environ.get("UV")
    if selected:
        return selected
    directory = base / "uv"
    binary = directory / "uv"
    if binary.is_file():
        return str(binary)
    target = triple()
    name = "uv-" + target + ".tar.gz"
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / (name + ".%d.partial" % os.getpid())
    try:
        download(UV_URL + name, partial)
        if digest(partial) != UV_SHA256[target]:
            raise RuntimeError("uv archive checksum mismatch: " + name)
        with tarfile.open(str(partial), "r:gz") as bundle:
            member = bundle.getmember("uv-" + target + "/uv")
            if not member.isfile():
                raise RuntimeError("Unexpected uv archive layout")
            staging = directory / ("uv.%d.partial" % os.getpid())
            with bundle.extractfile(member) as source, open(str(staging), "wb") as output:
                shutil.copyfileobj(source, output)
        os.chmod(str(staging), 0o755)
        os.replace(str(staging), str(binary))
    finally:
        if partial.exists():
            partial.unlink()
    log("uv %s installed in %s" % (UV_VERSION, directory))
    return str(binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="build_110_deps", help="Workspace-local directory for uv and CPython")
    args = parser.parse_args()
    base = local_base(args.base)
    base.mkdir(parents=True, exist_ok=True)
    uv = ensure_uv(base)
    env = dict(os.environ)
    env.update({
        "UV_PYTHON_INSTALL_DIR": str(base / "python"),
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON_INSTALL_BIN": "0",  # no launcher in ~/.local/bin
        "UV_NO_CONFIG": "1",
        "UV_NO_PROGRESS": "1",
    })
    # An active venv or conda environment (or a .venv in the directory) would be
    # returned by "uv python find" instead of the managed interpreter.
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(name, None)
    log("+ %s python install %s  (into %s)" % (uv, PYTHON_VERSION, base / "python"))
    subprocess.check_call([uv, "python", "install", PYTHON_VERSION], env=env, stdout=sys.stderr)
    python = subprocess.check_output([uv, "python", "find", "--system", PYTHON_VERSION], env=env,
                                     universal_newlines=True).strip()
    if not python or base not in Path(python).resolve().parents:
        raise RuntimeError("uv did not return a workspace-local CPython %s: %r" % (PYTHON_VERSION, python))
    facts = subprocess.check_output([python, "-c", (
        "import os,sys,sysconfig;"
        "print('%d.%d' % sys.version_info[:2]);"
        "print(os.path.join(sysconfig.get_path('include'), 'Python.h'));"
        "print(os.path.join(sysconfig.get_config_var('LIBDIR') or '', sysconfig.get_config_var('LDLIBRARY') or ''))"
    )], universal_newlines=True).split("\n")
    if facts[0] != PYTHON_VERSION:
        raise RuntimeError("Expected CPython %s, found %s" % (PYTHON_VERSION, facts[0]))
    for required in facts[1:3]:
        if not os.path.isfile(required):
            raise RuntimeError("The managed CPython lacks a development file: " + required)
    log("CPython ready: " + python)
    print(python)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise SystemExit("Python bootstrap failed: %s" % error)
