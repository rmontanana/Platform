#!/usr/bin/env python3
"""Build workspace-local Boost.Python/NumPy; never install system packages.

Preparation only: python3 scripts/bootstrap_boost.py --prepare-only
Build: python3 scripts/bootstrap_boost.py --python .venv/bin/python --jobs 2

Shared verbatim by PyClassifiers and Platform: --base selects the workspace-local
directory (build or build_*) that holds the download cache, sources and logs.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = "1.92.0"
ARCHIVE = "boost_1_92_0.tar.bz2"
SOURCE_NAME = "boost_1_92_0"
BASE_URL = f"https://archives.boost.io/release/{VERSION}/source/"
# Published official release metadata, also checked against the HTTPS JSON on
# each invocation. Pinning prevents a changed archive+metadata pair going unnoticed.
SHA256 = "5c1d40cb8e19adbf740a4ec2da35b3e58f3f5804b1dce44deb53df72193cbc6c"
ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.is_symlink():
        raise ValueError(f"Refusing symlink destination: {path}")
    path = path.resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Path must remain strictly inside workspace: {path}")
    first = path.relative_to(ROOT).parts[0]
    if not (first == "build" or first.startswith("build_")):
        raise ValueError(f"Boost outputs must remain under build or build_*: {path}")
    return path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fetch(url, destination=None):
    request = urllib.request.Request(url, headers={"User-Agent": "PyClassifiers-Boost-bootstrap"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if not response.geturl().startswith(BASE_URL):
            raise RuntimeError(f"Unexpected download redirect: {response.geturl()}")
        if destination is None:
            return response.read(1024 * 1024)
        with destination.open("xb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
    return None


def run(command, cwd, log):
    print("+", " ".join(map(str, command)), flush=True)
    with log.open("a", encoding="utf-8") as output:
        output.write("\n+ " + " ".join(map(str, command)) + "\n")
        output.flush()
        subprocess.run(list(map(str, command)), cwd=cwd, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def prepare(base):
    cache = local_path(base / "cache")
    sources = local_path(base / "sources")
    cache.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch(BASE_URL + ARCHIVE + ".json"))
    if metadata.get("file") != ARCHIVE or metadata.get("sha256") != SHA256:
        raise RuntimeError("Official Boost release metadata does not match the pinned identity")
    archive = local_path(cache / ARCHIVE)
    if not archive.exists():
        # Unique partial download: interrupted downloads are never accepted as cache.
        partial = cache / (ARCHIVE + f".{os.getpid()}.partial")
        fetch(BASE_URL + ARCHIVE, partial)
        if digest(partial) != SHA256:
            raise RuntimeError(f"Boost archive checksum mismatch: {partial}")
        partial.rename(archive)
    if digest(archive) != SHA256:
        raise RuntimeError(f"Cached Boost archive checksum mismatch: {archive}")
    source = local_path(sources / SOURCE_NAME)
    marker = local_path(source / ".pyclassifiers-source.json")
    if source.exists():
        if not marker.is_file() or json.loads(marker.read_text()).get("sha256") != SHA256:
            raise RuntimeError(f"Refusing unverified existing source tree: {source}")
    else:
        staging = Path(tempfile.mkdtemp(prefix="boost-extract-", dir=sources))
        with tarfile.open(archive, "r:bz2") as bundle:
            members = bundle.getmembers()
            for member in members:
                parts = Path(member.name).parts
                if not parts or parts[0] != SOURCE_NAME or ".." in parts or Path(member.name).is_absolute():
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
                # Release sources need only ordinary files/directories. Reject
                # links/devices rather than risking symlink-based traversal.
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"Unsupported archive member type: {member.name}")
            if not hasattr(tarfile, "data_filter"):
                raise RuntimeError("Use Python 3.12+ (or a patched Python with tarfile.data_filter)")
            bundle.extractall(staging, members=members, filter="data")
        extracted = staging / SOURCE_NAME
        (extracted / ".pyclassifiers-source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        extracted.rename(source)
        staging.rmdir()  # Only our now-empty staging directory; no existing build is removed.
    local_path(base / "boost-source.json").write_text(json.dumps({
        "version": VERSION, "url": BASE_URL + ARCHIVE,
        "metadata_url": BASE_URL + ARCHIVE + ".json", "sha256": SHA256,
        "official_metadata": metadata, "archive": str(archive), "source": str(source),
    }, indent=2) + "\n")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=".venv/bin/python", help="Python interpreter with NumPy already installed")
    parser.add_argument("--base", default="build_110_deps", help="Workspace-local directory for cache, sources and logs")
    parser.add_argument("--prefix", default="", help="Workspace-local install prefix (default: <base>/boost)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true", help="Download/verify/extract only; do not bootstrap or compile")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    base = local_path(args.base)
    prefix = local_path(args.prefix or base / "boost")
    source = prepare(base)
    print(f"Verified Boost {VERSION} sources: {source}", flush=True)
    if args.prepare_only:
        return
    python = Path(args.python).expanduser()
    if not python.is_absolute():
        python = ROOT / python
    # Preserve the venv path (resolving its symlink would select system Python).
    python = python.absolute()
    if not python.parent.resolve().is_relative_to(ROOT):
        raise RuntimeError("--python must name a workspace-local virtual environment")
    facts_code = """import json,sys,sysconfig,numpy
print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
'version':sys.version.split()[0], 'major':sys.version_info.major,'minor':sys.version_info.minor,
'include':sysconfig.get_path('include'),'libdir':sysconfig.get_config_var('LIBDIR'),
'numpy':numpy.__version__,'numpy_include':numpy.get_include()}))"""
    facts = json.loads(subprocess.check_output([str(python), "-c", facts_code], text=True))
    if facts["major"] != 3 or facts["minor"] != 12 or facts["prefix"] == facts["base_prefix"]:
        raise RuntimeError("This bootstrap requires an activated-by-path Python 3.12 virtual environment with NumPy")
    for path in (str(python), str(prefix)):
        if any(char in path for char in ('"', "\n", "\r", ";")):
            raise RuntimeError("Paths containing Jam metacharacters are unsupported")
    prefix.mkdir(parents=True, exist_ok=True)
    logs = local_path(base / "logs")
    logs.mkdir(exist_ok=True)
    log = local_path(logs / "boost-build.log")
    # B2 otherwise takes whatever gcc/clang is first on PATH. With CXX set (a
    # Conda or module toolchain, for instance) Boost must be built by that same
    # compiler, the one that compiles the project that links against it.
    compiler = os.environ.get("CXX", "").strip()
    toolset = []
    using_compiler = ""
    if compiler:
        resolved = shutil.which(compiler)
        if not resolved or any(char in resolved for char in ('"', "\n", "\r", ";", " ")):
            raise RuntimeError(f"CXX={compiler!r} is not a usable compiler path")
        banner = subprocess.run([resolved, "--version"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True).stdout.lower()
        name = "clang" if "clang" in banner else "gcc"
        toolset = [f"toolset={name}"]
        using_compiler = f'using {name} : : "{resolved}" ;\n'
        print(f"Boost toolset: {name} ({resolved})", flush=True)
    run(["./bootstrap.sh", "--with-libraries=python", f"--with-python={python}",
         "--with-python-version=3.12", f"--prefix={prefix}"]
        + ([f"--with-toolset={toolset[0].split('=')[1]}"] if toolset else []), source, log)
    config = local_path(base / "boost-user-config.jam")
    # B2 otherwise guesses headers beneath sys.prefix (the venv), where the
    # development headers do not live. Homebrew's LIBDIR is inside the framework;
    # python.jam recognizes that ancestry and adds the correct framework flags.
    for value in (facts["include"], facts["libdir"]):
        if not value or any(char in value for char in ('"', '\n', '\r', ';')):
            raise RuntimeError("Unsupported Python header/library path")
    config.write_text(
        using_compiler
        + f'using python : 3.12 : "{python}" : "{facts["include"]}" : "{facts["libdir"]}" ;\n',
        encoding="utf-8")
    run([source / "b2", f"--user-config={config}", f"-j{args.jobs}", *toolset, "--with-python",
         "python=3.12", "cxxstd=17", "variant=release", "link=shared", "threading=multi",
         "local-visibility=hidden",
         "--layout=system", f"--prefix={prefix}", "install"], source, log)
    components = {}
    for component in ("python", "numpy"):
        name = component + "312"
        libraries = sorted(str(p) for p in (prefix / "lib").glob(f"libboost_{name}.*"))
        # BoostConfig maps versioned components (python312/numpy312) to generic
        # boost_python/boost_numpy packages with a Python-specific variant file.
        config_dir = prefix / "lib" / "cmake" / f"boost_{component}-{VERSION}"
        config = config_dir / f"boost_{component}-config.cmake"
        variant = config_dir / f"libboost_{component}-variant-shared-py3.12.cmake"
        if not libraries or not config.is_file() or not variant.is_file():
            raise RuntimeError(f"Missing installed Boost {name} library or CMake variant under {prefix}")
        components[name] = {"libraries": libraries, "configs": [str(config), str(variant)]}
    local_path(base / "boost-build.json").write_text(json.dumps({
        "version": VERSION, "source_sha256": SHA256, "prefix": str(prefix),
        "python": facts, "components": components, "jobs": args.jobs, "log": str(log),
    }, indent=2) + "\n")
    print(f"Installed workspace-local Boost: {prefix}")
    print(f"CMake: -DBoost_DIR={prefix}/lib/cmake/Boost-{VERSION}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise SystemExit(f"Boost bootstrap failed: {error}")
