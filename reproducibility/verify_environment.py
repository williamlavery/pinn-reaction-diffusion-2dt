from __future__ import annotations

import hashlib
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_PYTHON = "3.11.15"
EXPECTED_PYTHON_PACKAGES = {
    "pysr": "1.5.10",
    "juliacall": "0.9.26",
    "juliapkg": "0.1.23",
    "numpy": "2.4.3",
    "sympy": "1.14.0",
    "torch": "2.5.1",
}
EXPECTED_JULIA = "1.12.6"
EXPECTED_JULIA_PACKAGES = {
    "SymbolicRegression": "1.11.3",
    "DynamicExpressions": "1.10.4",
    "PythonCall": "0.9.26",
}
EXPECTED_MANIFEST_SHA256 = (
    "b3be093a7a41477aadf8b26d48870303dff58908b677c8643fbae152dc1f9a05"
)


def installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


failures: list[str] = []


def check(label: str, actual: str, expected: str) -> None:
    status = "OK" if actual == expected else "MISMATCH"
    print(f"[{status:8}] {label}: {actual} (expected {expected})")
    if actual != expected:
        failures.append(f"{label}: got {actual!r}, expected {expected!r}")


print("Environment executable:", sys.executable)
print("Platform:", platform.platform())
print()

check("Python", platform.python_version(), EXPECTED_PYTHON)
for package_name, expected in EXPECTED_PYTHON_PACKAGES.items():
    check(package_name, installed_version(package_name), expected)

# Import PySR only after reporting the Python package layer. This initializes
# the Julia runtime selected by PYTHON_JULIAPKG_EXE/PROJECT.
from pysr import jl  # noqa: E402

print()
check("Julia", str(jl.seval("string(VERSION)")), EXPECTED_JULIA)

for package_name, expected in EXPECTED_JULIA_PACKAGES.items():
    actual = str(
        jl.seval(
            "using Pkg; "
            f'info = only(filter(p -> p.second.name == "{package_name}", '
            "collect(Pkg.dependencies()))); "
            "string(info.second.version)"
        )
    )
    check(f"{package_name}.jl", actual, expected)

project = Path(str(jl.seval("Base.active_project()"))).resolve()
expected_project = Path(os.environ["PYTHON_JULIAPKG_PROJECT"]).resolve() / "Project.toml"
check("Active Julia project", str(project), str(expected_project))

manifest = project.with_name("Manifest.toml")
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
check("Julia Manifest SHA-256", manifest_hash, EXPECTED_MANIFEST_SHA256)

print("Julia executable:", jl.seval("joinpath(Sys.BINDIR, Base.julia_exename())"))
print("Julia threads:", jl.seval("Threads.nthreads()"))

if failures:
    print("\nEnvironment verification failed:")
    for failure in failures:
        print(" -", failure)
    raise SystemExit(1)

print("\nEnvironment matches the locked reference stack.")
