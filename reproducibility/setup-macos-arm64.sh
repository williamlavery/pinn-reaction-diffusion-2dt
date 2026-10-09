#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-pinn-rd-2dt-repro}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_LOCK="$SCRIPT_DIR/conda-osx-arm64-explicit.txt"
PIP_LOCK="$SCRIPT_DIR/pip-requirements.txt"
JULIA_PROJECT_SOURCE="$SCRIPT_DIR/julia"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_environment.py"
JULIA_VERSION="1.12.6"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This lock is for macOS on Apple Silicon (arm64)." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required. Install Miniforge or Anaconda, then rerun this script." >&2
    exit 1
fi

for required_file in "$CONDA_LOCK" "$PIP_LOCK" \
    "$JULIA_PROJECT_SOURCE/Project.toml" \
    "$JULIA_PROJECT_SOURCE/Manifest.toml" "$VERIFY_SCRIPT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done

if conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "[1/7] Reusing the existing Conda environment: $ENV_NAME"
    if ! conda run --name "$ENV_NAME" \
        python -c 'import platform, sys; sys.exit(platform.python_version() != "3.11.15")'; then
        echo "The existing environment '$ENV_NAME' does not use Python 3.11.15." >&2
        echo "Choose another environment name or remove this environment explicitly." >&2
        exit 1
    fi
else
    echo "[1/7] Creating the exact osx-arm64 Conda environment: $ENV_NAME"
    conda create --yes --name "$ENV_NAME" --file "$CONDA_LOCK"
fi

ENV_PREFIX="$(
    conda run --name "$ENV_NAME" \
        python -c 'import sys; sys.stdout.write(sys.prefix)' \
        | awk 'NF {line=$0} END {print line}'
)"

if [[ -z "$ENV_PREFIX" || "$ENV_PREFIX" != /* || ! -d "$ENV_PREFIX" ]]; then
    echo "Could not determine a valid path for Conda environment '$ENV_NAME'." >&2
    echo "Resolved path: '${ENV_PREFIX:-<empty>}'" >&2
    exit 1
fi

JULIA_PROJECT="$ENV_PREFIX/julia-project"
mkdir -p "$JULIA_PROJECT"
cp "$JULIA_PROJECT_SOURCE/Project.toml" "$JULIA_PROJECT/Project.toml"
cp "$JULIA_PROJECT_SOURCE/Manifest.toml" "$JULIA_PROJECT/Manifest.toml"

echo "[2/7] Installing the exact pip-managed packages"
conda run --name "$ENV_NAME" \
    python -m pip install --no-deps --requirement "$PIP_LOCK"

JULIAUP_ROOT="$ENV_PREFIX/juliaup"
JULIAUP_DEPOT="$ENV_PREFIX/juliaup-depot"
JULIA_DEPOT="$ENV_PREFIX/julia-depot"
JULIAUP_BIN="$JULIAUP_ROOT/bin/juliaup"

echo "[3/7] Installing the isolated Juliaup and Julia $JULIA_VERSION"
if [[ ! -x "$JULIAUP_BIN" ]]; then
    echo "      Installing Juliaup inside the Conda environment"
    TEMP_DIR="$(mktemp -d)"
    trap 'rm -rf -- "$TEMP_DIR"' EXIT
    curl --fail --location --silent --show-error \
        https://install.julialang.org \
        --output "$TEMP_DIR/install-juliaup.sh"
    JULIAUP_DEPOT_PATH="$JULIAUP_DEPOT" \
        sh "$TEMP_DIR/install-juliaup.sh" \
        --yes \
        --path "$JULIAUP_ROOT" \
        --default-channel "$JULIA_VERSION" \
        --add-to-path=no \
        --background-selfupdate=0 \
        --startup-selfupdate=0
    rm -rf -- "$TEMP_DIR"
    trap - EXIT
fi

if [[ ! -x "$JULIAUP_BIN" ]]; then
    echo "Juliaup was not installed at the expected path: $JULIAUP_BIN" >&2
    exit 1
fi

echo "      Installing Julia $JULIA_VERSION"
JULIAUP_DEPOT_PATH="$JULIAUP_DEPOT" \
    "$JULIAUP_BIN" add "$JULIA_VERSION"
JULIA_LAUNCHER="$(dirname "$JULIAUP_BIN")/julia"
JULIA_EXE="$(
    JULIAUP_DEPOT_PATH="$JULIAUP_DEPOT" \
        "$JULIA_LAUNCHER" "+$JULIA_VERSION" \
        -e 'print(joinpath(Sys.BINDIR, Base.julia_exename()))'
)"

echo "[4/7] Restoring the exact Julia package manifest"
JULIAUP_DEPOT_PATH="$JULIAUP_DEPOT" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_LAUNCHER" "+$JULIA_VERSION" --project="$JULIA_PROJECT" \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "[5/7] Wiring JuliaCall to the locked Julia runtime and project"
conda env config vars set --name "$ENV_NAME" \
    JULIAUP_DEPOT_PATH="$JULIAUP_DEPOT" \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    PYTHON_JULIAPKG_EXE="$JULIA_EXE" \
    PYTHON_JULIAPKG_PROJECT="$JULIA_PROJECT"

echo "[6/7] Registering the Jupyter kernel"
conda run --name "$ENV_NAME" \
    python -m ipykernel install --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo "[7/7] Verifying Python, PySR, Julia, and the Julia manifest"
PYTHON_JULIAPKG_EXE="$JULIA_EXE" \
PYTHON_JULIAPKG_PROJECT="$JULIA_PROJECT" \
JULIAUP_DEPOT_PATH="$JULIAUP_DEPOT" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    conda run --name "$ENV_NAME" python "$VERIFY_SCRIPT"

echo
echo "Installation complete."
echo "Activate with: conda activate $ENV_NAME"
echo "Then run:      cd main/JN && jupyter lab paper_notebook.ipynb"
