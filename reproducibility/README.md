# Exact reference environment (macOS Apple Silicon)

This bundle recreates the Python and Julia stack used by the reference runs:

- Python 3.11.15
- PyTorch 2.5.1
- NumPy 2.4.3
- PySR 1.5.10
- JuliaCall 0.9.26 and JuliaPkg 0.1.23
- Julia 1.12.6
- SymbolicRegression.jl 1.11.3
- DynamicExpressions.jl 1.10.4

The Conda lock is platform-specific and supports macOS on Apple Silicon
(`osx-arm64`). The setup script downloads dependencies from Conda, PyPI, Juliaup,
and Julia's package servers.

## Install

From the repository root, run:

```bash
bash reproducibility/setup-macos-arm64.sh
```

The default environment name is `pinn-rd-2dt-repro`. To choose another name:

```bash
bash reproducibility/setup-macos-arm64.sh my-environment-name
```

The script performs the complete setup:

1. creates the exact Conda environment from the explicit lock;
2. installs the pinned pip packages;
3. installs an isolated Juliaup and Julia 1.12.6 inside the Conda environment;
4. copies and instantiates the committed Julia `Manifest.toml` inside the
   Conda environment;
5. configures JuliaCall to use that Julia runtime and project;
6. registers a Jupyter kernel; and
7. verifies every important version and the Julia manifest checksum.

If setup is interrupted, run the same command again. The script reuses a
partially created environment after confirming that it has the expected Python
version, then repeats the idempotent installation and verification steps.

Juliaup, Julia, and the Julia package depot are stored privately inside the
Conda environment. The installer disables PATH and shell-profile changes, so it
does not edit `.bash_profile`, `.zshrc`, or `.tcshrc`, and it does not depend on
a system-wide Julia install or the original `pinn-rd-2dt` environment.

## Run the notebook

```bash
conda activate pinn-rd-2dt-repro
cd main/JN
jupyter lab paper_notebook.ipynb
```

Select the `Python (pinn-rd-2dt-repro)` kernel if Jupyter does not select it
automatically.

## Verify again later

After activating the environment from the repository root:

```bash
python reproducibility/verify_environment.py
```

The verifier exits with a nonzero status if a pinned Python package, Julia
package, runtime version, active Julia project, or manifest checksum differs.

## Scope

The lock reproduces the environment on the same operating-system family and
architecture. Separate Conda locks are required for Linux, Intel macOS, or
Windows, and floating-point behavior may still differ across CPU architectures.
