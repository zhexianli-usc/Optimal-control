# dolfin-adjoint with FEniCSx

## Short answer

**There is no official “dolfin-adjoint for FEniCSx” package** that works with **DOLFINx** (FEniCSx) the same way `dolfin-adjoint` works with legacy FEniCS (dolfin).

- **dolfin-adjoint** (PyPI / GitHub) is for **legacy FEniCS (dolfin)** and uses `fenics-ufl-legacy`. It does **not** target FEniCSx/DOLFINx.
- **FEniCSx (DOLFINx)** is the new stack. Automatic adjoints for it are either done manually or via projects that are not yet the standard “dolfin-adjoint” install.

So you have two paths: use **legacy FEniCS + dolfin-adjoint**, or use **FEniCSx** with a **manual adjoint** (or a future DOLFINx-adjoint package).

---

## Option A: Legacy FEniCS + dolfin-adjoint (not FEniCSx)

This is the **classic** setup: **FEniCS 2019 (dolfin)** and **dolfin-adjoint**. It does **not** use FEniCSx.

1. **Install legacy FEniCS**  
   - Easiest on Windows: **Docker** (e.g. image `ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30`) or **WSL2 + Ubuntu** and then:
     ```bash
     sudo apt install fenics
     ```
   - Or conda (Linux/macOS): `conda install -c conda-forge fenics`

2. **Install dolfin-adjoint** in that same environment:
   ```bash
   pip install dolfin-adjoint
   ```
   Or from GitHub (latest):
   ```bash
   pip install git+https://github.com/dolfin-adjoint/dolfin-adjoint.git@main
   ```

3. **Use legacy API** in your script:
   ```python
   from dolfin import *
   from dolfin_adjoint import *
   ```
   This is what `Optimal_control.py` does. It is **not** FEniCSx.

---

## Option B: FEniCSx (DOLFINx) — no dolfin-adjoint package

For **FEniCSx (DOLFINx)** you have two options:

### B1. Use the manual-adjoint script (recommended)

Use **FEniCSx only** (no dolfin-adjoint), with a hand-written adjoint and scipy:

```bash
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx pyvista scipy -y
python Optimal_control_fenicsx.py
```

- Script: `Optimal_control_fenicsx.py`
- Same maths as the dolfin-adjoint example (Poisson optimal control), but implemented with **DOLFINx** and a **manual adjoint** + `scipy.optimize.minimize`.

No “install dolfin-adjoint with FEniCSx” step — it’s not used here.

### B2. Future: DOLFINx-adjoint / FEniCSx-adjoint

There is ongoing work on adjoint/AD for **DOLFINx** (sometimes under names like “dolfinx-adjoint” or “FEniCSx-adjoint”). As of now:

- There is **no** official, stable “install dolfin-adjoint with FEniCSx” single command.
- Check the [dolfin-adjoint GitHub](https://github.com/dolfin-adjoint) and [FEniCS Project](https://fenicsproject.org/) for any new packages or backends that support DOLFINx.

When such a package exists, installation might look like:

```bash
conda install -c conda-forge fenics-dolfinx
pip install dolfinx-adjoint   # or whatever the package name will be
```

Until then, **Option B1** (FEniCSx + manual adjoint) is the way to use “dolfin-adjoint-style” optimization with FEniCSx.

---

## Summary

| Goal                         | Install                                      | Script / API                    |
|-----------------------------|----------------------------------------------|----------------------------------|
| **Legacy FEniCS + adjoint** | Docker/WSL/conda FEniCS + `pip install dolfin-adjoint` | `from dolfin_adjoint import *`   |
| **FEniCSx + adjoint**       | FEniCSx only (conda/pip)                    | `Optimal_control_fenicsx.py` (manual adjoint) |

So: **you cannot “install dolfin-adjoint with FEniCSx”** in the sense of one package that adds dolfin-adjoint to DOLFINx. Use legacy FEniCS + dolfin-adjoint, or FEniCSx + the manual adjoint script (and later, any official DOLFINx-adjoint package when it appears).
