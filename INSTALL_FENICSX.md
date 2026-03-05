# Installing FEniCSx for Optimal Control

`Optimal_control_fenicsx.py` uses **FEniCSx** (DOLFINx) with a **manual adjoint** and scipy. No FEniCSx-adjoint package is required.

## Option 1: Conda (recommended on Windows and macOS)

**Windows:** `mpich` is not on conda-forge for Windows; omit it. FEniCSx will use bundled or system MPI.

```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx petsc4py petsc pyvista scipy -y
```

If you get `ModuleNotFoundError: No module named 'petsc4py'`, install it explicitly (pip does not provide Windows binaries):

```bash
conda install -c conda-forge petsc4py petsc -y
```

**Linux / macOS:** You can add `mpich` if you need it:

```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista scipy -y
```

Then run:

```bash
python Optimal_control_fenicsx.py
```

## Option 2: Debian / Ubuntu

```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx
pip install scipy
python Optimal_control_fenicsx.py
```

## Option 3: Pip (Linux; may need system libs)

```bash
pip install fenics-dolfinx scipy mpi4py
python Optimal_control_fenicsx.py
```

On Windows, FEniCSx via pip is not officially supported; use Conda or WSL.

## Run

Single process (script expects this by default):

```bash
python Optimal_control_fenicsx.py
```

With MPI (still one process):

```bash
mpiexec -n 1 python Optimal_control_fenicsx.py
```

## FEniCSx-adjoint (optional)

There is ongoing work on **dolfinx-adjoint** / **FEniCSx-adjoint** (automatic differentiation for DOLFINx). For now, this project uses a **manual adjoint** (same formulation as the legacy `Optimal_control.py` but implemented by hand with UFL and scipy), which works with standard FEniCSx only.
