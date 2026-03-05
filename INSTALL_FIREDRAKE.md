# Installing Firedrake on Windows

Firedrake does **not** run natively on Windows. Use one of the following.

---

## Option 1: Docker (quickest)

Official images already have Firedrake (+ PETSc, MPI, etc.) installed.

### Run a Firedrake container

```powershell
docker pull firedrakeproject/firedrake:latest
docker run -it -v "C:\Fourier Neural Operator\Optimal-control:/home/firedrake/shared" -w /home/firedrake/shared firedrakeproject/firedrake:latest
```

Inside the container:

```bash
python3 your_script.py
```

### Optional: Docker Compose

See `docker-compose.firedrake.yml` in this folder. From the project directory:

```powershell
docker compose -f docker-compose.firedrake.yml run firedrake python3 your_script.py
```

---

## Option 2: WSL2 + Ubuntu (native install)

1. **Enable WSL2 and install Ubuntu**
   - PowerShell (Admin): `wsl --install`
   - Restart and complete Ubuntu setup.

2. **Open Ubuntu** and run:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

3. **Download the Firedrake configure script**

   ```bash
   cd ~
   curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure
   chmod +x firedrake-configure
   ```

4. **Install system dependencies**

   ```bash
   sudo apt install $(python3 firedrake-configure --show-system-packages)
   ```

5. **Install PETSc** (clone, configure, build)

   ```bash
   git clone --branch $(python3 firedrake-configure --show-petsc-version) https://gitlab.com/petsc/petsc.git
   cd petsc
   python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure
   # Follow the make command printed by configure, e.g.:
   make PETSC_DIR=$PWD PETSC_ARCH=arch-firedrake-default all
   cd ..
   ```

6. **Create venv and install Firedrake**

   ```bash
   python3 -m venv venv-firedrake
   source venv-firedrake/bin/activate
   pip cache purge
   export $(python3 firedrake-configure --show-env)
   echo 'setuptools<81' > constraints.txt
   export PIP_CONSTRAINT=constraints.txt
   pip install --no-binary h5py 'firedrake[check]'
   firedrake-check
   ```

7. **Use your project** (e.g. project on C:)

   ```bash
   cd /mnt/c/Fourier\ Neural\ Operator/Optimal-control
   source ~/venv-firedrake/bin/activate
   python3 your_script.py
   ```

Full details: https://www.firedrakeproject.org/firedrake/install.html  
WSL-specific notes: https://github.com/firedrakeproject/firedrake/wiki/Installing-on-Windows-Subsystem-for-Linux

---

## Firedrake-adjoint (PDE-constrained optimization)

To use adjoint-based optimization (like dolfin-adjoint):

```bash
pip install firedrake-adjoint
```

Then in Python: `from firedrake_adjoint import *` and use `ReducedFunctional`, `minimize`, `Control`, etc.
