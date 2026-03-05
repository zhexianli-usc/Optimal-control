FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
RUN pip install dolfin-adjoint scipy
WORKDIR /root/shared
