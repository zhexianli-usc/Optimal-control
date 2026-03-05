from fenics import *
from fenics_adjoint import *
from collections import OrderedDict


data = Expression("16*x[0]*(x[0]-1)*x[1]*(x[1]-1)*sin(pi*t)", t=0, degree=4)
nu = Constant(1e-5)

mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "CG", 1)

dt = Constant(0.1)
T = 2

ctrls = OrderedDict()
t = float(dt)
while t <= T:
    ctrls[t] = Function(V)
    t += float(dt)

def solve_heat(ctrls):
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V, name="source")
    u_0 = Function(V, name="solution")
    d = Function(V, name="data")

    F = ( (u - u_0)/dt*v + nu*inner(grad(u), grad(v)) - f*v)*dx
    a, L = lhs(F), rhs(F)
    bc = DirichletBC(V, 0, "on_boundary")

    t = float(dt)

    j = 0.5*float(dt)*assemble((u_0 - d)**2*dx)

    while t <= T:
        # Update source term from control array
        f.assign(ctrls[t])

        # Update data function
        data.t = t
        d.assign(interpolate(data, V))

        # Solve PDE
        solve(a == L, u_0, bc)

        # Implement a trapezoidal rule
        if t > T - float(dt):
           weight = 0.5
        else:
           weight = 1

        j += weight*float(dt)*assemble((u_0 - d)**2*dx)

        # Update time
        t += float(dt)

    return u_0, d, j

u, d, j = solve_heat(ctrls)

alpha = Constant(1e-1)
regularisation = alpha/2*sum([1/dt*(fb-fa)**2*dx for fb, fa in
    zip(list(ctrls.values())[1:], list(ctrls.values())[:-1])])

J = j + assemble(regularisation)
m = [Control(c) for c in ctrls.values()]

rf = ReducedFunctional(J, m)
opt_ctrls = minimize(rf, options={"maxiter": 50})

from matplotlib import pyplot, rc
rc('text', usetex=True)
x = [c((0.5, 0.5)) for c in opt_ctrls]
pyplot.plot(x, label="$\\alpha={}$".format(float(alpha)))
pyplot.ylim([-3, 3])
pyplot.legend()
