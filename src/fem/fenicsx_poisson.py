import numpy as np
import time
import ufl
import dolfinx
from dolfinx import fem, io, mesh, plot, log
from ufl import ds, dx, grad, inner
from mpi4py import MPI
import petsc4py
from petsc4py.PETSc import ScalarType

msh = mesh.create_box(comm=MPI.COMM_WORLD,
                      points=((0., 0., 0.), (1., 1., 1.)), n=(100, 100, 100),
                      cell_type=mesh.CellType.hexahedron)
V = fem.FunctionSpace(msh, ("Lagrange", 1))
facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 1.0)))
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0.), dofs=dofs, V=V)

linear = False

if linear:
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10.
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    # problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly"})
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "bicg", "pc_type": "none"})

    start_time = time.time()
    uh = problem.solve()
    end_time = time.time()
    print(f"Time elapsed {end_time - start_time}")
    print(f"max of sol = {np.max(uh.x.array)}")
else:
    uh = fem.Function(V)
    v = ufl.TestFunction(V)
    f = 10.
    F_res = (1+uh**2)*inner(grad(uh), grad(v)) * dx - inner(f, v) * dx
    problem = dolfinx.fem.petsc.NonlinearProblem(F_res, uh, [bc])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    ksp = solver.krylov_solver
    opts = petsc4py.PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "none"
    ksp.setFromOptions()
    log.set_log_level(log.LogLevel.INFO)

    start_time = time.time()
    n, converged = solver.solve(uh)
    end_time = time.time()
    print(f"Number of interations: {n:d}")
    print(f"Time elapsed {end_time - start_time}")
    print(f"max of sol = {np.max(uh.x.array)}")

file = io.XDMFFile(msh.comm, "post-processing/vtk/fem/fenicsx_poisson.xdmf", "w")  
file.write_mesh(msh)
file.write_function(uh, 0) 

