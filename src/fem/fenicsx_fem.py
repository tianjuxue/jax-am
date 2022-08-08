import numpy as np
import dolfinx
from dolfinx import fem, io, mesh, plot, log
import ufl
from ufl import ds, dx, grad, inner
from mpi4py import MPI
import petsc4py
from petsc4py.PETSc import ScalarType
import time
import meshio
import sys
from src.fem.generate_mesh import cylinder_mesh 

np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)


def linear_poisson(N):
    msh = mesh.create_box(comm=MPI.COMM_WORLD,
                          points=((0., 0., 0.), (1., 1., 1.)), n=(N, N, N),
                          cell_type=mesh.CellType.hexahedron)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                           marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                          np.isclose(x[0], 1.0)))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0.), dofs=dofs, V=V)

 
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
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")
    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 
 
    # file = io.XDMFFile(msh.comm, "post-processing/vtk/fem/fenicsx_linear_poisson.xdmf", "w")  
    # file.write_mesh(msh)
    # file.write_function(uh, 0) 


    file = io.VTKFile(msh.comm, "post-processing/vtk/fem/fenicsx_linear_poisson.pvd", "w")  
    # file.write_mesh(msh)
    file.write_function(uh, 0) 

    print(uh.x.array)

    return solve_time


def nonlinear_poisson(N):
    msh = mesh.create_box(comm=MPI.COMM_WORLD,
                          points=((0., 0., 0.), (1., 1., 1.)), n=(N, N, N),
                          cell_type=mesh.CellType.hexahedron)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                           marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                          np.isclose(x[0], 1.0)))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0.), dofs=dofs, V=V)

    uh = fem.Function(V)
    v = ufl.TestFunction(V)
    f = 10.
    F_res = (1+uh**2)*inner(grad(uh), grad(v)) * dx - inner(f, v) * dx
    problem = dolfinx.fem.petsc.NonlinearProblem(F_res, uh, [bc])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    ksp = solver.krylov_solver
    opts = petsc4py.PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "bicg"
    opts[f"{option_prefix}pc_type"] = "none"
    ksp.setFromOptions()
    log.set_log_level(log.LogLevel.INFO)

    start_time = time.time()
    n, converged = solver.solve(uh)
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")
    print(f"Number of interations: {n:d}")
    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    file = io.XDMFFile(msh.comm, "post-processing/vtk/fem/fenicsx_nonlinear_poisson.xdmf", "w")  
    file.write_mesh(msh)
    file.write_function(uh, 0) 

    return solve_time


def linear_elasticity(N):
    L = 1
    E = 100.
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, L, L])],
                      [N,N,N], cell_type=mesh.CellType.hexahedron)
    V = fem.VectorFunctionSpace(msh, ("CG", 1))

    def boundary_left(x):
        return np.isclose(x[0], 0)

    def boundary_right(x):
        return np.isclose(x[0], L)

    fdim = msh.topology.dim - 1
    boundary_facets_left = mesh.locate_entities_boundary(msh, fdim, boundary_left)
    boundary_facets_right = mesh.locate_entities_boundary(msh, fdim, boundary_right)

    marked_facets = boundary_facets_right
    marked_values = np.full(len(boundary_facets_right), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_right[sorted_facets], marked_values[sorted_facets])
    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)

    u_left = np.array([1.,1.,1.], dtype=ScalarType)
    bc_left = fem.dirichletbc(u_left, fem.locate_dofs_topological(V, fdim, boundary_facets_left), V)

    u_right = np.array([0.1,0,0], dtype=ScalarType)
    bc_right = fem.dirichletbc(u_right, fem.locate_dofs_topological(V, fdim, boundary_facets_right), V)

    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, ScalarType((0, 10., 10.)))
    t = fem.Constant(msh, ScalarType((10., 0., 0.)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(t, v) * ds(2)

    # problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left, bc_right], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left], petsc_options={"ksp_type": "bicg", "pc_type": "none"})

    start_time = time.time()
    uh = problem.solve()
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    file = io.XDMFFile(msh.comm, "post-processing/vtk/fem/fenicsx_linear_elasticity.xdmf", "w")  
    file.write_mesh(msh)
    file.write_function(uh, 0) 

    return solve_time




def plasticity():
    meshio_mesh = cylinder_mesh()
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = f"post-processing/msh/cylinder.xdmf"
    out_mesh.write(xdmf_file)
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")
    E = 70e3
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))
    sig0 = 250.

    H = 10.

    deg_stress = 2
    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
 
    W_ele = ufl.TensorElement("Quadrature", msh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W = fem.FunctionSpace(msh, W_ele)
    V = fem.VectorFunctionSpace(msh, ("CG", 1))

    def boundary_top(x):
        # H = 10.
        return np.isclose(x[2], H)

    def boundary_bottom(x):
        return np.isclose(x[2], 0.)

    fdim = msh.topology.dim - 1
    boundary_facets_top = mesh.locate_entities_boundary(msh, fdim, boundary_top)
    boundary_facets_bottom = mesh.locate_entities_boundary(msh, fdim, boundary_bottom)

    marked_facets = boundary_facets_top
    marked_values = np.full(len(boundary_facets_top), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_top[sorted_facets], marked_values[sorted_facets])


    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=msh, metadata=metadata)

    u_top = np.array([0, 0, 1.], dtype=ScalarType)
    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, boundary_facets_top), V)
 

    u_bottom = np.array([0, 0, 0], dtype=ScalarType)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets_bottom), V)
 

    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def elastic_stress(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    sig = fem.Function(W)

    ppos = lambda x: (x + abs(x))/2.

    heaviside = lambda x: ufl.conditional(ufl.gt(x, 0.), x, 1e-10)

    def stress_fn(u_crt, u_old):
        EPS = 1e-10
        sig_elas = sig + elastic_stress(u_crt - u_old)
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(heaviside(3/2.*ufl.inner(s, s)))
        f_elas = sig_eq - sig0
        # Prevent divided by zero error
        # The original example (https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html)
        # didn't consider this, and can cause nan error in the solver.
        new_sig = sig_elas - ppos(f_elas)*s/(sig_eq + EPS)
        return new_sig

    x = ufl.SpatialCoordinate(msh)
    u_crt = fem.Function(V)
    u_old = fem.Function(V)
    v = ufl.TestFunction(V)
    F_res = ufl.inner(stress_fn(u_crt, u_old), epsilon(v)) * dxm


    problem = dolfinx.fem.petsc.NonlinearProblem(F_res, u_crt, [bc_bottom, bc_top])
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    ksp = solver.krylov_solver
    opts = petsc4py.PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "bicg"
    opts[f"{option_prefix}pc_type"] = "none"
    ksp.setFromOptions()
    log.set_log_level(log.LogLevel.INFO)

    start_time = time.time()
    n, converged = solver.solve(u_crt)
    end_time = time.time()


    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(u_crt.x.array)}")
    print(f"min of sol = {np.min(u_crt.x.array)}") 

    # file = io.VTKFile(msh.comm, "src/fem/tests/linear_elasticity_cylinder/fenicsx/sol.pvd", "w")  
    # file.write_function(u_crt, 0) 

    u_crt = fem.Function(V)
    u_crt.x.array[:] = 1.
    surface_area = fem.assemble_scalar(fem.form(u_crt[0]*ds(2)))

    # np.save(f"src/fem/tests/linear_elasticity_cylinder/fenicsx/surface_area.npy", surface_area)

    return solve_time





def linear_elasticity_cylinder():
    meshio_mesh = cylinder_mesh()
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = f"post-processing/msh/cylinder.xdmf"
    out_mesh.write(xdmf_file)
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")
    E = 100.
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    H = 10.

    V = fem.VectorFunctionSpace(msh, ("CG", 1))

    def boundary_top(x):
        # H = 10.
        return np.isclose(x[2], H)

    def boundary_bottom(x):
        return np.isclose(x[2], 0.)

    fdim = msh.topology.dim - 1
    boundary_facets_top = mesh.locate_entities_boundary(msh, fdim, boundary_top)
    boundary_facets_bottom = mesh.locate_entities_boundary(msh, fdim, boundary_bottom)

    marked_facets = boundary_facets_top
    marked_values = np.full(len(boundary_facets_top), 2, dtype=np.int32) 
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(msh, fdim, boundary_facets_top[sorted_facets], marked_values[sorted_facets])

    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tag, metadata=metadata)

    u_top = np.array([0, 0, 1], dtype=ScalarType)
    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, boundary_facets_top), V)
 

    u_bottom = np.array([0, 0, 0], dtype=ScalarType)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets_bottom), V)
 

    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    x = ufl.SpatialCoordinate(msh)
    f = ufl.as_vector((0.1*x[0], 0.2*x[1], 0.3*x[2]))


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # f = fem.Constant(msh, ScalarType((0, 0, 0)))
    t = fem.Constant(msh, ScalarType((1, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(t, v) * ds(2)

    # problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left, bc_right], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_bottom], petsc_options={"ksp_type": "bicg", "pc_type": "none"})

    start_time = time.time()
    uh = problem.solve()
    uh.name = 'sol'
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    file = io.VTKFile(msh.comm, "src/fem/tests/linear_elasticity_cylinder/fenicsx/sol.pvd", "w")  
    file.write_function(uh, 0) 

    uh = fem.Function(V)
    uh.x.array[:] = 1.
    surface_area = fem.assemble_scalar(fem.form(uh[0]*ds(2)))

    np.save(f"src/fem/tests/linear_elasticity_cylinder/fenicsx/surface_area.npy", surface_area)

    return solve_time


def performance_test():
    problems = [linear_elasticity, linear_poisson, nonlinear_poisson]
    # problems = [linear_elasticity]

    Ns = [25, 50, 100]
    # Ns = [50]

    solve_time = []
    for problem in problems:
        prob_time = []
        for N in Ns:
            st = problem(N)
            prob_time.append(st)
        solve_time.append(prob_time)
    
    solve_time = np.array(solve_time)
    np.savetxt(f"post-processing/txt/fenicsx_fem_time.txt", solve_time, fmt='%.3f')
    print(solve_time)


def debug():
    # linear_elasticity_cylinder()
    # linear_elasticity(10)
    # linear_poisson(10)
    plasticity()

if __name__ == "__main__":
    # performance_test()
    debug()
