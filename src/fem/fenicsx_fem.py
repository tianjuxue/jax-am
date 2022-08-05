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
from src.fem.generate_mesh import cylinder_mesh 


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
 
    file = io.XDMFFile(msh.comm, "post-processing/vtk/fem/fenicsx_linear_poisson.xdmf", "w")  
    file.write_mesh(msh)
    file.write_function(uh, 0) 

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


def linear_elasticity_cylinder():
    meshio_mesh = cylinder_mesh()
    cell_type = 'hexahedron'
    cells = meshio_mesh.get_cells_type(cell_type)
    out_mesh = meshio.Mesh(points=meshio_mesh.points, cells={cell_type: cells})
    xdmf_file = f"post-processing/msh/cylinder.xdmf"
    out_mesh.write(xdmf_file)
    mesh_xdmf_file = io.XDMFFile(MPI.COMM_WORLD, xdmf_file, 'r')
    msh = mesh_xdmf_file.read_mesh(name="Grid")


    L = 1
    E = 100.
    nu = 0.3
    mu = E/(2.*(1. + nu))
    lmbda = E*nu/((1+nu)*(1-2*nu))

    H = 10.
    R = 5.
 

    # msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, L, L])],
    #                   [10,10,10], cell_type=mesh.CellType.hexahedron)




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




    # uh = fem.Function(V)
    # uh.x.array[:] = 1.
    # a = fem.assemble_scalar(fem.form(uh[0]*ds(2)))
    # print(a)
    # uh.x.array[:] = 0
    # for i in range(len(uh.x.array)):
    #     uh.x.array[i] = 1
    #     tmp = fem.assemble_scalar(fem.form(uh[2]*ds(2)))
    #     if tmp > 0:
    #         print(f"At {i} step, tmp = {tmp}")
    #     uh.x.array[i] = 0
    # exit()


    u_top = np.array([0, 0, 1], dtype=ScalarType)
    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, boundary_facets_top), V)
 

    u_bottom = np.array([0, 0, 0], dtype=ScalarType)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets_bottom), V)
 

    def epsilon(u):
        return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    def sigma(u):
        return lmbda * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, ScalarType((0, 0, 0)))
    t = fem.Constant(msh, ScalarType((1, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(t, v) * ds(2)

    # problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left, bc_right], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_bottom], petsc_options={"ksp_type": "bicg", "pc_type": "none"})

    start_time = time.time()
    uh = problem.solve()
    end_time = time.time()
    solve_time = end_time - start_time
    print(f"Time elapsed {solve_time}")

    print(f"max of sol = {np.max(uh.x.array)}")
    print(f"min of sol = {np.min(uh.x.array)}") 

    file = io.XDMFFile(msh.comm, "post-processing/vtk/fem/fenicsx_linear_elasticity_cylinder.xdmf", "w")  
    file.write_mesh(msh)
    file.write_function(uh, 0) 
 
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
    linear_elasticity(10)

if __name__ == "__main__":
    # performance_test()
    debug()
