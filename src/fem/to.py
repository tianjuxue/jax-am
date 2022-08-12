import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import scipy.optimize as opt
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import meshio
import time
from src.fem.generate_mesh import box_mesh
from src.fem.jax_fem import Mesh, Laplace, solver, save_sol


class LinearElasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        self.params = None
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
    
    def stress_strain_fns(self):
        def stress(u_grad, theta):
            # E = 10.
            E = 1. + 9*theta**3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma

        vmap_stress = jax.vmap(stress)
        return vmap_stress

    def compute_physics(self, sol, u_grads):
        """

        Parameters
        ----------
        u_grads: ndarray
            (num_cells, num_quads, vec, dim)
        """

        thetas = np.repeat(self.params[:, None], self.num_quads, axis=1).reshape(-1)

        u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
        vmap_stress = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape, thetas).reshape(u_grads.shape)
        return sigmas


    def compute_compliance(self, location_fn, neumann_fn, sol):
        """Compute compliance

        Returns
        -------
        val: ndarray
            ()
        """
        boundary_inds = self.Neuman_boundary_conditions_inds([location_fn])[0]
        _, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :] * self.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
        traction = jax.vmap(jax.vmap(neumann_fn))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


def linear_elasticity():
    # meshio_mesh = box_mesh(50, 30, 1, 4., 1., 0.1)
    meshio_mesh = box_mesh(50, 30, 1, 4., 1., 0.1)

    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load(point):
        return np.logical_and(np.isclose(point[0], 4., atol=1e-5), np.isclose(point[1], 0.5, atol=0.05))

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., -1., 0.])

    dirichlet_bc_info = [[left, left, left], [0, 1, 2], [dirichlet_val, dirichlet_val, dirichlet_val]]
    neumann_bc_info = [[load], [neumann_val]]

    problem = LinearElasticity('linear_elasticity', mesh, dirichlet_bc_info, neumann_bc_info)
    sol = solver(problem)

    compliance = problem.compute_compliance(load, neumann_val, sol)
    print(f"########### compliance = {compliance}")

    vtu_path = f"src/fem/data/vtk/to/sol.vtu"
    save_sol(problem, sol, vtu_path)


def save_sol_params(problem, params, sol, sol_file):
    out_mesh = meshio.Mesh(points=problem.points, cells={'hexahedron': problem.cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    out_mesh.cell_data['theta'] = [onp.array(params, dtype=onp.float32)]
    out_mesh.write(sol_file)


def debug():

    files = glob.glob(f"src/fem/data/vtk/to/*")
    for f in files:
        os.remove(f)


    meshio_mesh = box_mesh(50, 30, 1, 4., 1., 0.1)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load(point):
        return np.logical_and(np.isclose(point[0], 4., atol=1e-5), np.isclose(point[1], 0.5, atol=0.05))

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., -1., 0.])

    dirichlet_bc_info = [[left, left, left], [0, 1, 2], [dirichlet_val, dirichlet_val, dirichlet_val]]
    neumann_bc_info = [[load], [neumann_val]]

    problem = LinearElasticity('linear_elasticity', mesh, dirichlet_bc_info, neumann_bc_info)

    key = jax.random.PRNGKey(seed=0)

    theta_ini = jax.random.uniform(key, (problem.num_cells,))

    theta_ini = 0.4*np.ones(problem.num_cells)

    node_inds_list, vec_inds_list, vals_list = problem.Dirichlet_boundary_conditions()

    # compliance = fn(theta_ini)

    def fn(params):
        print(f"\nStep {fn.counter}")
        problem.params = params
        sol = solver(problem, initial_guess=fn.sol, use_linearization_guess=False)
        compliance = problem.compute_compliance(load, neumann_val, sol)
        dofs = sol.reshape(-1)
        vtu_path = f"src/fem/data/vtk/to/sol_{fn.counter:03d}.vtu"
        save_sol_params(problem, params, sol, vtu_path)
        fn.dofs = dofs
        fn.sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        fn.counter += 1
        print(f"compliance = {compliance}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        return compliance

    fn.counter = 0
    fn.sol = np.zeros((problem.num_total_nodes, problem.vec))

    def apply_bc(res_fn):
        def A_fn(dofs):
            """Apply Dirichlet boundary conditions
            """
            sol = dofs.reshape((problem.num_total_nodes, problem.vec))
            res = res_fn(sol)
            for i in range(len(node_inds_list)):
                res = res.at[node_inds_list[i], vec_inds_list[i]].set(sol[node_inds_list[i], vec_inds_list[i]], unique_indices=True)
                res = res.at[node_inds_list[i], vec_inds_list[i]].add(-vals_list[i])
            return res.reshape(-1)
        return A_fn

    def J_fn(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        compliance = problem.compute_compliance(load, neumann_val, sol)
        return compliance

    def constraint_fn(params, dofs):
        problem.params = params
        res_fn = problem.compute_residual
        
        A_fn = apply_bc(res_fn)
        return A_fn(dofs)

    def get_partial_dofs_c_fn(params):
        def partial_dofs_c_fn(dofs):
            return constraint_fn(params, dofs)
        return partial_dofs_c_fn

    def get_partial_params_c_fn(dofs):
        def partial_params_c_fn(params):
            return constraint_fn(params, dofs)
        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, dofs):
        partial_c_fn = get_partial_params_c_fn(dofs)
        def vjp_linear_fn(v):
            primals, f_vjp = jax.vjp(partial_c_fn, params)
            val, = f_vjp(v)
            return val
        return jax.jit(vjp_linear_fn)

    def get_adjoint_linear_fn(params, dofs):
        partial_c_fn = get_partial_dofs_c_fn(params)
        def adjoint_linear_fn(adjoint):
            primals, f_vjp = jax.vjp(partial_c_fn, dofs)
            val, = f_vjp(adjoint)
            return val
        return jax.jit(adjoint_linear_fn)

    def fn_grad(params):
        dofs = fn.dofs
        partial_dJ_dx = jax.grad(J_fn)(dofs)
        adjoint_linear_fn = get_adjoint_linear_fn(params, dofs)
        vjp_linear_fn = get_vjp_contraint_fn_params(params, dofs)
        start = time.time()
        adjoint, info = jax.scipy.sparse.linalg.bicgstab(adjoint_linear_fn, partial_dJ_dx, x0=fn_grad.adjoint, M=None, tol=1e-10, atol=1e-10, maxiter=10000)
        fn_grad.adjoint = adjoint
        end = time.time()
        print(f"Adjoint solve took {end - start} [s]")
        total_dJ_dp = -vjp_linear_fn(adjoint)

        # 'L-BFGS-B' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(total_dJ_dp, order='F', dtype=onp.float64)

    fn_grad.adjoint = np.zeros(problem.num_total_nodes*problem.vec)

    bounds = Bounds(onp.zeros_like(theta_ini), onp.ones_like(theta_ini))

    linear_constraint = LinearConstraint([[1./len(theta_ini)]*len(theta_ini)], [-onp.inf], [0.4])


    g_fn = lambda x: 0.4 - np.mean(x)

    ineq_cons = {'type': 'eq',
                 'fun' : g_fn,
                 'jac' : jax.grad(g_fn)}

 
    options = {'maxiter': 10000, 'disp': True}  # CG or L-BFGS-B or Newton-CG or SLSQP or trust-constr
    res = opt.minimize(fun=fn,
                       x0=theta_ini,
                       method='trust-constr',
                       jac=fn_grad,
                       bounds=bounds,
                       callback=None,
                       constraints=[linear_constraint],
                       options=options)


if __name__=="__main__":
    # linear_elasticity()
    debug()
