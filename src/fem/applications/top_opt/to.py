import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import scipy.optimize as opt
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import meshio
import time

from src.fem.generate_mesh import box_mesh
from src.fem.jax_fem import Mesh, Laplace
from src.fem.solver import solver, linear_solver, apply_bc
from src.fem.utils import save_sol

from src.fem.applications.top_opt.AuTo.utilfuncs import computeLocalElements, computeFilter
from src.fem.applications.top_opt.AuTo.mmaOptimize import optimize


nelx, nely = 50, 30
elemSize = np.array([1., 1.])
mesh = {'nelx':nelx, 'nely':nely, 'elemSize':elemSize,\
        'ndof':3*(nelx+1)*(nely+1)*2, 'numElems':nelx*nely}

material = {'Emax':1., 'Emin':1e-3, 'nu':0.3, 'penal':3.}

filterRadius = 1.5
H, Hs = computeFilter(mesh, filterRadius)
ft = {'type':1, 'H':H, 'Hs':Hs}


globalVolumeConstraint = {'isOn':True, 'vf':0.5}


optimizationParams = {'maxIters':200,'minIters':100,'relTol':0.05}
projection = {'isOn':False, 'beta':4, 'c0':0.5}



class LinearElasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        self.params = None       
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        self.neumann_boundary_inds = self.Neuman_boundary_conditions_inds(neumann_bc_info[0])[0]

    def stress_strain_fns(self):
        def stress(u_grad, theta):
            # E = 10.
            # E = 1. + 9*theta**3

            E = material['Emin'] + (material['Emax'] - material['Emin'])*(theta+0.01)**material['penal']

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

    def compute_compliance(self, neumann_fn, sol):
        """Compute surface integral specified by neumann_fn: (traction, u) * ds
        For post-processing only.
        Example usage: compute the total force on a certain surface.

        Parameters
        ----------
        surface_fn: callable
            A function that inputs a point (ndarray) and returns the value.
        sol: ndarray
            (num_total_nodes, vec)

        Returns
        -------
        val: ndarray
            ()
        """
        boundary_inds = self.neumann_boundary_inds
        _, nanson_scale = self.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.cells][boundary_inds[:, 0]][:, None, :, :] * self.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
        traction = jax.vmap(jax.vmap(neumann_fn))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


def save_sol_params(problem, params, sol, sol_file):
    # TODO: a more general save sol func
    out_mesh = meshio.Mesh(points=problem.points, cells={'hexahedron': problem.cells})
    out_mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    out_mesh.cell_data['theta'] = [onp.array(params, dtype=onp.float32)]
    out_mesh.write(sol_file)


def debug():
    root_path = f'src/fem/applications/top_opt/data'

    files = glob.glob(os.path.join(root_path, 'vtk/*'))
    for f in files:
        os.remove(f)

    # meshio_mesh = box_mesh(50, 30, 1, 4., 1., 0.1)
    meshio_mesh = box_mesh(nelx, nely, 1, 50., 30., 1.)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load(point):
        return np.logical_and(np.isclose(point[0], 50., atol=1e-5), np.isclose(point[1], 15., atol=1.5))

    def dirichlet_val(point):
        return 0.

    def neumann_val(point):
        return np.array([0., -1., 0.])

    dirichlet_bc_info = [[left, left, left], [0, 1, 2], [dirichlet_val, dirichlet_val, dirichlet_val]]
    neumann_bc_info = [[load], [neumann_val]]

    problem = LinearElasticity('linear_elasticity', jax_mesh, dirichlet_bc_info, neumann_bc_info)

    key = jax.random.PRNGKey(seed=0)

    # theta_ini = jax.random.uniform(key, (problem.num_cells,))
    # theta_ini = 0.4*np.ones(problem.num_cells)
    # compliance = fn(theta_ini)

    def fn(params):
        print(f"\nStep {fn.counter}")
        problem.params = params
        # sol = solver(problem, initial_guess=fn.sol, use_linearization_guess=False)
        sol = linear_solver(problem)
        compliance = problem.compute_compliance(neumann_val, sol)
        dofs = sol.reshape(-1)
        vtu_path = os.path.join(root_path, f'vtk/sol_{fn.counter:03d}.vtu')
        save_sol_params(problem, params, sol, vtu_path)
        fn.dofs = dofs
        fn.sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        fn.counter += 1
        print(f"compliance = {compliance}")
        print(f"max theta = {np.max(params)}, min theta = {np.min(params)}, mean theta = {np.mean(params)}")
        return compliance

    fn.counter = 0
    fn.sol = np.zeros((problem.num_total_nodes, problem.vec))

    def J_fn(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        compliance = problem.compute_compliance(neumann_val, sol)
        return compliance

    def constraint_fn(params, dofs):
        problem.params = params
        res_fn = problem.compute_residual
        
        A_fn = apply_bc(res_fn, problem)
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

    @jax.jit
    def fn_grad(params):
        dofs = fn.dofs
        partial_dJ_dx = jax.grad(J_fn)(dofs)
        adjoint_linear_fn = get_adjoint_linear_fn(params, dofs)
        vjp_linear_fn = get_vjp_contraint_fn_params(params, dofs)
        # start = time.time()
        # adjoint, info = jax.scipy.sparse.linalg.bicgstab(adjoint_linear_fn, partial_dJ_dx, x0=fn_grad.adjoint, M=None, tol=1e-10, atol=1e-10, maxiter=10000)
        adjoint, info = jax.scipy.sparse.linalg.bicgstab(adjoint_linear_fn, partial_dJ_dx, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000)
        # fn_grad.adjoint = adjoint
        # end = time.time()
        # print(f"Adjoint solve took {end - start} [s]")
        total_dJ_dp = -vjp_linear_fn(adjoint)
        return total_dJ_dp

    # fn_grad.adjoint = np.zeros(problem.num_total_nodes*problem.vec)

    def objectiveHandle(rho):
        J = fn(rho)
        dJ = fn_grad(rho)
        return J, dJ

    def computeConstraints(rho, epoch): 
        @jax.jit
        def computeGlobalVolumeConstraint(rho):
            g = np.mean(rho)/globalVolumeConstraint['vf'] - 1.
            return g
        c, gradc = jax.value_and_grad(computeGlobalVolumeConstraint)(rho);
        c, gradc = c.reshape((1, 1)), gradc.reshape((1, -1))
        return c, gradc
 
    optimize(mesh, optimizationParams, ft, objectiveHandle, computeConstraints, numConstraints=1)


if __name__=="__main__":
    debug()
