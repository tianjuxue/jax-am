import numpy as onp
import jax
import jax.numpy as np
import meshio
import time
import os
import glob
import scipy.optimize as opt

from src.fem.jax_fem import Mesh, Laplace
from src.fem.solver import solver, adjoint_method
from src.fem.utils import modify_vtu_file, save_sol
from src.fem.generate_mesh import box_mesh

onp.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class LinearPoisson(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 1
        self.params = None
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info) 

    def get_tensor_map(self):
        return lambda x: x

    def get_mass_map(self):
        return lambda x: x

    def compute_source(self, sol):
        mass_kernel = self.get_mass_kernel(self.get_mass_map())
        cells_sol = sol[self.cells] # (num_cells, num_nodes, vec)
        val = jax.vmap(mass_kernel)(cells_sol, self.JxW) # (num_cells, num_nodes, vec)
        val = val.reshape(-1, self.vec) # (num_cells*num_nodes, vec)
        body_force = np.zeros_like(sol)
        body_force = body_force.at[self.cells.reshape(-1)].add(val) 
        return body_force 


    def compute_residual(self, sol):
        if self.name == 'inverse': 


            # TODO: TEST
            self.body_force = self.compute_source(self.params.reshape((self.num_total_nodes, self.vec)))


        return self.compute_residual_vars(sol)


def param_id():
    root_path = f"src/fem/apps/param_id/data/vtk/"
    Lx, Ly, Lz = 1., 1., 0.2
    Nx, Ny, Nz = 50, 50, 10
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def min_x_loc(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def max_x_loc(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def min_y_loc(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def max_y_loc(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def body_force(point):
        center = np.array([Lx/4., Ly/4., Lz/2.])
        val = 10.*np.exp(-10*np.sum((point - center)**2))
        return np.array([val])

    dirichlet_bc_info = [[min_x_loc, max_x_loc, min_y_loc, max_y_loc], 
                         [0]*4, 
                         [zero_dirichlet_val]*4]

    problem_fwd = LinearPoisson(f"forward", jax_mesh, dirichlet_bc_info=dirichlet_bc_info, source_info=body_force)
    sol = solver(problem_fwd, linear=True)
    vtu_path = os.path.join(root_path, f"{problem_fwd.name}/u.vtu")
    save_sol(problem_fwd, sol, vtu_path, point_infos=[('source', problem_fwd.body_force)])

    num_obs_pts = 1000
    observed_inds = onp.random.choice(onp.arange(len(jax_mesh.points)), size=num_obs_pts, replace=False)
    observed_points = jax_mesh.points[observed_inds]
    cells = [[i%num_obs_pts, (i + 1)%num_obs_pts, (i + 2)%num_obs_pts] for i in range(num_obs_pts)]
    mesh = meshio.Mesh(observed_points, [("triangle", cells)])
    mesh.write(os.path.join(root_path, f"{problem_fwd.name}/points.vtu"))
    true_vals = sol[observed_inds]

    problem_inv = LinearPoisson(f"inverse", jax_mesh, dirichlet_bc_info=dirichlet_bc_info)
    files = glob.glob(os.path.join(root_path, f'{problem_inv.name}/*'))
    for f in files:
        os.remove(f)

    def J_fn(dofs, params):
        """J(u, p)
        """
        sol = dofs.reshape((problem_inv.num_total_nodes, problem_inv.vec))
        pred_vals = sol[observed_inds]
        assert pred_vals.shape == true_vals.shape
        l2_loss = np.sum((pred_vals - true_vals)**2) 
        reg = np.sum(params**2)
        return l2_loss + reg

    def output_sol(params, dofs, obj_val):
        sol = dofs.reshape((problem_inv.num_total_nodes, problem_inv.vec))
        vtu_path = os.path.join(root_path, f'{problem_inv.name}/sol_{fn.counter:03d}.vtu')
        save_sol(problem_inv, sol, vtu_path, point_infos=[('source', params)])
        print(f"loss = {obj_val}")
        print(f"max source = {np.max(params)}, min source = {np.min(params)}, mean source = {np.mean(params)}")

    fn, fn_grad = adjoint_method(problem_inv, J_fn, output_sol, linear=True)

    params_ini = onp.zeros(problem_inv.num_total_nodes * problem_inv.vec)

    def objective_wrapper(x):
        obj_val = fn(x)
        print(f"obj_val = {obj_val}")
        return obj_val

    def derivative_wrapper(x):
        grads = fn_grad(x)
        print(f"grads.shape = {grads.shape}")

        # 'L-BFGS-B' requires the following conversion, otherwise we get an error message saying
        # -- input not fortran contiguous -- expected elsize=8 but got 4
        return onp.array(grads, order='F', dtype=onp.float64)

    bounds = None
    options = {'maxiter': 1000, 'disp': True}  # CG or L-BFGS-B or Newton-CG or SLSQP
    res = opt.minimize(fun=objective_wrapper,
                       x0=params_ini,
                       method='L-BFGS-B',
                       jac=derivative_wrapper,
                       bounds=bounds,
                       callback=None,
                       options=options)


if __name__=="__main__":
    param_id()
