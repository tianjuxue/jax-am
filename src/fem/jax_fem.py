import numpy as onp
import jax
import jax.numpy as np
import os
import sys
import time
from src.fem.generate_mesh import box_mesh, cylinder_mesh, global_args

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=3)


global_args['dim'] = 3
global_args['num_quads'] = 8
global_args['num_nodes'] = 8


class FEM:
    def __init__(self, mesh):
        self.mesh = mesh
        self.fem_pre_computations()

    def fem_pre_computations(self):
        """Many quantities can be pre-computed and stored for better performance.
        """
        def get_quad_points():
            quad_degree = 2
            quad_points = []
            for i in range(quad_degree):
                for j in range(quad_degree):
                    for k in range(quad_degree):
                       quad_points.append([(2*(k % 2) - 1) * np.sqrt(1./3.), 
                                           (2*(j % 2) - 1) * np.sqrt(1./3.), 
                                           (2*(i % 2) - 1) * np.sqrt(1./3.)])
            quad_points = np.array(quad_points) # (quad_degree^dim, dim)
            return quad_points

        def get_shape_val_functions():
            """Hard-coded first order shape functions in the parent domain.
            Important: f1-f8 order must match "self.cells" by gmsh file!
            """
            f1 = lambda x: -1./8.*(x[0] - 1)*(x[1] - 1)*(x[2] - 1)
            f2 = lambda x: 1./8.*(x[0] + 1)*(x[1] - 1)*(x[2] - 1)
            f3 = lambda x: -1./8.*(x[0] + 1)*(x[1] + 1)*(x[2] - 1) 
            f4 = lambda x: 1./8.*(x[0] - 1)*(x[1] + 1)*(x[2] - 1)
            f5 = lambda x: 1./8.*(x[0] - 1)*(x[1] - 1)*(x[2] + 1)
            f6 = lambda x: -1./8.*(x[0] + 1)*(x[1] - 1)*(x[2] + 1)
            f7 = lambda x: 1./8.*(x[0] + 1)*(x[1] + 1)*(x[2] + 1)
            f8 = lambda x: -1./8.*(x[0] - 1)*(x[1] + 1)*(x[2] + 1)
            return [f1, f2, f3, f4, f5, f6, f7, f8]

        def get_shape_grad_functions():
            shape_fns = get_shape_val_functions()
            return [jax.grad(f) for f in shape_fns]

        @jax.jit
        def get_shape_vals():
            """Pre-compute shape function values

            Returns
            -------
            shape_vals: ndarray
               (8, 8) = (num_quads, num_nodes)  
            """
            shape_val_fns = get_shape_val_functions()
            quad_points = get_quad_points()
            shape_vals = []
            for quad_point in quad_points:
                physical_shape_vals = []
                for shape_val_fn in shape_val_fns:
                    physical_shape_val = shape_val_fn(quad_point) 
                    physical_shape_vals.append(physical_shape_val)
         
                shape_vals.append(physical_shape_vals)

            shape_vals = np.array(shape_vals) # (num_quads, num_nodes, dim)
            assert shape_vals.shape == (global_args['num_quads'], global_args['num_nodes'])
            return shape_vals

        @jax.jit
        def get_shape_grads():
            """Pre-compute shape function gradients

            Returns
            -------
            shape_grads_physical: ndarray
                (cell, num_quads, num_nodes, dim)  
            JxW: ndarray
                (cell, num_quads)
            """
            shape_grad_fns = get_shape_grad_functions()
            quad_points = get_quad_points()
            shape_grads = []
            for quad_point in quad_points:
                physical_shape_grads = []
                for shape_grad_fn in shape_grad_fns:
                    # See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
                    # Page 147, Eq. (3.9.3)
                    physical_shape_grad = shape_grad_fn(quad_point)
                    physical_shape_grads.append(physical_shape_grad)
         
                shape_grads.append(physical_shape_grads)

            shape_grads = np.array(shape_grads) # (num_quads, num_nodes, dim)
            assert shape_grads.shape == (global_args['num_quads'], global_args['num_nodes'], global_args['dim'])

            physical_coos = np.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)

            # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
            jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True)
            jacbian_det = np.squeeze(np.linalg.det(jacobian_dx_deta)) # (num_cells, num_quads)
            jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
            shape_grads_physical = np.sum(shape_grads[None, :, :, None, :] * jacobian_deta_dx, axis=3)

            # For first order FEM with 8 quad points, those quad weights are all equal to one
            quad_weights = 1.
            JxW = jacbian_det * quad_weights
            return shape_grads_physical, JxW

        self.points = self.mesh.points
        self.cells = self.mesh.cells_dict['hexahedron'] 
        global_args['num_cells'] = len(self.cells)
        global_args['num_total_vertices'] = len(self.mesh.points)
        self.shape_vals = get_shape_vals()
        self.shape_grads, self.JxW = get_shape_grads()

        return self.compute_residual


class Laplace(FEM):
    def __init__(self, mesh):
        super().__init__(mesh)    

    def compute_residual(self, dofs):
        """

        Parameters
        ----------
        dofs: ndarray
            (num_nodes, vec) 
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(dofs, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2, keepdims=True) # (num_cells, num_quads, 1, vec, dim)  
        u_physics = self.compute_physics(dofs, u_grads) # (num_cells, num_quads, 1, vec, dim)  
        v_grads = self.shape_grads[:, :, :, None, :] # (num_cells, num_quads, num_nodes, vec, dim)
        # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
        weak_form = np.sum(u_physics * v_grads * self.JxW[:, :, None, None, None], axis=(1, -1)).reshape(-1, self.vec) 
        res = np.zeros_like(dofs)
        res = res.at[self.cells.reshape(-1)].add(weak_form)
        rhs = self.get_rhs()
        return res - rhs

    def compute_physics(self, dofs, u_grads):
        """Default
        """
        return u_grads

    def get_rhs(self):
        """Default
        """
        rhs = np.zeros((global_args['num_total_vertices'], self.vec))
        return rhs

    def save_sol(self, sol):
        self.mesh.point_data['sol'] = onp.array(sol.reshape((global_args['num_total_vertices'], self.vec)), dtype=onp.float32)
        self.mesh.write(f"post-processing/vtk/fem/jax_{self.name}.vtu")


class LinearPoisson(Laplace):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.body_force = self.get_body_force()
        self.name = 'linear_poisson'
        self.vec = 1

    def get_body_force(self):
        """Pre-compute right-hand-side body force
        In a more general case, this should return ndarray (num_cells, num_quads) ?
        """
        return 10.

    def get_rhs(self):
        rhs = np.zeros((global_args['num_total_vertices'], self.vec))
        v_vals = np.repeat(self.shape_vals[None, :, :, None], global_args['num_cells'], axis=0) # (num_cells, num_quads, num_nodes, 1)
        v_vals = np.repeat(v_vals, self.vec, axis=-1) # (num_cells, num_quads, num_nodes, vec)
        rhs_vals = np.sum(v_vals * self.body_force * self.JxW[:, :, None, None], axis=1).reshape(-1, self.vec) # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
        rhs = rhs.at[self.cells.reshape(-1)].add(rhs_vals)   
        return rhs

    def get_boundary_idx(self):
        EPS = 1e-5
        left_inds_x_nodes = onp.argwhere(self.mesh.points[:, 0] < EPS).reshape(-1)
        left_inds_x_vec = onp.zeros_like(left_inds_x_nodes)
        right_inds_x_nodes = onp.argwhere(self.mesh.points[:, 0] >  global_args['domain_x'] - EPS).reshape(-1)
        right_inds_x_vec = onp.zeros_like(right_inds_x_nodes)
        inds_nodes_list = [left_inds_x_nodes, right_inds_x_nodes]
        inds_vec_list = [left_inds_x_vec, right_inds_x_vec]
        vals_list = [0., 0.]
        return inds_nodes_list, inds_vec_list, vals_list


class NonelinearPoisson(LinearPoisson):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.name = 'nonlinear_poisson'

    def compute_physics(self, dofs, u_grads):
        """

        Parameters
        ----------
        u_grads: ndarray
            (num_cells, num_quads, 1, vec, dim)
        """
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec)
        u_vals = np.take(dofs, self.cells, axis=0)[:, None, :, :] * self.shape_vals[None, :, :, None] 
        u_vals = np.sum(u_vals, axis=2, keepdims=True) # (num_cells, num_quads, 1, vec)
        q = (1 + u_vals**2)[:, :, :, :, None] # (num_cells, num_quads, 1, vec, 1)
        return q * u_grads


class LinearElasticity(Laplace):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.name = 'linear_elasticity'
        self.vec = 3

    def compute_physics(self, dofs, u_grads):
        u_grads_reshape = u_grads.reshape(-1, self.vec, global_args['dim'])

        def strain(u_grad):
            E = 100.
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            eps = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(eps)*np.eye(global_args['dim']) + 2*mu*eps
            return sigma

        strain_vmap = jax.vmap(strain)
        stress = strain_vmap(u_grads_reshape).reshape(u_grads.shape)
        return stress

    def get_boundary_idx_free_tensile(self):
        """Alternative boundary conditions
        """
        EPS = 1e-5
        left_inds_x_nodes = onp.argwhere(self.mesh.points[:, 0] < EPS).reshape(-1)
        left_inds_x_vec = onp.zeros_like(left_inds_x_nodes)
        right_inds_x_nodes = onp.argwhere(self.mesh.points[:, 0] >  global_args['domain_x'] - EPS).reshape(-1)
        right_inds_x_vec = onp.zeros_like(right_inds_x_nodes)
        corner_inds_yz_nodes = onp.array([0, 0])
        corner_inds_yz_vec = onp.array([1, 2])
        inds_nodes_list = [left_inds_x_nodes, right_inds_x_nodes, corner_inds_yz_nodes]
        inds_vec_list = [left_inds_x_vec, right_inds_x_vec, corner_inds_yz_vec]
        vals_list = [0., 0.1, 0.]
        return inds_nodes_list, inds_vec_list, vals_list
  

    def get_boundary_idx(self):
        # TODO
        EPS = 1e-5
        inds_nodes_list = []
        inds_vec_list = []
        inds = [0, 1, 2]
        for i in range(len(inds)):
            left_inds_nodes = onp.argwhere(self.mesh.points[:, 0] < EPS).reshape(-1)
            left_inds_vec = onp.ones_like(left_inds_nodes, dtype=np.int32)*inds[i]
            right_inds_nodes = onp.argwhere(self.mesh.points[:, 0] > global_args['domain_x'] - EPS).reshape(-1)
            right_inds_vec = onp.ones_like(right_inds_nodes, dtype=np.int32)*inds[i]
            inds_nodes_list += [left_inds_nodes, right_inds_nodes]
            inds_vec_list += [left_inds_vec, right_inds_vec]

        vals_list = [0., 0.1, 0., 0., 0., 0.] # left_x, right_x, left_y, right_y, left_z, right_z
        return inds_nodes_list, inds_vec_list, vals_list


class LinearElasticityCylinder(LinearElasticity):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.name = 'linear_elasticity_cylinder'
 

    def get_boundary_idx(self):
        # TODO
        EPS = 1e-5
 
        inds_nodes_list = []
        inds_vec_list = []
        inds = [0, 1, 2]
        H = 10.
        for i in range(len(inds)):
            bottom_inds_nodes = onp.argwhere(self.mesh.points[:, 2] < EPS).reshape(-1)
            bottom_inds_vec = onp.ones_like(bottom_inds_nodes, dtype=np.int32)*inds[i]
            up_inds_nodes = onp.argwhere(self.mesh.points[:, 2] > H - EPS).reshape(-1)
            up_inds_vec = onp.ones_like(up_inds_nodes, dtype=np.int32)*inds[i]
            inds_nodes_list += [bottom_inds_nodes, up_inds_nodes]
            inds_vec_list += [bottom_inds_vec, up_inds_vec]

        vals_list = [0., 0., 0., 0., 0., 1.] # bottom_x, up_x, bottom_y, up_y, bottom_z, up_z
        return inds_nodes_list, inds_vec_list, vals_list




def solver(problem):
    def operator_to_matrix(operator_fn):
        J = jax.jacfwd(operator_fn)(np.zeros(global_args['num_total_vertices']*problem.vec))
        return J

    def apply_bc(res_fn, inds_nodes_list, inds_vec_list, vals_list):
        def A_fn(dofs):
            """Apply B.C. conditions
            """
            dofs = dofs.reshape(global_args['num_total_vertices'], problem.vec)
            res = res_fn(dofs)
            for i in range(len(inds_nodes_list)):
                res = res.at[inds_nodes_list[i], inds_vec_list[i]].set(dofs[inds_nodes_list[i], inds_vec_list[i]], unique_indices=True)
                res = res.at[inds_nodes_list[i], inds_vec_list[i]].add(-vals_list[i])
            return res.reshape(-1)
        
        return A_fn

    def get_A_fn_linear_fn(sol):
        def A_fn_linear_fn(inc):
            primals, tangents = jax.jvp(A_fn, (sol,), (inc,))
            return tangents
        return A_fn_linear_fn

    def get_A_fn_linear_fn_JFNK(sol):
        def A_fn_linear_fn(inc):
            EPS = 1e-3
            return (A_fn(sol + EPS*inc) - A_fn(sol))/EPS
        return A_fn_linear_fn


    res_fn = problem.fem_pre_computations()
    inds_nodes_list, inds_vec_list, vals_list = problem.get_boundary_idx()

    print("Done pre-computing")

    A_fn = apply_bc(res_fn, inds_nodes_list, inds_vec_list, vals_list)

    start = time.time()

    sol = np.zeros((global_args['num_total_vertices'], problem.vec))
    for i in range(len(inds_nodes_list)):
        sol = sol.at[inds_nodes_list[i], inds_vec_list[i]].set(vals_list[i])
    sol = sol.reshape(-1)

    tol = 1e-6  
    step = 0
    b = -A_fn(sol)
    res_val = np.linalg.norm(b)
    print(f"step = {step}, res l_2 = {res_val}") 
    while res_val > tol:
        A_fn_linear = get_A_fn_linear_fn(sol)
        debug = False
        if debug:
            # Check onditional number of the matrix
            A_dense = operator_to_matrix(A_fn_linear)
            print(np.linalg.cond(A_dense))
            print(np.max(A_dense))
            # print(A_dense)

        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000)
        sol = sol + inc
        b = -A_fn(sol)
        res_val = np.linalg.norm(b)
        step += 1
        print(f"step = {step}, res l_2 = {res_val}") 

    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s], finished in {step} steps")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    problem.save_sol(sol)

    return solve_time


def debug():
    # mesh = box_mesh(10, 20, 30)
    # problem = LinearElasticity(mesh)
    mesh = cylinder_mesh()
    problem = LinearElasticityCylinder(mesh)
    st = solver(problem)


def performance_test():
    # Problems = [LinearElasticity, LinearPoisson, NonelinearPoisson]
    Problems = [LinearElasticity]

    # Ns = [25, 50, 100]
    Ns = [10]

    solve_time = []
    for Problem in Problems:
        prob_time = []
        for N in Ns:
            mesh = box_mesh(N, N, N)
            problem = Problem(mesh)
            st = solver(problem)
            prob_time.append(st)
        solve_time.append(prob_time)
    
    solve_time = onp.array(solve_time)
    platform = jax.lib.xla_bridge.get_backend().platform
    onp.savetxt(f"post-processing/txt/jax_fem_{platform}_time.txt", solve_time, fmt='%.3f')
    print(solve_time)


if __name__ == "__main__":
    # box_mesh()
    # performance_test()
    debug()
