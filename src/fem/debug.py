import numpy as onp
import jax
import jax.numpy as np
import time
import os
import glob
from functools import partial
from src.fem.generate_mesh import box_mesh, cylinder_mesh
from src.fem.solver import solver, assign_bc, get_A_fn_linear_fn, row_elimination, get_flatten_fn
from src.fem.jax_fem import Mesh, Laplace, HyperElasticity
from src.fem.utils import save_sol


os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def assign_ones(sol, problem):
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(1.)
    return sol


def assign_zero_bc(sol, problem):
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(0.)
    return sol


class LinearizedHyperElasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        self.disp_grad = np.zeros((len(self.cells)*self.num_quads, self.vec, self.dim))
        self.C = self.update_C()


    # def stress_strain_fns(self):
    #     def psi(F):
    #         E = 70e3
    #         nu = 0.3
    #         mu = E/(2.*(1. + nu))
    #         kappa = E/(3.*(1. - 2.*nu))
    #         J = np.linalg.det(F)
    #         Jinv = J**(-2./3.)
    #         I1 = np.trace(F.T @ F)
    #         energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
    #         return energy
    #     P_fn = jax.grad(psi)

    #     def first_PK_stress(u_grad):
    #         I = np.eye(self.dim)
    #         F = u_grad + I
    #         P = P_fn(F)
    #         return P
    #     vmap_stress = jax.vmap(first_PK_stress)
    #     return vmap_stress



    # def stress_strain_fns(self):
    #     E = 70e3
    #     nu = 0.3
    #     mu = E/(2.*(1. + nu))
    #     kappa = E/(3.*(1. - 2.*nu))
    #     def stress(u_grads):
    #         I = np.eye(self.dim)
    #         F = u_grads + I[None, :, :]
    #         I1 = np.trace(np.transpose(F, axes=(0, 2, 1)) @ F, axis1=1, axis2=2)[:, None, None]
    #         F_inv_T = np.transpose(np.linalg.inv(F), axes=(0, 2, 1))
    #         J = np.linalg.det(F)[:, None, None]
    #         P = mu * J**(-2./3.) * (F - 1./3.*I1*F_inv_T) + kappa * J * (J - 1) * F_inv_T
    #         return P

    #     return stress

 
    def stress_strain_fns(self):
        def stress(u_grad):
            return np.sum(self.C * u_grad[:, None, :], axis=(2))
        return stress

    def update_C(self):
        def psi(F):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(disp_grad):
            I = np.eye(self.dim)
            F = disp_grad + I
            P = P_fn(F)
            return P

        def C_fn(disp_grad):
            return jax.jacfwd(first_PK_stress)(disp_grad)
       
        # return jax.vmap(C_fn)(self.disp_grad) 
        return jax.vmap(C_fn)(self.disp_grad).reshape(-1, self.vec*self.dim, self.vec*self.dim)


    def compute_physics(self, sol, u_grads):
        """

        Parameters
        ----------
        u_grads: ndarray
            (num_cells, num_quads, vec, dim)
        """
        # Remark: This is faster than double vmap by reducing ~30% computing time
        u_grads_reshape = u_grads.reshape(-1, self.vec * self.dim)
        vmap_stress = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
        return sigmas


    def jacobi_preconditioner(self):
        C_0 = self.C[:, 0:self.dim, 0:self.dim] # (num_cells*num_quads, dim, dim)
        C_1 = self.C[:, self.dim:2*self.dim, self.dim:2*self.dim]
        C_2 = self.C[:, 2*self.dim:, 2*self.dim:]
        # (num_cells, num_quads, num_nodes, dim) -> (num_cells*num_quads, num_nodes, 1, dim)
        shape_grads_reshape = self.shape_grads.reshape(-1, self.num_nodes, 1, self.dim)
        # (num_cells*num_quads, num_nodes, 1, dim) @ (num_cells*num_quads, 1, dim, dim) @ (num_cells*num_quads, num_nodes, dim, 1)
        # (num_cells*num_quads, num_nodes) -> (num_cells, num_quads, num_nodes) -> (num_cells, num_nodes)
        val_0 = np.sum((shape_grads_reshape @ C_0[:, None, :, :] @ np.transpose(shape_grads_reshape, 
                       axes=(0, 1, 3, 2))).reshape(self.num_cells, self.num_quads, self.num_nodes) * self.JxW[:, :, None], axis=1)
        val_1 = np.sum((shape_grads_reshape @ C_1[:, None, :, :] @ np.transpose(shape_grads_reshape, 
                       axes=(0, 1, 3, 2))).reshape(self.num_cells, self.num_quads, self.num_nodes) * self.JxW[:, :, None], axis=1)
        val_2 = np.sum((shape_grads_reshape @ C_2[:, None, :, :] @ np.transpose(shape_grads_reshape, 
                       axes=(0, 1, 3, 2))).reshape(self.num_cells, self.num_quads, self.num_nodes) * self.JxW[:, :, None], axis=1)
        # (vec, num_cells, num_nodes) -> (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
        vals = np.transpose(np.stack((val_0, val_1, val_2)), axes=(1, 2, 0)).reshape(-1, self.vec)
        jacobi = np.zeros((self.num_total_nodes, self.vec))
        jacobi = jacobi.at[self.cells.reshape(-1)].add(vals)
        return jacobi.reshape(-1)



    def update_disp_grad(self, disp_sol):
        print(f"do update")
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(disp_sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)  
        self.disp_grad = u_grads.reshape(-1, self.vec, self.dim)  # (num_cells*num_quads, vec, dim)  
        self.C = self.update_C()
        print(f"finish update")



def custom_solve(problem, problem_aux):

    start = time.time()
    res_fn = problem.compute_residual
    sol = np.zeros((problem.num_total_nodes, problem.vec))
    dofs = sol.reshape(-1)
    res_fn_dofs = get_flatten_fn(res_fn, problem)

    print(f"compute jacobi preconditioner")
    jacobi = problem.jacobi_preconditioner()
    print(f"np.min(jacobi) = {np.min(jacobi)}, np.max(jacobi) = {np.max(jacobi)}")

    jacobi = assign_ones(jacobi.reshape((sol.shape)), problem).reshape(-1)

    print(f"finish jacobi preconditioner")


    # for ind in range(len(dofs)):
    #     test_vec = np.zeros(problem.num_total_nodes*problem.vec)
    #     test_vec = test_vec.at[ind].set(1.)
    #     print(f"{res_fn_dofs(test_vec)[ind]}, {jacobi[ind]}, ratio = {res_fn_dofs(test_vec)[ind]/jacobi[ind]}")
 


    def jacobi_precond(x):
        return x * (1./jacobi)
 
    print("Start timing")

    res_fn_final = row_elimination(res_fn_dofs, problem)
    b = np.zeros((problem.num_total_nodes, problem.vec))
    b = assign_bc(b, problem).reshape(-1)


    dofs = np.ones_like(dofs)

    dofs = assign_bc(sol, problem).reshape(-1)
    dofs, info = jax.scipy.sparse.linalg.bicgstab(res_fn_final, b, x0=dofs, M=jacobi_precond, tol=1e-10, atol=1e-10, maxiter=10000)

    problem.update_disp_grad(dofs.reshape(sol.shape))

    b_rhs = np.zeros((problem.num_total_nodes, problem.vec))

    b_rhs = -problem_aux.compute_residual(dofs.reshape(sol.shape))
    b_rhs = assign_zero_bc(b_rhs, problem).reshape(-1)
    res_val = np.linalg.norm(b_rhs)
    print(f"Before, res l_2 = {res_val}") 

    tol = 1e-6
    while res_val > tol:
        inc, info = jax.scipy.sparse.linalg.bicgstab(res_fn_final, b_rhs, x0=None, M=jacobi_precond, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
        dofs = dofs + inc
        problem.update_disp_grad(dofs.reshape(sol.shape))

        jacobi = problem.jacobi_preconditioner()
        jacobi = assign_ones(jacobi.reshape((sol.shape)), problem).reshape(-1)


        b_rhs = -problem_aux.compute_residual(dofs.reshape(sol.shape))
        b_rhs = assign_zero_bc(b_rhs, problem).reshape(-1)
        res_val = np.linalg.norm(b_rhs)
        print(f"res l_2 = {res_val}") 

    sol = dofs.reshape(sol.shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s]")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol


def debug_problem():
    problem_name = "debug"
    num_hex = 10

    meshio_mesh = box_mesh(100, 20, 100, 10., 2., 10.)

    # meshio_mesh = cylinder_mesh()

    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], 10., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1*10.

    dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          zero_dirichlet_val, zero_dirichlet_val, dirichlet_val]]


    problem_aux = HyperElasticity(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info)
    # sol = solver(problem_aux, use_linearization_guess=True)

    problem = LinearizedHyperElasticity(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info)
    sol = custom_solve(problem, problem_aux)

    # sol = solver(problem)


    # jax_vtu_path = f"src/fem/data/vtk/{problem.name}/disp.vtu"
    # save_sol(problem, sol, jax_vtu_path)


if __name__=="__main__":
    debug_problem()
