import numpy as onp
import jax
import jax.numpy as np
from src.fem.apps.multi_scale.utils import flat_to_tensor, tensor_to_flat

from src.fem.generate_mesh import box_mesh
from src.fem.jax_fem import Mesh, Laplace
from src.fem.solver import solver, assign_bc
from src.fem.utils import save_sol
from src.fem.apps.multi_scale.arguments import args
from src.fem.apps.multi_scale.utils import tensor_to_flat, tensor_to_flat
from src.fem.apps.multi_scale.trainer import get_nn_batch_forward, H_to_C



class HyperElasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info=None, periodic_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, periodic_bc_info, neumann_bc_info, source_info)

    def compute_physics(self, sol, u_grads):
        u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
        vmap_stress, _ = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
        return sigmas

    def compute_energy(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        F_reshape = u_grads.reshape(-1, self.vec, self.dim) + np.eye(self.dim)[None, :, :]

        H_reshape = u_grads.reshape(-1, self.vec, self.dim)
        # print(H_reshape)
        print(np.mean(H_reshape, axis=0))

        _, vmap_energy  = self.stress_strain_fns()
        psi = vmap_energy(F_reshape).reshape(u_grads.shape[:2]) # (num_cells, num_quads)
        energy = np.sum(psi * self.JxW)
        return energy


class Homogenization(HyperElasticity):
    def __init__(self, name, mesh, dirichlet_bc_info=None, periodic_bc_info=None, neumann_bc_info=None, source_info=None):
        super().__init__(name, mesh, dirichlet_bc_info, periodic_bc_info, neumann_bc_info, source_info)
        self.nn_batch_forward = get_nn_batch_forward()

    def stress_strain_fns(self):
        def psi(F):
            C = F.T @ F
            C_flat = tensor_to_flat(C)
            energy = self.nn_batch_forward(C_flat[None, :])[0]
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        vmap_stress = jax.vmap(first_PK_stress)
        vmap_energy = jax.vmap(psi)
        return vmap_stress, vmap_energy


class Bulk(HyperElasticity):
    def __init__(self, name, mesh, dirichlet_bc_info=None, periodic_bc_info=None, neumann_bc_info=None, source_info=None):
        super().__init__(name, mesh, dirichlet_bc_info, periodic_bc_info, neumann_bc_info, source_info)
        self.physical_quad_points = self.get_physical_quad_points()
        self.E, self.nu = self.compute_moduli()
 
    def stress_strain_fns(self):
        def psi(F, E, nu):
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, E, nu)
            return P
        vmap_stress = lambda x: jax.vmap(first_PK_stress)(x, self.E, self.nu)
        vmap_energy = lambda x: jax.vmap(psi)(x, self.E, self.nu)
        return vmap_stress, vmap_energy

    def compute_moduli(self):
        # TODO: a lot of redundant code here
        center = np.array([args.L/2., args.L/2., args.L/2.])
        def E_map(point):
            E = np.where(np.max(np.absolute(point - center)) < args.L*0.3, 1e2, 1e2) # 1e3, 1e2
            return E

        def nu_map(point):
            nu = np.where(np.max(np.absolute(point - center)) < args.L*0.3, 0.4, 0.4) # 0.3, 0.4
            return nu

        E = jax.vmap(jax.vmap(E_map))(self.physical_quad_points).reshape(-1)
        nu = jax.vmap(jax.vmap(nu_map))(self.physical_quad_points).reshape(-1)

        return E, nu



def homogenization_problem():
    # problem_name = "homogenization"

    problem_name = "debug"

    L = args.L
 
    # meshio_mesh = box_mesh(10, 2, 10, 10*L, 2*L, 10*L)

    meshio_mesh = box_mesh(10, 10, 10, L, L, L)

    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def corner(point):
        return np.isclose(np.linalg.norm(point), 0., atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    # def top(point):
    #     return np.isclose(point[2], 10*L, atol=1e-5)

    def top(point):
        return np.isclose(point[2], L, atol=1e-5)

    dirichlet_corner = lambda _: 0. 
    dirichlet_bottom_z = lambda _: 0. 
    # dirichlet_top_z = lambda _: 0.025*10*L
    # dirichlet_top_z = lambda _: 0.025*L
    dirichlet_top_z = lambda _: 0.1*L


    location_fns = [corner]*3 + [bottom] + [top]
    value_fns = [dirichlet_corner]*3 + [dirichlet_bottom_z] + [dirichlet_top_z]
    vecs = [0, 1, 2, 2, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    # problem = Homogenization(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info)
    problem = Bulk(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info)



    # sol = np.zeros((problem.num_total_nodes, problem.vec))
    # sol = assign_bc(sol, problem)
    # energy = problem.compute_energy(sol)
    # print(f"Initial energy = {energy}")

    sol = solver(problem, use_linearization_guess=True)


    # dirichlet_bc_info[-1][-1] = lambda _: 0.05*L
    # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    # sol = solver(problem, initial_guess=sol, use_linearization_guess=False)


    # dirichlet_bc_info[-1][-1] = lambda _: 0.075*L
    # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    # sol = solver(problem, initial_guess=sol, use_linearization_guess=False)

    # dirichlet_bc_info[-1][-1] = lambda _: 0.1*L
    # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    # sol = solver(problem, initial_guess=sol, use_linearization_guess=False)



    energy = problem.compute_energy(sol)
    print(f"Final energy = {energy}")

    jax_vtu_path = f"src/fem/apps/multi_scale/data/vtk/{problem.name}/sol_disp.vtu"
    save_sol(problem, sol, jax_vtu_path)

    return problem


def debug():
    H_bar = np.array([[-0.009, 0., 0.],
                      [0., -0.009, 0.],
                      [0., 0., 0.025]])

    # H_bar = np.array([[-0.0087985632, -0.0010044279,  0.0020509221],
    #                   [-0.0010366058, -0.0100220639, -0.0009761167],
    #                   [ 0.000076172,  -0.0000362885,  0.025       ]])

    # H_bar_zero = np.array([[0., 0., 0.],
    #                   [0., 0., 0.],
    #                   [0., 0., 0.]])


    H_flat = tensor_to_flat(H_bar)
    C_flat, _ = H_to_C(H_flat)

    nn_batch_forward = get_nn_nn_batch_forward()
    energy = nn_batch_forward(C_flat[None, :])
    print(energy)


if __name__=="__main__":
    homogenization_problem()
    # debug()
