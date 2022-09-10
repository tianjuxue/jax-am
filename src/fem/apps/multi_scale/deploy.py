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
from src.fem.apps.multi_scale.trainer import H_to_C
from src.fem.apps.multi_scale.fem_model import HyperElasticity


def homogenization_problem():
    problem_name = "homogenization_debug"

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
    dirichlet_top_z = lambda _: 0.025*L
    # dirichlet_top_z = lambda _: 0.1*L


    location_fns = [corner]*3 + [bottom] + [top]
    value_fns = [dirichlet_corner]*3 + [dirichlet_bottom_z] + [dirichlet_top_z]
    vecs = [0, 1, 2, 2, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    # problem = HyperElasticity(f"{problem_name}", jax_mesh, mode='nn', dirichlet_bc_info=dirichlet_bc_info)
    problem = HyperElasticity(f"{problem_name}", jax_mesh,  mode='dns', dirichlet_bc_info=dirichlet_bc_info)


    sol = np.zeros((problem.num_total_nodes, problem.vec))
    sol = assign_bc(sol, problem)
    energy = problem.compute_energy(sol)
    print(f"Initial energy = {energy}")

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

    H_flat = tensor_to_flat(H_bar)
    C_flat, _ = H_to_C(H_flat)

    hyperparam = 'default'
    nn_batch_forward = get_nn_batch_forward(hyperparam)
    energy = nn_batch_forward(C_flat[None, :])
    print(energy)


if __name__=="__main__":
    homogenization_problem()
    # debug()
