import numpy as onp
import jax
import jax.numpy as np
from modules.fem.apps.multi_scale.utils import flat_to_tensor, tensor_to_flat

from modules.fem.generate_mesh import box_mesh
from modules.fem.jax_fem import Mesh, Laplace
from modules.fem.solver import solver, assign_bc
from modules.fem.utils import save_sol
from modules.fem.apps.multi_scale.arguments import args
from modules.fem.apps.multi_scale.utils import tensor_to_flat, tensor_to_flat
from modules.fem.apps.multi_scale.trainer import H_to_C
from modules.fem.apps.multi_scale.fem_model import HyperElasticity


# def homogenization_problem():
#     problem_name = "homogenization_debug"

#     L = args.L
 
#     # meshio_mesh = box_mesh(10, 2, 10, 10*L, 2*L, 10*L)

#     meshio_mesh = box_mesh(10, 10, 10, L, L, L)

#     jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

#     def corner(point):
#         return np.isclose(np.linalg.norm(point), 0., atol=1e-5)

#     def bottom(point):
#         return np.isclose(point[2], 0., atol=1e-5)

#     # def top(point):
#     #     return np.isclose(point[2], 10*L, atol=1e-5)

#     def top(point):
#         return np.isclose(point[2], L, atol=1e-5)

#     dirichlet_corner = lambda _: 0. 
#     dirichlet_bottom_z = lambda _: 0. 
#     # dirichlet_top_z = lambda _: 0.025*10*L
#     dirichlet_top_z = lambda _: 0.025*L
#     # dirichlet_top_z = lambda _: 0.1*L


#     location_fns = [corner]*3 + [bottom] + [top]
#     value_fns = [dirichlet_corner]*3 + [dirichlet_bottom_z] + [dirichlet_top_z]
#     vecs = [0, 1, 2, 2, 2]
#     dirichlet_bc_info = [location_fns, vecs, value_fns]

#     # problem = HyperElasticity(f"{problem_name}", jax_mesh, mode='nn', dirichlet_bc_info=dirichlet_bc_info)
#     problem = HyperElasticity(f"{problem_name}", jax_mesh,  mode='dns', dirichlet_bc_info=dirichlet_bc_info)


#     sol = np.zeros((problem.num_total_nodes, problem.vec))
#     sol = assign_bc(sol, problem)
#     energy = problem.compute_energy(sol)
#     print(f"Initial energy = {energy}")

#     sol = solver(problem, use_linearization_guess=True)


#     # dirichlet_bc_info[-1][-1] = lambda _: 0.05*L
#     # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
#     # sol = solver(problem, initial_guess=sol, use_linearization_guess=False)


#     # dirichlet_bc_info[-1][-1] = lambda _: 0.075*L
#     # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
#     # sol = solver(problem, initial_guess=sol, use_linearization_guess=False)

#     # dirichlet_bc_info[-1][-1] = lambda _: 0.1*L
#     # problem.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
#     # sol = solver(problem, initial_guess=sol, use_linearization_guess=False)



#     energy = problem.compute_energy(sol)
#     print(f"Final energy = {energy}")

#     jax_vtu_path = f"modules/fem/apps/multi_scale/data/vtk/{problem.name}/sol_disp.vtu"
#     save_sol(problem, sol, jax_vtu_path)

#     return problem


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



def homogenization_problem():
    problem_name = "homogenization_debug"
    args.units_x = 2
    args.units_y = 2
    args.units_z = 2
    L = args.L
    meshio_mesh = box_mesh(args.num_hex*args.units_x, args.num_hex*args.units_y, args.num_hex*args.units_z,
                           L*args.units_x, L*args.units_y, L*args.units_z)

    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def corner(point):
        return np.isclose(np.linalg.norm(point), 0., atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], args.units_z*L, atol=1e-5)

    dirichlet_zero = lambda _: 0. 
    # dirichlet_top_z = lambda _: 0.025*args.units_z*L
    dirichlet_top_z = lambda _: 0.1*args.units_z*L
  
 
    # dirichlet_bc_info = [[corner]*3 + [bottom] + [top], 
    #                      [0, 1, 2, 2, 2]
    #                      [dirichlet_zero]*3 + [dirichlet_zero] + [dirichlet_top_z], ]


    dirichlet_bc_info = [[bottom, bottom, bottom, top, top, top], 
                         [0, 1, 2, 0, 1, 2], 
                         [dirichlet_zero, dirichlet_zero, dirichlet_zero, 
                          dirichlet_zero, dirichlet_zero, dirichlet_top_z]]


    problem = HyperElasticity(f"{problem_name}", jax_mesh, mode='nn', dirichlet_bc_info=dirichlet_bc_info)
    # problem = HyperElasticity(f"{problem_name}", jax_mesh,  mode='dns', dirichlet_bc_info=dirichlet_bc_info)

    sol = np.zeros((problem.num_total_nodes, problem.vec))
    dofs = sol.reshape(-1)
    # dofs = assign_bc(dofs, problem)
    energy = problem.compute_energy(dofs.reshape(sol.shape))
    print(f"Initial energy = {energy}")

    # sol = solver(problem, use_linearization_guess=True)

    sol = solver(problem)

    energy = problem.compute_energy(sol)
    print(f"Final energy = {energy}")
    # traction = problem.compute_traction(top, sol)
    # print(f"traction = {traction}")

    jax_vtu_path = f"modules/fem/apps/multi_scale/data/vtk/{problem.name}/sol_disp.vtu"
    save_sol(problem, sol, jax_vtu_path)

    return problem


if __name__=="__main__":
    # debug()
    homogenization_problem()
