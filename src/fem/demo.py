import jax
import jax.numpy as np
import os
from src.fem.jax_fem import Mesh, LinearElasticity
from src.fem.solver import solver
from src.fem.generate_mesh import box_mesh
from src.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def problem():
    """Can be used to test the memory limit of JAX-FEM
    """
    problem_name = f'linear_elasticity'
    meshio_mesh = box_mesh(100, 100, 100)
    # meshio_mesh = box_mesh(10, 10, 10)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
 
    problem = LinearElasticity(problem_name, mesh, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, linear=True, precond=False)
    vtk_path = f"src/fem/data/vtk/{problem_name}/u.vtu"
    save_sol(problem, sol, vtk_path)


if __name__ == "__main__":
    problem()
