import numpy as onp
import jax
import jax.numpy as np
import time
import os
import glob
from functools import partial
from src.fem.generate_mesh import box_mesh, cylinder_mesh
from src.fem.solver import solver, assign_bc, get_A_fn_linear_fn, row_elimination, get_flatten_fn
from src.fem.jax_fem import Mesh, Laplace, HyperElasticity, LinearElasticity
from src.fem.utils import save_sol


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def debug_problem():
    problem_name = "debug"
    num_hex = 10

    meshio_mesh = box_mesh(100, 20, 100, 10., 2., 10.)
    # meshio_mesh = box_mesh(10, 40, 10, 10., 2., 10.)
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


    problem = HyperElasticity(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem, precond=True)


if __name__=="__main__":
    debug_problem()
