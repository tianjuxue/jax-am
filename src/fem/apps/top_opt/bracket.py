import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import os
import meshio
import time

from src.fem.generate_mesh import box_mesh
from src.fem.jax_fem import Mesh, Laplace, LinearPoisson
from src.fem.solver import solver
from src.fem.utils import save_sol

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def problem():
    problem_name = f'bracket'
    root_path = f"src/fem/apps/top_opt/data/"
    mesh_file = os.path.join(root_path, f"abaqus/BRACKET.inp")

    meshio_mesh = meshio.read(mesh_file)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

 
    problem = LinearPoisson(problem_name, jax_mesh, dirichlet_bc_info=[[],[],[]])

    files = glob.glob(os.path.join(root_path, f"vtk/{problem_name}/*"))
    for f in files:
        os.remove(f)
 
    sol = solver(problem)
    vtk_path = os.path.join(root_path, f"vtk/{problem_name}/u.vtu")
    save_sol(problem, sol, vtk_path)

 
if __name__ == "__main__":
    problem()
