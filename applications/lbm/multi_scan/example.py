import numpy as onp
import jax.numpy as np
import jax
import os

from jax_am.lbm.core import simulation
from jax_am.lbm.utils import ST, shape_wrapper, compute_cell_centroid, to_id_xyz
from jax_am.common import make_video, json_parse, box_mesh

onp.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

crt_file_path = os.path.dirname(__file__)
data_dir = os.path.join(crt_file_path, 'data')



def case_study():
    def initialize_phase(lattice_id, cell_centroids):
        id_x, id_y, id_z = to_id_xyz(lattice_id, lbm_args)
        plate_z = 0.8 * domain_z
        flag =  cell_centroids[lattice_id, 2] < plate_z
        tmp = np.where(flag, ST.LIQUID, ST.GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, ST.WALL, tmp)


    lbm_args = json_parse(os.path.join(crt_file_path, 'lbm_params.json'))
    Nx, Ny, Nz = lbm_args['Nx']['value'], lbm_args['Ny']['value'], lbm_args['Nz']['value']
    domain_x, domain_y, domain_z = Nx, Ny, Nz
    meshio_mesh = box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z)
    initialize_phase_vmap = shape_wrapper(jax.vmap(initialize_phase, in_axes=(0, None)), lbm_args)
    initial_phase = initialize_phase_vmap(np.arange(Nx*Ny*Nz), compute_cell_centroid(meshio_mesh))
    simulation(lbm_args, data_dir, meshio_mesh, initial_phase)
 

if __name__== "__main__":
    case_study()
