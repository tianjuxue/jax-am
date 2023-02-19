import numpy as onp
import jax.numpy as np
import jax
import os

from jax_am.lbm.core import simulation
from jax_am.lbm.utils import ST, shape_wrapper, compute_cell_centroid, to_id_xyz
from jax_am.common import make_video, json_parse, box_mesh


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def case_study():
    # h=1/2gt^2, h=0.5[m], t=0.319[s], simulation total t=0.4[s], output 40 steps, should see object on ground at the 16 step.
    def initialize_phase(lattice_id, cell_centroids):
        id_x, id_y, id_z = to_id_xyz(lattice_id, lbm_args)
        flag_x = np.logical_and(cell_centroids[lattice_id, 0] > 0.2 * domain_x, cell_centroids[lattice_id, 0] < 0.8 * domain_x)
        flag_y = np.logical_and(cell_centroids[lattice_id, 1] > 0.2 * domain_y, cell_centroids[lattice_id, 1] < 0.8 * domain_y)
        flag_z = np.logical_and(cell_centroids[lattice_id, 2] > 0.5 * domain_z, cell_centroids[lattice_id, 2] < 0.8 * domain_z) 
        flag = np.logical_and(np.logical_and(flag_x, flag_y), flag_z)
        tmp = np.where(flag, ST.LIQUID, ST.GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, ST.WALL, tmp)

    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    lbm_args = json_parse(os.path.join(crt_file_path, 'lbm_params.json'))
    Nx, Ny, Nz = lbm_args['Nx']['value'], lbm_args['Ny']['value'], lbm_args['Nz']['value']
    domain_x, domain_y, domain_z = Nx, Ny, Nz
    meshio_mesh = box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z)
    initialize_phase_vmap = shape_wrapper(jax.vmap(initialize_phase, in_axes=(0, None)), lbm_args)
    initial_phase = initialize_phase_vmap(np.arange(Nx*Ny*Nz), compute_cell_centroid(meshio_mesh))
    simulation(lbm_args, data_dir, meshio_mesh, initial_phase, fluid_only=True)
 

if __name__== "__main__":
    case_study()
