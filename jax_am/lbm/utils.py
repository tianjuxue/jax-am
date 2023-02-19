import numpy as onp
import jax.numpy as np
import jax
import os 
import glob
import meshio

class State:
    LIQUID = 0
    LG = 1
    GAS = 2 
    WALL =  3


ST = State()


def clean_sols(data_dir):
    vtk_dir = os.path.join(data_dir, 'vtk')
    os.makedirs(vtk_dir, exist_ok=True)
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)


def compute_cell_centroid(meshio_mesh):
    points = meshio_mesh.points
    cells = meshio_mesh.cells_dict['hexahedron']
    cell_centroids = np.mean(points[cells], axis=1)
    return cell_centroids


def to_id(idx, idy, idz, lbm_args):
    return idx * lbm_args['Ny']['value'] * lbm_args['Nz']['value'] + idy * lbm_args['Nz']['value'] + idz


def to_id_xyz(lattice_id, lbm_args):
    id_z = lattice_id % lbm_args['Nz']['value']
    lattice_id = lattice_id // lbm_args['Nz']['value']
    id_y = lattice_id % lbm_args['Ny']['value']
    id_x = lattice_id // lbm_args['Ny']['value']    
    return id_x, id_y, id_z 


def shape_wrapper(f, lbm_args):
    def shape_wrapper(*args):
        return jax.tree_util.tree_map(lambda x: x.reshape((lbm_args['Nx']['value'], 
            lbm_args['Ny']['value'], lbm_args['Nz']['value']) + x.shape[1:]), f(*args))
    return shape_wrapper