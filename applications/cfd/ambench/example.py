import sys
sys.path.append('../../../')
import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import glob
from functools import partial

from jax_am.cfd.cfd_am import mesh3d, AM_3d
from jax_am.cfd.json_parser import cfd_parse

from jax_am.common import box_mesh


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

jax.config.update("jax_enable_x64", True)


def integrator():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')

    domain_x = 3.e-3
    domain_y = 4.e-4
    domain_z = 4.e-4
    Nx = 300
    Ny = 100 
    Nz = 100 

    cfd_args = cfd_parse(os.path.join(crt_file_path, 'cfd_params.json'))

    mesh = mesh3d([domain_x, domain_y, domain_z], [Nx, Ny, Nz])
    Nx_local = 80
    Ny_local = 50
    Nz_local = 50

    mesh_local = mesh3d([Nx_local/Nx*domain_x,Ny_local/Ny*domain_y,Nz_local/Nz*domain_z], 
                        [Nx_local,Ny_local,Nz_local])

    meshio_mesh = box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z)


    cfd_args['mesh'] = mesh
    cfd_args['mesh_local'] = mesh_local
    cfd_args['meshio_mesh'] = meshio_mesh
    cfd_args['cp'] = lambda T: 0.2174*np.clip(T,300,1609)+370.
    cfd_args['latent_heat'] = 270000.
    cfd_args['k'] = lambda T: 0.01674*np.clip(T,300,1563)+3.9
    cfd_args['heat_source'] = 1
    cfd_args['phi'] = 4e-7
    cfd_args['data_dir'] = data_dir

    cfd_solver = AM_3d(cfd_args)
    ts = np.arange(0., cfd_args['t_OFF'] + 1e-10, cfd_args['dt'])
    cfd_solver.write_sols(0)
    for (i, t_crt) in enumerate(ts[1:]):
        cfd_solver.time_integration()

        if (i + 1) % cfd_args['check_sol_interval'] == 0:
            cfd_solver.inspect_sol(i + 1, len(ts[1:]))

        if (i + 1) % cfd_args['write_sol_interval'] == 0:
            cfd_solver.write_sols(i + 1)


if __name__=="__main__":
    integrator()
