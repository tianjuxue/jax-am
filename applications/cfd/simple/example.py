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

    domain_x = 1.e-3
    domain_y = 2.e-4
    domain_z = 1.e-4
    Nx = 464
    Ny = 93 
    Nz = 46 

    mesh = mesh3d([domain_x, domain_y, domain_z], [Nx, Ny, Nz])
    # mesh_local = mesh
    mesh_local = mesh3d([0.75*domain_x, 0.5*domain_y, 0.4*domain_z], 
                        [round(0.75*Nx), round(0.5*Ny), round(0.4*Nz)])
    meshio_mesh = box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z)

    cfd_args = cfd_parse(os.path.join(crt_file_path, 'cfd_params.json'))
    cfd_args['mesh'] = mesh
    cfd_args['mesh_local'] = mesh_local
    cfd_args['meshio_mesh'] = meshio_mesh
    cfd_args['cp'] = lambda T: (0.2441*np.clip(T,300,1563)+338.39) 
    cfd_args['k'] = lambda T: 0.0163105*np.clip(T,300,1563)+4.5847
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
