import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
from functools import partial
from modules.phase_field.integrator import MultiVarSolver
from modules.phase_field.utils import Field
from modules.phase_field.yaml_parse import args


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def set_params():
    '''
    If a certain parameter is not set, a default value will be used according to the YAML file.
    '''
    args['case'] = 'fd_example'
    args['num_grains'] = 20000
    args['domain_x'] = 1.
    args['domain_y'] = 0.2
    args['domain_z'] = 0.1
    args['r_beam'] = 0.03
    args['power'] = 100
    args['write_sol_interval'] = 1000

    # If args['ad_hoc'] = 0.1, we get curved long grains. 
    args['ad_hoc'] = 1.


def pre_processing():
    '''
    We use Neper to generate polycrystal structure.
    Neper has two major functions: generate a polycrystal structure, and mesh it.
    See https://neper.info/ for more information.
    '''
    set_params()
    neper_path = os.path.join(args['root_path'], f"data/neper/{args['case']}")
    os.makedirs(neper_path, exist_ok=True)
    os.system(f'''neper -T -n {args['num_grains']} -id 1 -regularization 0 -domain "cube({args['domain_x']},\
               {args['domain_y']},{args['domain_z']})" \
                -o {neper_path}/domain -format tess,obj,ori''')
    os.system(f"neper -T -loadtess {neper_path}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
    os.system(f"neper -M -rcl 1 -elttype hex -faset faces {neper_path}/domain.tess")

    # Optional, write the Neper files to local for visualization
    polycrystal = Field()
    polycrystal.write_vtu_files()


@partial(jax.jit, static_argnums=(1,))
def get_T_laser(t, polycrystal):
    '''
    Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
    '''
    centroids = polycrystal.centroids

    Q = 25
    alpha = 5.2
    kappa = 2.7*1e-2
    x0 = 0.2*args['domain_x']
    vel = 0.6*args['domain_x'] / args['laser_path']['time'][-1]

    X = centroids[:, 0] - x0 - vel * t
    Y = centroids[:, 1] - 0.5*args['domain_y']
    Z = centroids[:, 2] - args['domain_z']
    R = np.sqrt(X**2 + Y**2 + Z**2)
    T = args['T_ambient'] + Q / (2 * np.pi * kappa) / R * np.exp(-vel / (2*alpha) * (R + X))

    # TODO: Not quite elegant here
    T = np.where(T > 2000., 2000., T)

    return T[:, None]


def run():
    set_params()
    solver = MultiVarSolver(get_T_laser)
    solver.solve()


def post_processing():
    set_params()
    polycrystal = Field()
    cell_ori_inds_3D = polycrystal.convert_to_3D_images()


if __name__ == "__main__":
    # pre_processing()
    run()
    # post_processing()
