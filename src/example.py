import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
from src.integrator import MultiVarSolver
from src.utils import Field
from src.yaml_parse import args


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
    
    # args['ad_hoc'] = 0.1


def generate_neper():
    '''
    We use Neper to generate polycrystal structure.
    Neper has two major functions: generate a polycrystal structure, and mesh it.
    See https://neper.info/ for more information.
    '''
    set_params()
    os.system(f'''neper -T -n {args['num_grains']} -id 1 -regularization 0 -domain "cube({args['domain_x']},\
               {args['domain_y']},{args['domain_z']})" \
                -o post-processing/neper/{args['case']}/domain -format tess,obj,ori''')
    os.system(f"neper -T -loadtess post-processing/neper/{args['case']}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
    os.system(f"neper -M -rcl 1 -elttype hex -faset faces post-processing/neper/{args['case']}/domain.tess")

    # Optional, write the Neper files to local for visualization
    polycrystal = Field()
    polycrystal.write_vtu_files()


def run():
    set_params()
    solver = MultiVarSolver()
    solver.solve()


if __name__ == "__main__":
    # generate_neper()
    run()
