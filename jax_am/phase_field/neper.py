import os
from jax_am.phase_field.utils import Field


def pre_processing(pf_args):
    """We use Neper to generate polycrystal structure.
    Neper has two major functions: generate a polycrystal structure, and mesh it.
    See https://neper.info/ for more information.
    """
    neper_path = os.path.join(pf_args['data_dir'], f"neper")
    os.makedirs(neper_path, exist_ok=True)
    os.system(f'''neper -T -n {pf_args['num_grains']} -id 1 -regularization 0 -domain "cube({pf_args['domain_x']},\
               {pf_args['domain_y']},{pf_args['domain_z']})" \
                -o {neper_path}/domain -format tess,obj,ori''')
    os.system(f"neper -T -loadtess {neper_path}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
    os.system(f"neper -M -rcl 1 -elttype hex -faset faces {neper_path}/domain.tess")

